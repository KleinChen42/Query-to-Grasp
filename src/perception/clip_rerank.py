"""CLIP/OpenCLIP reranking for 2D detection candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from src.env.camera_utils import normalize_rgb
from src.perception.grounding_dino import DetectionCandidate

LOGGER = logging.getLogger(__name__)

ClipScoreFn = Callable[[Sequence[np.ndarray], Sequence[str]], np.ndarray]


@dataclass
class RankedCandidate:
    """A detection candidate scored by CLIP and a simple fused 2D score."""

    box_xyxy: np.ndarray
    det_score: float
    clip_score: float
    fused_2d_score: float
    phrase: str
    rank: int = 0
    image_crop_path: str | None = None
    source: str = "clip_rerank"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable candidate."""

        return {
            "box_xyxy": np.asarray(self.box_xyxy, dtype=float).tolist(),
            "det_score": float(self.det_score),
            "clip_score": float(self.clip_score),
            "fused_2d_score": float(self.fused_2d_score),
            "phrase": self.phrase,
            "rank": int(self.rank),
            "image_crop_path": self.image_crop_path,
            "source": self.source,
            "metadata": self.metadata,
        }


def rerank_candidates_with_clip(
    image: np.ndarray,
    candidates: Sequence[DetectionCandidate],
    text_prompt: str | Sequence[str],
    detector_weight: float = 0.5,
    clip_weight: float = 0.5,
    score_fn: ClipScoreFn | None = None,
    clip_model: Any | None = None,
    crop_output_dir: str | Path | None = None,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: str | None = None,
) -> list[RankedCandidate]:
    """Rerank candidates by combining detector confidence and CLIP similarity."""

    rgb = normalize_rgb(image)
    prompts = _normalize_prompts(text_prompt)
    crops: list[np.ndarray] = []
    valid_candidates: list[DetectionCandidate] = []
    for candidate in candidates:
        crop = crop_candidate(rgb, candidate.box_xyxy)
        if crop.size == 0:
            LOGGER.debug("Skipping candidate with empty crop: %s", candidate.box_xyxy)
            continue
        crops.append(crop)
        valid_candidates.append(candidate)

    if not valid_candidates:
        return []

    scorer = score_fn or (clip_model.score if clip_model is not None else _get_cached_open_clip_scorer(model_name, pretrained, device).score)
    raw_scores = np.asarray(scorer(crops, prompts), dtype=np.float32)
    clip_scores = _collapse_prompt_scores(raw_scores, expected_count=len(valid_candidates))

    crop_paths = _save_crops_if_requested(crops, valid_candidates, crop_output_dir)
    ranked: list[RankedCandidate] = []
    for index, (candidate, clip_score) in enumerate(zip(valid_candidates, clip_scores)):
        det_score = float(candidate.det_score)
        fused_score = detector_weight * det_score + clip_weight * float(clip_score)
        ranked.append(
            RankedCandidate(
                box_xyxy=np.asarray(candidate.box_xyxy, dtype=np.float32),
                det_score=det_score,
                clip_score=float(clip_score),
                fused_2d_score=float(fused_score),
                phrase=candidate.phrase,
                image_crop_path=crop_paths[index],
                metadata={
                    "detector_weight": detector_weight,
                    "clip_weight": clip_weight,
                    "prompts": prompts,
                },
            )
        )

    ranked.sort(key=lambda item: (-item.fused_2d_score, -item.clip_score, -item.det_score, item.phrase))
    for rank, candidate in enumerate(ranked):
        candidate.rank = rank
    return ranked


def crop_candidate(image: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    """Clip a candidate box to image bounds and return its RGB crop."""

    rgb = normalize_rgb(image)
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = _box_to_int_bounds(box_xyxy, width=width, height=height)
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return rgb[y0:y1, x0:x1].copy()


class OpenCLIPScorer:
    """Small cached OpenCLIP scoring wrapper."""

    def __init__(self, model_name: str, pretrained: str, device: str | None = None) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "CLIP reranking requires `open_clip_torch` and `torch`. Install them "
                "with `pip install open_clip_torch torch`, or pass a custom score_fn "
                "for tests/mocks."
            ) from exc

        self.open_clip = open_clip
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def score(self, crops: Sequence[np.ndarray], prompts: Sequence[str]) -> np.ndarray:
        """Return CLIP cosine-like scores scaled to ``[0, 1]``."""

        from PIL import Image

        if not crops:
            return np.empty((0, len(prompts)), dtype=np.float32)

        image_tensors = [
            self.preprocess(Image.fromarray(normalize_rgb(crop), mode="RGB"))
            for crop in crops
        ]
        image_batch = self.torch.stack(image_tensors).to(self.device)
        text_tokens = self.tokenizer(list(prompts)).to(self.device)

        with self.torch.no_grad():
            image_features = self.model.encode_image(image_batch)
            text_features = self.model.encode_text(text_tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ text_features.T

        scores = ((similarities.detach().cpu().numpy() + 1.0) * 0.5).astype(np.float32)
        return np.clip(scores, 0.0, 1.0)


@lru_cache(maxsize=4)
def _get_cached_open_clip_scorer(model_name: str, pretrained: str, device: str | None) -> OpenCLIPScorer:
    return OpenCLIPScorer(model_name=model_name, pretrained=pretrained, device=device)


def _normalize_prompts(text_prompt: str | Sequence[str]) -> list[str]:
    if isinstance(text_prompt, str):
        prompts = [text_prompt]
    else:
        prompts = list(text_prompt)
    prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
    if not prompts:
        raise ValueError("At least one CLIP text prompt is required.")
    return prompts


def _collapse_prompt_scores(scores: np.ndarray, expected_count: int) -> np.ndarray:
    if scores.ndim == 1:
        if scores.shape[0] != expected_count:
            raise ValueError(f"score_fn returned {scores.shape[0]} scores for {expected_count} candidates.")
        return np.clip(scores, 0.0, 1.0)
    if scores.ndim == 2:
        if scores.shape[0] != expected_count:
            raise ValueError(f"score_fn returned shape {scores.shape}; expected first dimension {expected_count}.")
        return np.clip(np.max(scores, axis=1), 0.0, 1.0)
    raise ValueError(f"score_fn must return a 1D or 2D array, got shape {scores.shape}.")


def _save_crops_if_requested(
    crops: Sequence[np.ndarray],
    candidates: Sequence[DetectionCandidate],
    crop_output_dir: str | Path | None,
) -> list[str | None]:
    if crop_output_dir is None:
        return [candidate.image_crop_path for candidate in candidates]

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to save candidate crops. Install it with `pip install pillow`.") from exc

    output_dir = Path(crop_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str | None] = []
    for index, crop in enumerate(crops):
        path = output_dir / f"candidate_{index:03d}.png"
        Image.fromarray(normalize_rgb(crop), mode="RGB").save(path)
        paths.append(str(path))
    return paths


def _box_to_int_bounds(box_xyxy: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    box = np.asarray(box_xyxy, dtype=float)
    if box.shape != (4,):
        raise ValueError(f"Box must have shape (4,), got {box.shape}")
    x0 = int(np.floor(np.clip(min(box[0], box[2]), 0, width)))
    y0 = int(np.floor(np.clip(min(box[1], box[3]), 0, height)))
    x1 = int(np.ceil(np.clip(max(box[0], box[2]), 0, width)))
    y1 = int(np.ceil(np.clip(max(box[1], box[3]), 0, height)))
    return x0, y0, x1, y1

