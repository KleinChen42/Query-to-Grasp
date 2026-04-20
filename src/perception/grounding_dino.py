"""GroundingDINO candidate detection adapter.

The wrapper supports two common installation styles:

1. Hugging Face Transformers GroundingDINO models, selected by ``backend="hf"``.
2. The original GroundingDINO repository API, selected by
   ``backend="groundingdino"`` and supplied with config/checkpoint paths.

Imports are lazy so tests and downstream modules can import this file without
large model dependencies installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import logging
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from src.env.camera_utils import normalize_rgb

LOGGER = logging.getLogger(__name__)


@dataclass
class DetectionCandidate:
    """A 2D open-vocabulary detection candidate."""

    box_xyxy: np.ndarray
    det_score: float
    phrase: str
    image_crop_path: str | None = None
    source: str = "groundingdino"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable candidate."""

        return {
            "box_xyxy": np.asarray(self.box_xyxy, dtype=float).tolist(),
            "det_score": float(self.det_score),
            "phrase": self.phrase,
            "image_crop_path": self.image_crop_path,
            "source": self.source,
            "metadata": self.metadata,
        }


class GroundingDINOAdapter(Protocol):
    """Protocol for detector backends."""

    def predict(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
        top_k: int,
    ) -> list[DetectionCandidate]:
        """Return candidate detections."""


def detect_candidates(
    image: np.ndarray,
    text_prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    top_k: int = 20,
    model: GroundingDINOAdapter | None = None,
    save_overlay_path: str | Path | None = None,
    backend: str = "auto",
    model_config_path: str | Path | None = None,
    model_checkpoint_path: str | Path | None = None,
    model_id: str = "IDEA-Research/grounding-dino-tiny",
    device: str | None = None,
    mock_box_position: str = "center",
) -> list[DetectionCandidate]:
    """Run GroundingDINO and return sorted ``xyxy`` detection candidates."""

    rgb = normalize_rgb(image)
    detector = model or _get_cached_adapter(
        backend=backend,
        model_config_path=str(model_config_path) if model_config_path is not None else None,
        model_checkpoint_path=str(model_checkpoint_path) if model_checkpoint_path is not None else None,
        model_id=model_id,
        device=device,
        mock_box_position=mock_box_position,
    )
    candidates = detector.predict(
        image=rgb,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        top_k=top_k,
    )
    candidates = _clip_and_sort_candidates(candidates, width=rgb.shape[1], height=rgb.shape[0], top_k=top_k)

    if save_overlay_path is not None:
        save_detection_overlay(rgb, candidates, save_overlay_path)

    return candidates


def save_detection_overlay(image: np.ndarray, candidates: Sequence[DetectionCandidate], path: str | Path) -> Path:
    """Save a simple RGB overlay with detection boxes and scores."""

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required for detection overlays. Install it with `pip install pillow`.") from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = Image.fromarray(normalize_rgb(image), mode="RGB")
    draw = ImageDraw.Draw(canvas)
    for index, candidate in enumerate(candidates):
        x0, y0, x1, y1 = np.asarray(candidate.box_xyxy, dtype=float).tolist()
        label = f"{index}: {candidate.phrase} {candidate.det_score:.2f}"
        draw.rectangle([(x0, y0), (x1, y1)], outline="red", width=3)
        draw.text((x0 + 2, max(0.0, y0 - 12)), label, fill="red")

    canvas.save(output_path)
    LOGGER.info("Saved detection overlay to %s", output_path)
    return output_path


@lru_cache(maxsize=4)
def _get_cached_adapter(
    backend: str,
    model_config_path: str | None,
    model_checkpoint_path: str | None,
    model_id: str,
    device: str | None,
    mock_box_position: str,
) -> GroundingDINOAdapter:
    backend_name = backend.lower()
    if backend_name == "auto":
        if model_config_path and model_checkpoint_path:
            backend_name = "groundingdino"
        else:
            backend_name = "hf"

    if backend_name in {"hf", "transformers"}:
        return HuggingFaceGroundingDINOAdapter(model_id=model_id, device=device)
    if backend_name == "mock":
        return MockGroundingDINOAdapter(box_position=mock_box_position)
    if backend_name in {"groundingdino", "original"}:
        if not model_config_path or not model_checkpoint_path:
            raise RuntimeError(
                "The original GroundingDINO backend requires --model-config-path and "
                "--model-checkpoint-path. Alternatively use --detector-backend hf."
            )
        return OriginalGroundingDINOAdapter(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=device,
        )

    raise ValueError(f"Unknown GroundingDINO backend: {backend}")


class MockGroundingDINOAdapter:
    """Deterministic image-size-aware detector for lightweight smoke tests."""

    def __init__(self, box_position: str = "center") -> None:
        allowed_positions = {"center", "left", "right", "all"}
        if box_position not in allowed_positions:
            raise ValueError(f"mock_box_position must be one of {sorted(allowed_positions)}, got {box_position!r}.")
        self.box_position = box_position

    def predict(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
        top_k: int,
    ) -> list[DetectionCandidate]:
        rgb = normalize_rgb(image)
        height, width = rgb.shape[:2]
        prompt = text_prompt.strip() or "mock object"
        specs = {
            "center": ((0.30, 0.25, 0.70, 0.75), 0.95),
            "left": ((0.08, 0.25, 0.48, 0.75), 0.85),
            "right": ((0.52, 0.25, 0.92, 0.75), 0.80),
        }
        positions = ["center", "left", "right"] if self.box_position == "all" else [self.box_position]
        candidates: list[DetectionCandidate] = []
        for position in positions:
            box_fractions, score = specs[position]
            candidates.append(
                DetectionCandidate(
                    box_xyxy=_fractional_box_to_pixels(box_fractions, width=width, height=height),
                    det_score=score,
                    phrase=prompt,
                    source="mock",
                    metadata={
                        "backend": "mock",
                        "box_position": position,
                        "box_threshold": box_threshold,
                        "text_threshold": text_threshold,
                    },
                )
            )
        return candidates[: max(1, top_k)]


class HuggingFaceGroundingDINOAdapter:
    """GroundingDINO adapter using ``transformers`` zero-shot object detection."""

    def __init__(self, model_id: str, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face GroundingDINO backend requires `torch` and `transformers`. "
                "Install them with `pip install torch transformers`, or use the original "
                "GroundingDINO backend with config/checkpoint paths."
            ) from exc

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.model_id = model_id

    def predict(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
        top_k: int,
    ) -> list[DetectionCandidate]:
        from PIL import Image

        rgb = normalize_rgb(image)
        height, width = rgb.shape[:2]
        pil_image = Image.fromarray(rgb, mode="RGB")
        prompt = _normalize_detection_prompt(text_prompt)

        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = self.torch.tensor([[height, width]], device=self.device)
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs.get("input_ids"),
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold,
                text_threshold,
                target_sizes,
            )[0]

        boxes = _tensor_to_numpy(results.get("boxes", []))
        scores = _tensor_to_numpy(results.get("scores", []))
        labels = results.get("labels", results.get("text_labels", []))
        candidates: list[DetectionCandidate] = []
        for box, score, label in zip(boxes, scores, labels):
            candidates.append(
                DetectionCandidate(
                    box_xyxy=np.asarray(box, dtype=np.float32),
                    det_score=float(score),
                    phrase=str(label),
                    source="transformers_groundingdino",
                    metadata={"model_id": self.model_id},
                )
            )
        return candidates[:top_k]


class OriginalGroundingDINOAdapter:
    """Adapter for the original GroundingDINO repository inference helpers."""

    def __init__(self, model_config_path: str, model_checkpoint_path: str, device: str | None = None) -> None:
        try:
            import torch
            import groundingdino.datasets.transforms as transforms
            from groundingdino.util.inference import load_model, predict
        except ImportError as exc:
            raise RuntimeError(
                "Original GroundingDINO backend requires the GroundingDINO repository "
                "package plus `torch`. Install GroundingDINO and provide config/checkpoint "
                "paths, or use --detector-backend hf."
            ) from exc

        self.torch = torch
        self.transforms = transforms
        self.predict_fn = predict
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(model_config_path, model_checkpoint_path, device=self.device)
        self.transform = transforms.Compose(
            [
                transforms.RandomResize([800], max_size=1333),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
        top_k: int,
    ) -> list[DetectionCandidate]:
        from PIL import Image

        rgb = normalize_rgb(image)
        height, width = rgb.shape[:2]
        pil_image = Image.fromarray(rgb, mode="RGB")
        transformed_image, _ = self.transform(pil_image, None)
        boxes, logits, phrases = self.predict_fn(
            model=self.model,
            image=transformed_image,
            caption=_normalize_detection_prompt(text_prompt),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        boxes_np = _cxcywh_to_xyxy_pixels(_tensor_to_numpy(boxes), width=width, height=height)
        logits_np = _tensor_to_numpy(logits)
        candidates = [
            DetectionCandidate(
                box_xyxy=box.astype(np.float32),
                det_score=float(score),
                phrase=str(phrase),
                source="original_groundingdino",
            )
            for box, score, phrase in zip(boxes_np, logits_np, phrases)
        ]
        return candidates[:top_k]


def _clip_and_sort_candidates(
    candidates: Sequence[DetectionCandidate],
    width: int,
    height: int,
    top_k: int,
) -> list[DetectionCandidate]:
    clipped: list[DetectionCandidate] = []
    for candidate in candidates:
        box = np.asarray(candidate.box_xyxy, dtype=np.float32).copy()
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
        if box[2] <= box[0] or box[3] <= box[1]:
            LOGGER.debug("Skipping empty detection box: %s", box)
            continue
        candidate.box_xyxy = box
        clipped.append(candidate)

    return sorted(clipped, key=lambda item: float(item.det_score), reverse=True)[:top_k]


def _normalize_detection_prompt(text_prompt: str) -> str:
    prompt = text_prompt.strip().lower()
    if not prompt:
        raise ValueError("Detection prompt must be non-empty.")
    if not prompt.endswith("."):
        prompt += "."
    return prompt


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _cxcywh_to_xyxy_pixels(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    cx, cy, w_box, h_box = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xyxy = np.stack(
        [
            (cx - 0.5 * w_box) * width,
            (cy - 0.5 * h_box) * height,
            (cx + 0.5 * w_box) * width,
            (cy + 0.5 * h_box) * height,
        ],
        axis=1,
    )
    return xyxy.astype(np.float32)


def _fractional_box_to_pixels(box_fractions: tuple[float, float, float, float], width: int, height: int) -> np.ndarray:
    x0, y0, x1, y1 = box_fractions
    return np.array(
        [
            x0 * float(width - 1),
            y0 * float(height - 1),
            x1 * float(width - 1),
            y1 * float(height - 1),
        ],
        dtype=np.float32,
    )
