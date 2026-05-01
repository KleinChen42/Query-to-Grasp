"""Opt-in execution video capture for simulated pick and pick-place demos."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.env.camera_utils import extract_observation_frame
from src.io.export_utils import save_rgb_png, write_json


@dataclass
class ExecutionVideoRecorder:
    """Collect RGB frames from env.step observations and write a demo MP4."""

    output_dir: Path
    fps: float = 24.0
    camera_name: str | None = "base_camera"
    every_n_steps: int = 1
    output_width: int | None = None
    output_height: int | None = None
    fallback_observation_fn: Callable[[], Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive.")
        if self.every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive.")
        if (self.output_width is None) != (self.output_height is None):
            raise ValueError("output_width and output_height must be provided together.")
        if self.output_width is not None and self.output_width <= 0:
            raise ValueError("output_width must be positive.")
        if self.output_height is not None and self.output_height <= 0:
            raise ValueError("output_height must be positive.")
        self.output_dir = Path(self.output_dir)
        self.frames_dir = self.output_dir / "execution_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._step_index = 0
        self._frames: list[Path] = []
        self._records: list[dict[str, Any]] = []
        self._capture_failures: list[dict[str, Any]] = []

    def record_step(self, *, stage: str, action: np.ndarray, observation: Any, info: dict[str, Any]) -> None:
        """Record one action step when the sampling interval allows it."""

        self._step_index += 1
        if (self._step_index - 1) % self.every_n_steps != 0:
            return
        rgb = self._extract_rgb(observation)
        capture_source = "step_observation"
        if rgb is None and self.fallback_observation_fn is not None:
            try:
                rgb = self._extract_rgb(self.fallback_observation_fn())
                capture_source = "fallback_sensor_observation"
            except Exception as exc:  # pragma: no cover - environment-specific fallback
                self._capture_failures.append(
                    {"step_index": self._step_index, "stage": stage, "reason": f"fallback_failed: {exc}"}
                )
        if rgb is None:
            self._capture_failures.append(
                {"step_index": self._step_index, "stage": stage, "reason": "capture_failed_missing_rgb"}
            )
            return

        source_height, source_width = rgb.shape[:2]
        rgb = self._prepare_output_frame(rgb)
        output_height, output_width = rgb.shape[:2]
        frame_path = self.frames_dir / f"{self._step_index:05d}_{_slug(stage)}.png"
        save_rgb_png(rgb, frame_path)
        self._frames.append(frame_path)
        self._records.append(
            {
                "step_index": self._step_index,
                "stage": stage,
                "frame": str(frame_path),
                "capture_source": capture_source,
                "action": np.asarray(action, dtype=float).reshape(-1).tolist(),
                "info_keys": sorted(str(key) for key in info),
                "source_resolution": [int(source_width), int(source_height)],
                "output_resolution": [int(output_width), int(output_height)],
            }
        )

    def finalize(self) -> dict[str, Any]:
        """Write MP4 and metadata, returning a JSON-serializable summary."""

        video_path = self.output_dir / "execution_video.mp4"
        video_status = "not_written_no_frames"
        if self._frames:
            video_status = write_frames_to_video(self._frames, video_path=video_path, fps=self.fps)
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "ok" if self._frames and video_path.exists() else video_status,
            "video_path": str(video_path) if video_path.exists() else None,
            "frames_dir": str(self.frames_dir),
            "frame_count": len(self._frames),
            "step_count_seen": self._step_index,
            "fps": float(self.fps),
            "camera_name": self.camera_name,
            "every_n_steps": int(self.every_n_steps),
            "output_width": self.output_width,
            "output_height": self.output_height,
            "resize_mode": "letterbox" if self.output_width is not None else "native",
            "source_resolutions": sorted({tuple(record["source_resolution"]) for record in self._records}),
            "output_resolutions": sorted({tuple(record["output_resolution"]) for record in self._records}),
            "records": self._records,
            "capture_failures": self._capture_failures,
            "metadata": self.metadata,
        }
        write_json(manifest, self.output_dir / "execution_video_metadata.json")
        return manifest

    def _extract_rgb(self, observation: Any) -> np.ndarray | None:
        if observation is None:
            return None
        try:
            frame = extract_observation_frame(observation, camera_name=self.camera_name)
        except Exception:
            return None
        if frame.rgb is None:
            return None
        return np.asarray(frame.rgb, dtype=np.uint8)

    def _prepare_output_frame(self, rgb: np.ndarray) -> np.ndarray:
        """Return an RGB frame at the requested output size without changing aspect ratio."""

        if self.output_width is None or self.output_height is None:
            return rgb
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - dependency checked by frame export tests
            raise RuntimeError("Pillow is required for high-resolution execution video resizing.") from exc

        source_height, source_width = rgb.shape[:2]
        target_width = int(self.output_width)
        target_height = int(self.output_height)
        scale = min(target_width / max(source_width, 1), target_height / max(source_height, 1))
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        image = Image.fromarray(rgb.astype(np.uint8, copy=False), mode="RGB")
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        image = image.resize((resized_width, resized_height), resample=resampling)
        canvas = Image.new("RGB", (target_width, target_height), color=(12, 16, 20))
        offset = ((target_width - resized_width) // 2, (target_height - resized_height) // 2)
        canvas.paste(image, offset)
        return np.asarray(canvas, dtype=np.uint8)


def write_frames_to_video(frame_paths: list[Path], video_path: Path, fps: float) -> str:
    """Write PNG frames to one MP4 using OpenCV."""

    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV is required to write execution videos.") from exc

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return "not_written_unreadable_first_frame"
    height, width = first.shape[:2]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        return "not_written_writer_unavailable"
    try:
        for path in frame_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    return "ok" if video_path.exists() and video_path.stat().st_size > 0 else "not_written_empty"


def _slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
    return slug[:48] or "stage"
