"""Probe available ManiSkill RGB-D camera views for multiview experiments."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.camera_utils import extract_observation_frame, flatten_observation_keys  # noqa: E402
from src.env.maniskill_env import ManiSkillScene  # noqa: E402
from src.io.export_utils import write_json  # noqa: E402


LOGGER = logging.getLogger(__name__)

CAMERA_DATA_MARKERS = {
    "rgb",
    "color",
    "colour",
    "depth",
    "seg",
    "segmentation",
    "actor_seg",
    "object_seg",
    "label",
}
CAMERA_PARAM_MARKERS = {
    "intrinsic",
    "intrinsics",
    "intrinsic_cv",
    "camera_matrix",
    "extrinsic",
    "extrinsics",
    "cam2world",
    "camera_to_world",
    "c2w",
    "pose",
    "k",
}
CAMERA_CONTAINERS = {"sensor_data", "sensor_param", "sensors", "cameras", "camera"}
STRICT_CAMERA_PARAM_MARKERS = CAMERA_PARAM_MARKERS - {"pose", "k"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List usable RGB-D camera names from one ManiSkill observation.")
    parser.add_argument("--env-id", default="PickCube-v1", help="ManiSkill environment id.")
    parser.add_argument("--obs-mode", default="rgbd", help="Observation mode passed to ManiSkill.")
    parser.add_argument("--control-mode", default=None, help="Optional control mode passed to ManiSkill.")
    parser.add_argument("--seed", type=int, default=0, help="Reset seed.")
    parser.add_argument(
        "--camera-name",
        action="append",
        default=[],
        help="Optional camera name to explicitly probe. May be provided multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "camera_view_probe",
        help="Directory for camera_view_report.json.",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")

    scene = ManiSkillScene(env_name=args.env_id, obs_mode=args.obs_mode, control_mode=args.control_mode)
    try:
        raw_observation = scene.reset(seed=args.seed)
        report = build_camera_probe_report(raw_observation, requested_camera_names=args.camera_name)
        report.update({"env_id": args.env_id, "obs_mode": args.obs_mode, "seed": args.seed})
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.output_dir / "camera_view_report.json"
        write_json(report, report_path)
        print_camera_report(report)
        LOGGER.info("Wrote camera probe report to %s", report_path)
    finally:
        scene.close()


def build_camera_probe_report(
    observation: Mapping[str, Any],
    requested_camera_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a compact RGB-D camera probe report from a raw observation."""

    observation_keys = flatten_observation_keys(observation)
    inferred_camera_names = infer_candidate_camera_names(observation_keys)
    camera_names = _unique_names([None, *(requested_camera_names or []), *inferred_camera_names])
    probes = [probe_camera(observation, camera_name) for camera_name in camera_names]

    usable_named_views = [
        probe["camera_name"]
        for probe in probes
        if probe["camera_name"] is not None and probe["usable_rgbd"] and probe["source_matches_camera_name"]
    ]

    return {
        "inferred_camera_names": inferred_camera_names,
        "usable_named_views": usable_named_views,
        "num_leaf_keys": len(observation_keys),
        "observation_keys": observation_keys,
        "probes": probes,
    }


def infer_candidate_camera_names(observation_keys: Sequence[str]) -> list[str]:
    """Infer camera names from dotted ManiSkill observation leaf keys.

    ManiSkill observations commonly use paths like
    ``sensor_data.base_camera.rgb`` and ``sensor_param.base_camera.intrinsic_cv``.
    This helper stays schema-light so it can also catch nearby variants.
    """

    names: set[str] = set()
    for key in observation_keys:
        parts = [part for part in key.replace("[", ".").replace("]", "").split(".") if part]
        lowered = [part.lower() for part in parts]
        if len(parts) < 2:
            continue

        has_camera_container = False
        for container in CAMERA_CONTAINERS:
            if container in lowered:
                has_camera_container = True
                index = lowered.index(container)
                if index + 1 < len(parts):
                    names.add(parts[index + 1])

        leaf = lowered[-1]
        if _leaf_matches_any(leaf, CAMERA_DATA_MARKERS):
            names.add(parts[-2])
        elif _leaf_matches_any(leaf, STRICT_CAMERA_PARAM_MARKERS):
            names.add(parts[-2])
        elif has_camera_container and leaf in {"pose", "k"}:
            names.add(parts[-2])

    return sorted(name for name in names if _looks_like_camera_name(name))


def probe_camera(observation: Mapping[str, Any], camera_name: str | None) -> dict[str, Any]:
    """Probe one camera name and summarize extracted modalities."""

    frame = extract_observation_frame(observation, camera_name=camera_name)
    source_matches = source_keys_match_camera_name(frame.source_keys, camera_name)
    return {
        "camera_name": camera_name,
        "display_name": camera_name or "default",
        "rgb_present": frame.rgb is not None,
        "depth_present": frame.depth is not None,
        "segmentation_present": frame.segmentation is not None,
        "intrinsic_present": frame.camera_info.intrinsic is not None,
        "extrinsic_present": frame.camera_info.extrinsic is not None,
        "rgb_shape": _shape(frame.rgb),
        "depth_shape": _shape(frame.depth),
        "segmentation_shape": _shape(frame.segmentation),
        "source_keys": frame.source_keys,
        "source_matches_camera_name": source_matches,
        "usable_rgbd": frame.rgb is not None and frame.depth is not None,
    }


def source_keys_match_camera_name(source_keys: Mapping[str, str | None], camera_name: str | None) -> bool:
    """Return whether selected source paths actually include the requested camera name."""

    if camera_name is None:
        return True
    camera_name_lower = camera_name.lower()
    selected_sources = [source.lower() for source in source_keys.values() if source]
    return bool(selected_sources) and any(camera_name_lower in source for source in selected_sources)


def print_camera_report(report: Mapping[str, Any]) -> None:
    """Print a small table for terminal diagnostics."""

    print("Camera view probe")
    print(f"  env_id: {report.get('env_id', 'unknown')}")
    print(f"  obs_mode: {report.get('obs_mode', 'unknown')}")
    print(f"  inferred_camera_names: {', '.join(report.get('inferred_camera_names', [])) or 'none'}")
    print(f"  usable_named_views: {', '.join(report.get('usable_named_views', [])) or 'none'}")
    print("")
    print("  view                 rgb    depth  intr   extr   source_match")
    for probe in report.get("probes", []):
        print(
            "  "
            f"{str(probe['display_name'])[:20]:20} "
            f"{_yes_no(probe['rgb_present']):6} "
            f"{_yes_no(probe['depth_present']):6} "
            f"{_yes_no(probe['intrinsic_present']):6} "
            f"{_yes_no(probe['extrinsic_present']):6} "
            f"{_yes_no(probe['source_matches_camera_name'])}"
        )


def _unique_names(names: Sequence[str | None]) -> list[str | None]:
    seen: set[str | None] = set()
    unique: list[str | None] = []
    for name in names:
        cleaned = name.strip() if isinstance(name, str) else name
        if cleaned == "":
            cleaned = None
        if cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique


def _leaf_matches_any(leaf: str, markers: set[str]) -> bool:
    return any(marker == leaf or marker in leaf for marker in markers)


def _looks_like_camera_name(name: str) -> bool:
    lowered = name.lower()
    if lowered in CAMERA_DATA_MARKERS or lowered in CAMERA_PARAM_MARKERS:
        return False
    if lowered in CAMERA_CONTAINERS:
        return False
    return True


def _shape(array: np.ndarray | None) -> list[int] | None:
    return list(array.shape) if array is not None else None


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    main()
