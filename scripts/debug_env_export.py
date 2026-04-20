"""Debug export script for Phase 1 environment observations."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.camera_utils import ObservationFrame  # noqa: E402
from src.env.maniskill_env import ManiSkillScene  # noqa: E402
from src.geometry.rgbd_to_pointcloud import generate_and_save_pointcloud  # noqa: E402
from src.io.export_utils import export_observation_frame  # noqa: E402


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one ManiSkill RGB-D observation and point cloud.")
    parser.add_argument("--env-id", default="PickCube-v1", help="ManiSkill environment id.")
    parser.add_argument("--obs-mode", default="rgbd", help="Observation mode passed to ManiSkill.")
    parser.add_argument("--control-mode", default=None, help="Optional control mode passed to ManiSkill.")
    parser.add_argument("--camera-name", default=None, help="Optional camera key to prefer in observation parsing.")
    parser.add_argument("--seed", type=int, default=0, help="Reset seed.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "env_debug", help="Artifact directory root.")
    parser.add_argument("--depth-scale", type=float, default=1.0, help="Divide depth values by this scale before projection.")
    parser.add_argument("--fallback-fov-degrees", type=float, default=60.0, help="Fallback FOV if intrinsics are unavailable.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{timestamp}_{args.env_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    scene = ManiSkillScene(
        env_name=args.env_id,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        camera_name=args.camera_name,
    )
    try:
        scene.reset(seed=args.seed)
        frame = scene.get_observation(camera_name=args.camera_name)
        print_observation_summary(frame)

        export_observation_frame(frame=frame, output_dir=run_dir, env_name=args.env_id, step_name="reset")
        if frame.rgb is None or frame.depth is None:
            LOGGER.warning("Skipping point cloud export because RGB or depth was not found.")
            return

        pointcloud_path = run_dir / "pointcloud.ply"
        generate_and_save_pointcloud(
            rgb=frame.rgb,
            depth=frame.depth,
            intrinsic=frame.camera_info.intrinsic,
            extrinsic=frame.camera_info.extrinsic,
            output_path=pointcloud_path,
            depth_scale=args.depth_scale,
            fallback_fov_degrees=args.fallback_fov_degrees,
        )
        LOGGER.info("Done. Artifacts written to %s", run_dir)
    finally:
        scene.close()


def print_observation_summary(frame: ObservationFrame) -> None:
    """Print a compact summary suitable for notebook/script debugging."""

    print("Observation summary")
    print(f"  RGB:          {_shape_dtype(frame.rgb)} from {frame.source_keys.get('rgb')}")
    print(f"  Depth:        {_shape_dtype(frame.depth)} from {frame.source_keys.get('depth')}")
    print(f"  Segmentation: {_shape_dtype(frame.segmentation)} from {frame.source_keys.get('segmentation')}")
    print(f"  Intrinsic:    {_shape_dtype(frame.camera_info.intrinsic)} from {frame.source_keys.get('intrinsic')}")
    print(f"  Extrinsic:    {_shape_dtype(frame.camera_info.extrinsic)} from {frame.source_keys.get('extrinsic')}")
    print(f"  Leaf keys:    {len(frame.observation_keys)}")


def _shape_dtype(array: object) -> str:
    if array is None:
        return "missing"
    shape = getattr(array, "shape", None)
    dtype = getattr(array, "dtype", None)
    return f"shape={tuple(shape)}, dtype={dtype}"


if __name__ == "__main__":
    main()

