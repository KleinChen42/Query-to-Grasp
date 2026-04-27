"""Run a minimal oracle-target simulated pick smoke test."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.maniskill_env import ManiSkillScene  # noqa: E402
from src.io.export_utils import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an oracle-target ManiSkill simulated pick smoke test.")
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", default="rgbd")
    parser.add_argument("--control-mode", default="pd_ee_delta_pos")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-xyz", nargs=3, type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "oracle_pick_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{timestamp}_seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    scene = ManiSkillScene(env_name=args.env_id, obs_mode=args.obs_mode, control_mode=args.control_mode)
    try:
        scene.reset(seed=args.seed)
        target_xyz = np.asarray(args.target_xyz, dtype=np.float32) if args.target_xyz is not None else find_oracle_target_xyz(scene.env)
        if target_xyz is None:
            raise RuntimeError("Could not discover an oracle object pose. Pass --target-xyz x y z explicitly.")

        result = scene.execute_pick(target_xyz, executor="sim_topdown")
        summary = {
            "env_id": args.env_id,
            "seed": args.seed,
            "control_mode": args.control_mode,
            "target_xyz": target_xyz.astype(float).tolist(),
            "pick_success": bool(result.get("pick_success", result.get("success", False))),
            "grasp_attempted": bool(result.get("grasp_attempted", False)),
            "task_success": result.get("task_success"),
            "is_grasped": result.get("is_grasped"),
            "pick_stage": result.get("stage"),
            "runtime_seconds": time.perf_counter() - start_time,
            "artifacts": str(run_dir),
        }
        write_json(result, run_dir / "pick_result.json")
        write_json(summary, run_dir / "summary.json")
        print("Oracle pick smoke complete")
        print(f"  Target XYZ:   {summary['target_xyz']}")
        print(f"  Attempted:    {summary['grasp_attempted']}")
        print(f"  Pick success: {summary['pick_success']}")
        print(f"  Task success: {summary['task_success']}")
        print(f"  Stage:        {summary['pick_stage']}")
        print(f"  Artifacts:    {run_dir}")
    finally:
        scene.close()


def find_oracle_target_xyz(env: Any) -> np.ndarray | None:
    """Best-effort object pose discovery for ManiSkill oracle smoke tests."""

    unwrapped = getattr(env, "unwrapped", env)
    candidate_names = ["obj", "cube", "cube_actor", "object", "target_object"]
    for name in candidate_names:
        candidate = getattr(unwrapped, name, None)
        xyz = _pose_xyz(candidate)
        if xyz is not None:
            return xyz

    for value in vars(unwrapped).values():
        xyz = _pose_xyz(value)
        if xyz is not None:
            return xyz
    return None


def _pose_xyz(value: Any) -> np.ndarray | None:
    pose = getattr(value, "pose", None)
    position = getattr(pose, "p", None) if pose is not None else None
    if position is None:
        return None
    try:
        if hasattr(position, "detach"):
            position = position.detach()
        if hasattr(position, "cpu"):
            position = position.cpu()
        if hasattr(position, "numpy"):
            position = position.numpy()
        array = np.asarray(position, dtype=np.float32).reshape(-1, 3)
    except Exception:
        return None
    if array.size == 0 or not np.all(np.isfinite(array[0])):
        return None
    return array[0]


if __name__ == "__main__":
    main()
