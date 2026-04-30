from __future__ import annotations

import numpy as np

from scripts.run_oracle_place_benchmark import (
    aggregate_oracle_place_rows,
    find_stackcube_oracle_place_targets,
)


def test_find_stackcube_oracle_place_targets_reads_cube_a_and_cube_b() -> None:
    env = _FakeStackCubeEnv()

    targets = find_stackcube_oracle_place_targets(env)

    np.testing.assert_allclose(targets.pick_xyz, [0.1, 0.2, 0.02])
    np.testing.assert_allclose(targets.place_xyz, [-0.1, -0.2, 0.02])
    assert targets.metadata["cubeA_attribute_type"] == "_FakeActor"
    assert targets.metadata["cubeB_attribute_type"] == "_FakeActor"


def test_aggregate_oracle_place_rows_reports_place_and_task_rates() -> None:
    rows = [
        _row(seed=0, pick_success=True, place_success=True, task_success=True, stage="success"),
        _row(seed=1, pick_success=True, place_success=False, task_success=False, stage="place_not_confirmed"),
        _row(seed=2, pick_success=False, place_success=False, task_success=False, stage="run_failed", failed=True),
    ]

    metrics = aggregate_oracle_place_rows(rows)

    assert metrics["total_runs"] == 3
    assert metrics["failed_runs"] == 1
    assert metrics["grasp_attempted_rate"] == 2 / 3
    assert metrics["place_attempted_rate"] == 2 / 3
    assert metrics["pick_success_rate"] == 2 / 3
    assert metrics["place_success_rate"] == 1 / 3
    assert metrics["task_success_rate"] == 1 / 3
    assert metrics["pick_stage_counts"] == {
        "place_not_confirmed": 1,
        "run_failed": 1,
        "success": 1,
    }
    assert metrics["pick_target_source_counts"] == {"oracle_cubeA_pose": 3}
    assert metrics["place_target_source_counts"] == {"oracle_cubeB_pose": 3}


def _row(
    seed: int,
    pick_success: bool,
    place_success: bool,
    task_success: bool,
    stage: str,
    failed: bool = False,
) -> dict[str, object]:
    return {
        "seed": seed,
        "pick_stage": stage,
        "stage": stage,
        "grasp_attempted": not failed,
        "place_attempted": not failed,
        "pick_success": pick_success,
        "place_success": place_success,
        "task_success": task_success,
        "is_grasped": pick_success,
        "run_failed": failed,
        "pick_target_source": "oracle_cubeA_pose",
        "place_target_source": "oracle_cubeB_pose",
        "runtime_seconds": 1.0,
    }


class _FakePose:
    def __init__(self, xyz: list[float]) -> None:
        self.p = np.asarray([xyz], dtype=np.float32)


class _FakeActor:
    def __init__(self, xyz: list[float]) -> None:
        self.pose = _FakePose(xyz)


class _FakeStackCubeEnv:
    def __init__(self) -> None:
        self.unwrapped = self
        self.cubeA = _FakeActor([0.1, 0.2, 0.02])
        self.cubeB = _FakeActor([-0.1, -0.2, 0.02])
        self.cube_half_size = np.asarray([0.02], dtype=np.float32)
