from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.generate_paper_revision_results_summary import RunSet, build_summary, write_outputs


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "pick_success",
        "place_attempted",
        "place_success",
        "task_success",
        "has_3d_target",
        "top1_changed_by_rerank",
        "num_detections",
        "num_ranked_candidates",
    ]
    with (path / "benchmark_rows.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_build_summary_combines_multiple_sources(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_rows(
        first,
        [
            {"seed": 0, "pick_success": True, "place_attempted": True, "place_success": True, "task_success": True, "has_3d_target": True, "top1_changed_by_rerank": False, "num_detections": 1, "num_ranked_candidates": 1},
            {"seed": 1, "pick_success": False, "place_attempted": True, "place_success": False, "task_success": False, "has_3d_target": True, "top1_changed_by_rerank": False, "num_detections": 2, "num_ranked_candidates": 2},
        ],
    )
    _write_rows(
        second,
        [
            {"seed": 2, "pick_success": True, "place_attempted": False, "place_success": False, "task_success": False, "has_3d_target": True, "top1_changed_by_rerank": True, "num_detections": 3, "num_ranked_candidates": 3},
        ],
    )
    runset = RunSet(
        label="combined",
        group="test",
        sources=(first, second),
        env_id="StackCube-v1",
        view_mode="single",
        target_mode="refined",
        pick_source="query",
        place_source="predicted",
        claim_boundary="diagnostic",
    )

    summary = build_summary([runset], strict=True)

    row = summary["rows"][0]
    assert row["total_runs"] == 3
    assert row["seed_min"] == 0
    assert row["seed_max"] == 2
    assert row["pick_success_count"] == 2
    assert row["pick_success_rate"] == 2 / 3
    assert row["place_attempted_count"] == 2
    assert row["task_success_count"] == 1
    assert row["top1_changed_count"] == 1
    assert row["mean_num_detections"] == 2.0


def test_write_outputs_creates_markdown_csv_json(tmp_path: Path) -> None:
    summary = {
        "total_runsets": 1,
        "missing_artifacts": [],
        "rows": [
            {
                "group": "clip_ablation",
                "label": "PickCube with CLIP",
                "env_id": "PickCube-v1",
                "view_mode": "single",
                "target_mode": "refined",
                "pick_source": "query",
                "place_source": "none",
                "claim_boundary": "clip ablation",
                "total_runs": 2,
                "pick_success_count": 2,
                "pick_success_rate": 1.0,
                "place_attempted_count": 0,
                "place_attempted_rate": 0.0,
                "place_success_count": 0,
                "place_success_rate": 0.0,
                "task_success_count": 0,
                "task_success_rate": 0.0,
                "target_source_count": 2,
                "target_source_rate": 1.0,
                "top1_changed_count": 0,
                "top1_changed_rate": 0.0,
                "mean_num_detections": 1.0,
                "mean_num_ranked_candidates": 1.0,
            }
        ],
    }

    write_outputs(summary, tmp_path)

    assert (tmp_path / "paper_revision_results_summary.md").exists()
    assert (tmp_path / "paper_revision_results_summary.csv").exists()
    loaded = json.loads((tmp_path / "paper_revision_results_summary.json").read_text(encoding="utf-8"))
    assert loaded["rows"][0]["label"] == "PickCube with CLIP"
    assert "CLIP Reranking Ablation" in (tmp_path / "paper_revision_results_summary.md").read_text(encoding="utf-8")
