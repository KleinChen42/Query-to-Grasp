from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.generate_per_query_diagnostics_table import (
    build_table_rows,
    render_markdown_table,
    row_from_query_metrics,
    rows_from_benchmark,
    write_rows_csv,
)


def test_build_table_rows_expands_per_query_metrics(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "ambiguity_hf_with_clip"
    _write_summary(
        benchmark_dir,
        skip_clip=False,
        per_query_metrics={
            "object": _metrics(total_runs=3, raw=2.0, ranked=2.0, changed=1.0, target=1.0, runtime=10.0),
            "red cube": _metrics(total_runs=3, raw=1.0, ranked=1.0, changed=0.0, target=1.0, runtime=8.0),
        },
    )

    rows = build_table_rows([f"Ambiguity with CLIP={benchmark_dir}"])

    assert [row["query"] for row in rows] == ["object", "red cube"]
    assert rows[0]["benchmark"] == "Ambiguity with CLIP"
    assert rows[0]["detector_backend"] == "hf"
    assert rows[0]["skip_clip"] == "False"
    assert rows[0]["mean_raw_num_detections"] == 2.0
    assert rows[0]["fraction_top1_changed_by_rerank"] == 1.0


def test_row_from_query_metrics_defaults_missing_rerank_fields() -> None:
    row = row_from_query_metrics(
        label="Old summary",
        query="cube",
        detector_backend="hf",
        skip_clip="True",
        metrics={
            "total_runs": 2,
            "mean_num_detections": 1.5,
            "fraction_with_3d_target": 0.5,
        },
    )

    assert row["mean_raw_num_detections"] == 1.5
    assert row["mean_num_ranked_candidates"] == 0.0
    assert row["fraction_top1_changed_by_rerank"] == 0.0
    assert row["mean_runtime_seconds"] == 0.0


def test_build_table_rows_reports_missing_summaries(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError) as error:
        build_table_rows([f"Missing={missing_dir}"])

    message = str(error.value)
    assert "Missing benchmark_summary.json" in message
    assert str(missing_dir / "benchmark_summary.json") in message


def test_build_table_rows_can_skip_missing_summaries(tmp_path: Path) -> None:
    existing_dir = tmp_path / "existing"
    missing_dir = tmp_path / "missing"
    _write_summary(
        existing_dir,
        skip_clip=True,
        per_query_metrics={"cube": _metrics(total_runs=1, raw=1.0, ranked=1.0, changed=0.0, target=1.0, runtime=1.0)},
    )

    rows = build_table_rows([f"Existing={existing_dir}", f"Missing={missing_dir}"], skip_missing=True)

    assert len(rows) == 1
    assert rows[0]["benchmark"] == "Existing"


def test_build_table_rows_requires_per_query_metrics(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "old_summary"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "benchmark_summary.json").write_text(
        json.dumps({"aggregate_metrics": {"total_runs": 1}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No per-query metrics"):
        build_table_rows([f"Old={benchmark_dir}"])


def test_rows_from_benchmark_ignores_non_dict_query_metrics(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "mixed_summary"
    _write_summary(
        benchmark_dir,
        skip_clip=False,
        per_query_metrics={
            "cube": _metrics(total_runs=1, raw=1.0, ranked=1.0, changed=0.0, target=1.0, runtime=1.0),
            "broken": "not a metrics object",
        },
    )

    rows = rows_from_benchmark("Mixed", benchmark_dir)

    assert len(rows) == 1
    assert rows[0]["query"] == "cube"


def test_render_markdown_table_and_csv(tmp_path: Path) -> None:
    rows = [
        {
            "benchmark": "Ambiguity | CLIP",
            "query": "object",
            "detector_backend": "hf",
            "skip_clip": "False",
            "total_runs": 3,
            "mean_raw_num_detections": 2.0,
            "mean_num_detections": 2.0,
            "mean_num_ranked_candidates": 2.0,
            "fraction_top1_changed_by_rerank": 0.5,
            "fraction_with_3d_target": 1.0,
            "pick_success_rate": 0.0,
            "mean_runtime_seconds": 12.25,
        }
    ]

    markdown = render_markdown_table(rows)
    csv_path = tmp_path / "per_query.csv"
    write_rows_csv(rows, csv_path)

    assert "# Per-Query Diagnostics Table" in markdown
    assert "Ambiguity \\| CLIP" in markdown
    assert "fraction_top1_changed_by_rerank" in markdown
    assert "0.5000" in markdown
    csv_rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    assert csv_rows[0]["benchmark"] == "Ambiguity | CLIP"
    assert csv_rows[0]["mean_runtime_seconds"] == "12.25"


def _write_summary(
    benchmark_dir: Path,
    skip_clip: bool,
    per_query_metrics: dict[str, object],
) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_runs": 3,
        "detector_backend": "hf",
        "skip_clip": skip_clip,
        "aggregate_metrics": {
            "total_runs": 3,
            "mean_raw_num_detections": 1.5,
            "mean_num_ranked_candidates": 1.5,
            "fraction_top1_changed_by_rerank": 0.5,
        },
        "per_query_metrics": per_query_metrics,
    }
    (benchmark_dir / "benchmark_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _metrics(
    total_runs: int,
    raw: float,
    ranked: float,
    changed: float,
    target: float,
    runtime: float,
) -> dict[str, float | int]:
    return {
        "total_runs": total_runs,
        "mean_raw_num_detections": raw,
        "mean_num_detections": raw,
        "mean_num_ranked_candidates": ranked,
        "fraction_top1_changed_by_rerank": changed,
        "fraction_with_3d_target": target,
        "pick_success_rate": 0.0,
        "mean_runtime_seconds": runtime,
    }
