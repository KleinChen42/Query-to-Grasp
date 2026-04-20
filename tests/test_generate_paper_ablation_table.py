from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.generate_paper_ablation_table import (
    build_table_rows,
    parse_benchmark_spec,
    render_markdown_table,
    write_rows_csv,
)


def test_parse_benchmark_spec_with_label() -> None:
    label, path = parse_benchmark_spec("hf clip=outputs/benchmark_hf_with_clip")

    assert label == "hf clip"
    assert path == Path("outputs/benchmark_hf_with_clip")


def test_parse_benchmark_spec_without_label_uses_directory_name() -> None:
    label, path = parse_benchmark_spec("outputs/benchmark_hf_no_clip")

    assert label == "benchmark_hf_no_clip"
    assert path == Path("outputs/benchmark_hf_no_clip")


def test_build_table_rows_uses_summary_metrics(tmp_path: Path) -> None:
    no_clip_dir = tmp_path / "no_clip"
    clip_dir = tmp_path / "with_clip"
    _write_summary(no_clip_dir, skip_clip=True, raw=1.0, changed=0.0, runtime=1.0)
    _write_summary(clip_dir, skip_clip=False, raw=2.0, changed=0.5, runtime=2.0)

    rows = build_table_rows([f"Detector only={no_clip_dir}", f"Detector + CLIP={clip_dir}"])

    assert [row["label"] for row in rows] == ["Detector only", "Detector + CLIP"]
    assert rows[0]["skip_clip"] == "True"
    assert rows[1]["skip_clip"] == "False"
    assert rows[1]["mean_raw_num_detections"] == 2.0
    assert rows[1]["fraction_top1_changed_by_rerank"] == 0.5


def test_render_markdown_table_and_csv(tmp_path: Path) -> None:
    rows = [
        {
            "label": "Detector + CLIP",
            "detector_backend": "hf",
            "skip_clip": "False",
            "total_runs": 6,
            "mean_raw_num_detections": 2.0,
            "mean_num_ranked_candidates": 2.0,
            "fraction_top1_changed_by_rerank": 0.5,
            "fraction_with_3d_target": 1.0,
            "pick_success_rate": 0.0,
            "mean_runtime_seconds": 2.25,
        }
    ]

    markdown = render_markdown_table(rows)
    csv_path = tmp_path / "table.csv"
    write_rows_csv(rows, csv_path)

    assert "# Paper Ablation Table" in markdown
    assert "Detector + CLIP" in markdown
    assert "0.5000" in markdown
    csv_rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    assert csv_rows[0]["label"] == "Detector + CLIP"
    assert csv_rows[0]["mean_runtime_seconds"] == "2.25"


def _write_summary(
    benchmark_dir: Path,
    skip_clip: bool,
    raw: float,
    changed: float,
    runtime: float,
) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_runs": 6,
        "detector_backend": "hf",
        "skip_clip": skip_clip,
        "aggregate_metrics": {
            "total_runs": 6,
            "mean_raw_num_detections": raw,
            "mean_num_detections": raw,
            "mean_num_ranked_candidates": raw,
            "fraction_top1_changed_by_rerank": changed,
            "fraction_with_3d_target": 1.0,
            "pick_success_rate": 0.0,
            "mean_runtime_seconds": runtime,
        },
    }
    (benchmark_dir / "benchmark_summary.json").write_text(json.dumps(summary), encoding="utf-8")
