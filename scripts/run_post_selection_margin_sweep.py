"""Run and summarize a compact post-selection continuity margin sweep."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES_FILE = PROJECT_ROOT / "configs" / "ambiguity_queries.txt"

try:
    from scripts.generate_paper_ablation_table import benchmark_summary_path
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    from generate_paper_ablation_table import benchmark_summary_path  # type: ignore


LOGGER = logging.getLogger(__name__)

TABLE_COLUMNS = [
    "label",
    "margin",
    "detector_backend",
    "skip_clip",
    "total_runs",
    "closed_loop_execution_rate",
    "closed_loop_resolution_rate",
    "closed_loop_still_needed_rate",
    "closed_loop_selected_object_change_rate",
    "closed_loop_before_selected_received_observation_rate",
    "closed_loop_before_selected_gained_view_support_rate",
    "closed_loop_final_selected_absorbed_extra_view_rate",
    "closed_loop_extra_view_third_object_involved_rate",
    "mean_closed_loop_preferred_merge_rate",
    "closed_loop_post_selection_continuity_eligibility_rate",
    "closed_loop_post_selection_continuity_apply_rate",
    "mean_closed_loop_delta_selected_overall_confidence",
    "mean_closed_loop_delta_selected_num_views",
    "mean_selected_overall_confidence",
    "mean_runtime_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a compact post-selection continuity margin sweep over the existing multiview fusion benchmark."
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=DEFAULT_QUERIES_FILE,
        help="Text file with one query per line. Blank lines and # comments are ignored by the benchmark runner.",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Integer seeds passed to the benchmark runner.")
    parser.add_argument("--num-runs", type=int, default=1, help="Fallback number of seeds when --seeds is omitted.")
    parser.add_argument("--margins", nargs="*", type=float, default=[0.03, 0.05, 0.08, 0.12], help="Post-selection continuity margins to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "post_selection_margin_sweep")
    parser.add_argument(
        "--detector-backend",
        default="hf",
        choices=["auto", "hf", "transformers", "groundingdino", "original", "mock"],
    )
    parser.add_argument("--mock-box-position", default="center", choices=["center", "left", "right", "all"])
    parser.add_argument("--skip-clip", dest="skip_clip", action="store_true", default=False)
    parser.add_argument("--use-clip", dest="skip_clip", action="store_false", help="Run CLIP reranking. This is the default unless --skip-clip is set.")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", default="rgbd")
    parser.add_argument("--view-preset", default="tabletop_3")
    parser.add_argument("--camera-name", default="base_camera")
    parser.add_argument("--merge-distance", type=float, default=0.08)
    parser.add_argument("--closed-loop-max-extra-views", type=int, default=1)
    parser.add_argument("--selected-object-continuity-distance-scale", type=float, default=1.0)
    parser.add_argument("--generate-policy-report", action="store_true", help="Generate one aggregate re-observation report across all sweep runs.")
    parser.add_argument("--fail-on-child-error", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing benchmark_summary.json directories instead of rerunning them.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_specs: list[str] = []
    for margin in args.margins:
        margin_dir = args.output_dir / margin_tag(margin)
        benchmark_specs.append(f"{margin_label(margin)}={margin_dir}")
        if args.skip_existing and benchmark_summary_path(margin_dir).exists():
            LOGGER.info("Reusing existing margin run: %.4f at %s", margin, margin_dir)
            continue
        run_command(build_benchmark_command(args, margin=margin, output_dir=margin_dir))

    rows = build_table_rows(args.margins, args.output_dir)
    conclusion = build_conclusion(rows)
    write_rows_csv(rows, args.output_dir / "margin_sweep_summary.csv")
    write_json(
        {
            "rows": rows,
            "conclusion": conclusion,
        },
        args.output_dir / "margin_sweep_summary.json",
    )
    (args.output_dir / "margin_sweep_summary.md").write_text(render_markdown(rows, conclusion), encoding="utf-8")

    if args.generate_policy_report:
        run_command(build_policy_report_command(benchmark_specs, args.output_dir))

    print(f"Wrote margin sweep markdown: {args.output_dir / 'margin_sweep_summary.md'}")
    print(f"Wrote margin sweep CSV:      {args.output_dir / 'margin_sweep_summary.csv'}")
    print(f"Wrote margin sweep JSON:     {args.output_dir / 'margin_sweep_summary.json'}")
    return 0


def build_benchmark_command(args: argparse.Namespace, margin: float, output_dir: Path) -> list[str]:
    """Build one multiview fusion benchmark command for a specific margin."""

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_multiview_fusion_benchmark.py"),
        "--queries-file",
        str(args.queries_file),
        "--num-runs",
        str(args.num_runs),
        "--detector-backend",
        args.detector_backend,
        "--mock-box-position",
        args.mock_box_position,
        "--depth-scale",
        str(args.depth_scale),
        "--env-id",
        args.env_id,
        "--obs-mode",
        args.obs_mode,
        "--view-preset",
        args.view_preset,
        "--camera-name",
        str(args.camera_name),
        "--merge-distance",
        str(args.merge_distance),
        "--enable-closed-loop-reobserve",
        "--closed-loop-max-extra-views",
        str(args.closed_loop_max_extra_views),
        "--enable-selected-object-continuity",
        "--selected-object-continuity-distance-scale",
        str(args.selected_object_continuity_distance_scale),
        "--enable-post-reobserve-selection-continuity",
        "--post-reobserve-selection-margin",
        str(margin),
        "--output-dir",
        str(output_dir),
    ]
    if args.seeds:
        command.append("--seeds")
        command.extend(str(seed) for seed in args.seeds)
    if args.skip_clip:
        command.append("--skip-clip")
    else:
        command.append("--use-clip")
    if args.fail_on_child_error:
        command.append("--fail-on-child-error")
    return command


def build_policy_report_command(benchmark_specs: list[str], output_dir: Path) -> list[str]:
    """Build the aggregate re-observation report command for all sweep runs."""

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_reobserve_policy_report.py"),
        "--output-md",
        str(output_dir / "reobserve_policy_report_margin_sweep.md"),
        "--output-json",
        str(output_dir / "reobserve_policy_report_margin_sweep.json"),
    ]
    for spec in benchmark_specs:
        command.extend(["--benchmark", spec])
    return command


def build_table_rows(margins: list[float], output_dir: Path) -> list[dict[str, Any]]:
    """Build one table row per margin run."""

    rows = [row_from_benchmark(margin, output_dir / margin_tag(margin)) for margin in margins]
    if not rows:
        raise ValueError("No margin rows were available to summarize.")
    return rows


def row_from_benchmark(margin: float, benchmark_dir: Path) -> dict[str, Any]:
    """Load one benchmark summary into a sweep row."""

    summary = load_benchmark_summary(benchmark_dir)
    metrics = summary.get("aggregate_metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "label": margin_label(margin),
        "margin": float(margin),
        "detector_backend": _value(summary.get("detector_backend")),
        "skip_clip": _value(summary.get("skip_clip")),
        "total_runs": _as_int(summary.get("total_runs", metrics.get("total_runs"))),
        "closed_loop_execution_rate": _as_float(metrics.get("closed_loop_execution_rate")),
        "closed_loop_resolution_rate": _as_float(metrics.get("closed_loop_resolution_rate")),
        "closed_loop_still_needed_rate": _as_float(metrics.get("closed_loop_still_needed_rate")),
        "closed_loop_selected_object_change_rate": _as_float(metrics.get("closed_loop_selected_object_change_rate")),
        "closed_loop_before_selected_received_observation_rate": _as_float(
            metrics.get("closed_loop_before_selected_received_observation_rate")
        ),
        "closed_loop_before_selected_gained_view_support_rate": _as_float(
            metrics.get("closed_loop_before_selected_gained_view_support_rate")
        ),
        "closed_loop_final_selected_absorbed_extra_view_rate": _as_float(
            metrics.get("closed_loop_final_selected_absorbed_extra_view_rate")
        ),
        "closed_loop_extra_view_third_object_involved_rate": _as_float(
            metrics.get("closed_loop_extra_view_third_object_involved_rate")
        ),
        "mean_closed_loop_preferred_merge_rate": _as_float(metrics.get("mean_closed_loop_preferred_merge_rate")),
        "closed_loop_post_selection_continuity_eligibility_rate": _as_float(
            metrics.get("closed_loop_post_selection_continuity_eligibility_rate")
        ),
        "closed_loop_post_selection_continuity_apply_rate": _as_float(
            metrics.get("closed_loop_post_selection_continuity_apply_rate")
        ),
        "mean_closed_loop_delta_selected_overall_confidence": _as_float(
            metrics.get("mean_closed_loop_delta_selected_overall_confidence")
        ),
        "mean_closed_loop_delta_selected_num_views": _as_float(
            metrics.get("mean_closed_loop_delta_selected_num_views")
        ),
        "mean_selected_overall_confidence": _as_float(metrics.get("mean_selected_overall_confidence")),
        "mean_runtime_seconds": _as_float(metrics.get("mean_runtime_seconds")),
        "benchmark_dir": str(benchmark_dir),
    }


def build_conclusion(rows: list[dict[str, Any]]) -> str:
    """Build a short rule-based conclusion from the sweep rows."""

    best_resolution = max(rows, key=lambda row: (float(row.get("closed_loop_resolution_rate", 0.0)), float(row.get("closed_loop_post_selection_continuity_apply_rate", 0.0))))
    best_apply = max(rows, key=lambda row: float(row.get("closed_loop_post_selection_continuity_apply_rate", 0.0)))

    resolution_rate = float(best_resolution.get("closed_loop_resolution_rate", 0.0))
    apply_rate = float(best_apply.get("closed_loop_post_selection_continuity_apply_rate", 0.0))

    if resolution_rate <= 0.0 and apply_rate <= 0.0:
        return (
            "Across the tested margins, post-selection continuity still did not apply and did not improve closed-loop "
            "resolution. The next bottleneck is likely continuity eligibility or memory association, not the margin value alone."
        )
    if resolution_rate <= 0.0:
        return (
            f"Margin {best_apply['label']} increased post-selection continuity application to {apply_rate:.4f}, "
            "but closed-loop resolution remained flat. Extra-view continuity is engaging, but it is not yet sufficient to resolve uncertainty."
        )
    return (
        f"Margin {best_resolution['label']} achieved the strongest closed-loop resolution ({resolution_rate:.4f}) "
        f"with post-selection continuity application {float(best_resolution.get('closed_loop_post_selection_continuity_apply_rate', 0.0)):.4f}."
    )


def run_command(command: list[str]) -> None:
    """Run a child command from the project root."""

    LOGGER.info("Running: %s", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the sweep rows as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TABLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write a JSON file with parent creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def render_markdown(rows: list[dict[str, Any]], conclusion: str) -> str:
    """Render the sweep summary as Markdown."""

    lines = [
        "# Post-Selection Margin Sweep",
        "",
        "| " + " | ".join(TABLE_COLUMNS) + " |",
        "| " + " | ".join(["---", "---:"] + ["---"] * 2 + ["---:"] * (len(TABLE_COLUMNS) - 4)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(column)) for column in TABLE_COLUMNS) + " |")
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            conclusion,
            "",
        ]
    )
    return "\n".join(lines)


def margin_tag(margin: float) -> str:
    """Return a stable directory tag for one margin value."""

    return f"margin_{margin_label(margin).replace('.', 'p')}"


def margin_label(margin: float) -> str:
    """Return a compact human-readable label for one margin value."""

    text = f"{margin:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def load_benchmark_summary(benchmark_dir: str | Path) -> dict[str, Any]:
    """Load benchmark_summary.json from a benchmark directory."""

    path = benchmark_summary_path(benchmark_dir)
    if not path.exists():
        raise FileNotFoundError(f"Required benchmark summary not found: {path}")
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}.")
    return data


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return _format_float(value)
    return _escape_table_cell(_value(value))


def _format_float(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.4f}"


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _value(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value)


def _as_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
