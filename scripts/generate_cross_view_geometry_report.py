"""Generate a cross-view geometry sanity report from fusion debug artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.export_utils import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect per-view 2D/3D geometry consistency for fusion runs.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--run-dir", type=Path, help="One fusion debug run directory containing memory_state.json.")
    source_group.add_argument("--benchmark-dir", type=Path, help="Fusion benchmark directory containing benchmark_rows.json.")
    parser.add_argument("--output-md", type=Path, default=None, help="Markdown output path.")
    parser.add_argument("--output-json", type=Path, default=None, help="JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(run_dir=args.run_dir, benchmark_dir=args.benchmark_dir)

    output_json = args.output_json or _default_output_path(args, "cross_view_geometry_report.json")
    output_md = args.output_md or _default_output_path(args, "cross_view_geometry_report.md")
    write_json(report, output_json)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(report), encoding="utf-8")

    print(f"Wrote cross-view geometry JSON: {output_json}")
    print(f"Wrote cross-view geometry MD:   {output_md}")
    print(render_console_summary(report))
    return 0


def build_report(
    run_dir: str | Path | None = None,
    benchmark_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a geometry report from one run or a benchmark directory."""

    if run_dir is None and benchmark_dir is None:
        raise ValueError("Either run_dir or benchmark_dir is required.")
    if run_dir is not None and benchmark_dir is not None:
        raise ValueError("Provide only one of run_dir or benchmark_dir.")

    if run_dir is not None:
        runs = [analyze_run(resolve_path(run_dir))]
        source = str(run_dir)
        source_type = "run"
    else:
        benchmark_path = resolve_path(benchmark_dir)
        rows = load_json_list(benchmark_path / "benchmark_rows.json")
        runs = [analyze_run(resolve_path(row.get("artifacts"))) for row in rows if row.get("artifacts")]
        source = str(benchmark_dir)
        source_type = "benchmark"

    aggregate = aggregate_runs(runs)
    aggregate["conclusion"] = build_conclusion(aggregate)
    return {
        "source": source,
        "source_type": source_type,
        "aggregate": aggregate,
        "runs": runs,
    }


def analyze_run(run_dir: Path) -> dict[str, Any]:
    """Analyze one fusion debug run directory."""

    memory_state_path = run_dir / "memory_state.json"
    memory_state = load_json_dict(memory_state_path)
    candidates = extract_geometry_candidates(memory_state, run_dir=run_dir)
    by_label = group_candidates_by_label(candidates)
    label_stats = {
        label: geometry_stats_for_candidates(items)
        for label, items in sorted(by_label.items())
        if label
    }
    top_rank_candidates = [candidate for candidate in candidates if candidate["rank"] == 0]
    return {
        "run_dir": str(run_dir),
        "query": memory_state.get("query", {}).get("raw_query"),
        "selected_object_id": memory_state.get("selected_object_id"),
        "selection_label": memory_state.get("selection_label"),
        "num_candidates": len(candidates),
        "num_top_rank_candidates": len(top_rank_candidates),
        "extrinsic_source_counts": count_values(candidate.get("extrinsic_key") for candidate in candidates),
        "top_rank_distance_stats": distance_stats(pairwise_distances([item["world_xyz"] for item in top_rank_candidates])),
        "label_stats": label_stats,
        "candidates": candidates,
    }


def extract_geometry_candidates(memory_state: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """Extract candidates with camera/world geometry and observation metadata."""

    rows: list[dict[str, Any]] = []
    for view in memory_state.get("views", []):
        if not isinstance(view, dict):
            continue
        view_id = str(view.get("view_id") or "")
        view_dir = resolve_view_dir(view.get("artifacts"), run_dir=run_dir, view_id=view_id)
        metadata = load_optional_json_dict(view_dir / "observation" / "metadata.json")
        camera_info = metadata.get("camera_info", {}) if isinstance(metadata.get("camera_info"), dict) else {}
        extrinsic = matrix4(camera_info.get("extrinsic"))
        intrinsic = matrix3(camera_info.get("intrinsic"))
        ranked_candidates = view.get("reranked_candidates", [])
        candidates_3d = view.get("candidates_3d", [])
        if not isinstance(ranked_candidates, list) or not isinstance(candidates_3d, list):
            continue

        for index, candidate_3d in enumerate(candidates_3d):
            if not isinstance(candidate_3d, dict):
                continue
            ranked = ranked_candidates[index] if index < len(ranked_candidates) and isinstance(ranked_candidates[index], dict) else {}
            camera_xyz = xyz(candidate_3d.get("camera_xyz"))
            world_xyz = xyz(candidate_3d.get("world_xyz"))
            if world_xyz is None:
                continue
            recomputed_world = transform_point(extrinsic, camera_xyz) if extrinsic is not None and camera_xyz is not None else None
            rows.append(
                {
                    "view_id": view_id,
                    "rank": index,
                    "phrase": str(ranked.get("phrase") or ""),
                    "det_score": as_float(ranked.get("det_score")),
                    "clip_score": as_float(ranked.get("clip_score")),
                    "fused_2d_score": as_float(ranked.get("fused_2d_score")),
                    "box_xyxy": vector(candidate_3d.get("box_xyxy"), expected_len=4),
                    "camera_xyz": camera_xyz,
                    "world_xyz": world_xyz,
                    "recomputed_world_xyz": recomputed_world,
                    "world_recompute_error": (
                        euclidean_distance(world_xyz, recomputed_world) if recomputed_world is not None else None
                    ),
                    "camera_position": camera_position(extrinsic),
                    "extrinsic_key": camera_info.get("extrinsic_key"),
                    "intrinsic_key": camera_info.get("intrinsic_key"),
                    "has_intrinsic": intrinsic is not None,
                    "has_extrinsic": extrinsic is not None,
                    "num_points": as_int(candidate_3d.get("num_points")),
                    "depth_valid_ratio": as_float(candidate_3d.get("depth_valid_ratio")),
                    "view_artifacts": str(view_dir),
                }
            )
    return rows


def geometry_stats_for_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Return compact geometry stats for a candidate group."""

    distances = pairwise_distances([candidate["world_xyz"] for candidate in candidates])
    recompute_errors = [
        candidate["world_recompute_error"]
        for candidate in candidates
        if candidate.get("world_recompute_error") is not None
    ]
    return {
        "num_candidates": len(candidates),
        "views": sorted({candidate["view_id"] for candidate in candidates}),
        "world_distance_stats": distance_stats(distances),
        "mean_world_recompute_error": mean(recompute_errors),
        "extrinsic_source_counts": count_values(candidate.get("extrinsic_key") for candidate in candidates),
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate geometry sanity checks across runs."""

    if not runs:
        return {
            "total_runs": 0,
            "mean_top_rank_pairwise_distance": 0.0,
            "mean_same_label_pairwise_distance": 0.0,
            "mean_world_recompute_error": 0.0,
            "extrinsic_source_counts": {},
        }

    same_label_means: list[float] = []
    recompute_errors: list[float] = []
    extrinsic_sources: list[str] = []
    for run in runs:
        same_label_means.extend(
            stats["world_distance_stats"]["mean"]
            for stats in run["label_stats"].values()
            if stats["world_distance_stats"]["count"] > 0
        )
        recompute_errors.extend(
            candidate["world_recompute_error"]
            for candidate in run["candidates"]
            if candidate.get("world_recompute_error") is not None
        )
        extrinsic_sources.extend(candidate.get("extrinsic_key") or "missing" for candidate in run["candidates"])

    return {
        "total_runs": len(runs),
        "mean_top_rank_pairwise_distance": mean(
            run["top_rank_distance_stats"]["mean"] for run in runs if run["top_rank_distance_stats"]["count"] > 0
        ),
        "mean_same_label_pairwise_distance": mean(same_label_means),
        "mean_world_recompute_error": mean(recompute_errors),
        "extrinsic_source_counts": count_values(extrinsic_sources),
    }


def build_conclusion(aggregate: dict[str, Any]) -> str:
    """Return a small rule-based geometry conclusion."""

    same_label_distance = as_float(aggregate.get("mean_same_label_pairwise_distance"))
    recompute_error = as_float(aggregate.get("mean_world_recompute_error"))
    extrinsic_sources = aggregate.get("extrinsic_source_counts", {})
    uses_gl_extrinsic = any("cam2world_gl" in str(key) for key in extrinsic_sources)

    if recompute_error >= 0.1:
        return (
            "Stored world coordinates do not match extrinsic * camera_xyz. "
            "Inspect the lifting transform implementation before tuning fusion."
        )
    if same_label_distance > 0.5 and uses_gl_extrinsic:
        return (
            "World coordinates are internally consistent with the selected extrinsic, "
            "but same-label estimates remain far apart. The selected source is cam2world_gl, "
            "so the next check should compare OpenGL cam2world against extrinsic_cv/OpenCV convention."
        )
    if same_label_distance > 0.5:
        return (
            "World coordinates are internally consistent, but same-label estimates are far apart. "
            "Focus next on detector crop consistency, depth statistics, and camera convention."
        )
    return "Cross-view geometry looks reasonably consistent under the current report thresholds."


def render_markdown(report: dict[str, Any]) -> str:
    """Render a geometry sanity report as Markdown."""

    aggregate = report["aggregate"]
    lines = [
        "# Cross-View Geometry Sanity Report",
        "",
        f"- Source: `{report['source']}`",
        f"- Source type: `{report['source_type']}`",
        f"- Runs: {aggregate['total_runs']}",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| mean_top_rank_pairwise_distance | {format_float(aggregate['mean_top_rank_pairwise_distance'])} |",
        f"| mean_same_label_pairwise_distance | {format_float(aggregate['mean_same_label_pairwise_distance'])} |",
        f"| mean_world_recompute_error | {format_float(aggregate['mean_world_recompute_error'])} |",
        "",
        "Extrinsic sources:",
        "",
    ]
    for source, count in sorted(aggregate["extrinsic_source_counts"].items()):
        lines.append(f"- `{source}`: {count}")

    lines.extend(["", "## Runs", ""])
    for run in report["runs"]:
        lines.extend(render_run_markdown(run))

    lines.extend(["## Conclusion", "", aggregate["conclusion"], ""])
    return "\n".join(lines)


def render_run_markdown(run: dict[str, Any]) -> list[str]:
    """Render one run section."""

    lines = [
        f"### {Path(run['run_dir']).name}",
        "",
        f"- Query: `{run.get('query')}`",
        f"- Selected object: `{run.get('selected_object_id')}`",
        f"- Selection label: `{run.get('selection_label')}`",
        f"- Top-rank pairwise distance mean: {format_float(run['top_rank_distance_stats']['mean'])}",
        "",
        "| view | rank | phrase | det | fused | box_xyxy | camera_xyz | world_xyz | cam_pos | extrinsic | points | depth_ratio | xform_err |",
        "| --- | ---: | --- | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for candidate in sorted(run["candidates"], key=lambda item: (item["view_id"], item["rank"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(candidate["view_id"]),
                    str(candidate["rank"]),
                    str(candidate["phrase"]),
                    format_float(candidate["det_score"]),
                    format_float(candidate["fused_2d_score"]),
                    format_vector(candidate["box_xyxy"]),
                    format_vector(candidate["camera_xyz"]),
                    format_vector(candidate["world_xyz"]),
                    format_vector(candidate["camera_position"]),
                    str(candidate["extrinsic_key"]),
                    str(candidate["num_points"]),
                    format_float(candidate["depth_valid_ratio"]),
                    format_float(candidate["world_recompute_error"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_console_summary(report: dict[str, Any]) -> str:
    """Return a concise terminal summary."""

    aggregate = report["aggregate"]
    return "\n".join(
        [
            "Cross-view geometry report complete",
            f"  Runs:                     {aggregate['total_runs']}",
            f"  Top-rank distance mean:   {format_float(aggregate['mean_top_rank_pairwise_distance'])}",
            f"  Same-label distance mean: {format_float(aggregate['mean_same_label_pairwise_distance'])}",
            f"  World recompute error:    {format_float(aggregate['mean_world_recompute_error'])}",
            f"  Conclusion:               {aggregate['conclusion']}",
        ]
    )


def group_candidates_by_label(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group candidate rows by phrase."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(str(candidate.get("phrase") or ""), []).append(candidate)
    return grouped


def resolve_view_dir(value: Any, run_dir: Path, view_id: str) -> Path:
    """Resolve a view artifact directory."""

    if value:
        path = resolve_path(value)
        if path.exists():
            return path
    return run_dir / "views" / slug(view_id)


def resolve_path(value: Any) -> Path:
    """Resolve a possibly relative artifact path against the project root."""

    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _default_output_path(args: argparse.Namespace, filename: str) -> Path:
    if args.run_dir is not None:
        return args.run_dir / filename
    return args.benchmark_dir / filename


def transform_point(matrix: list[list[float]], point: list[float]) -> list[float]:
    """Apply a 4x4 transform to a 3D point."""

    homogeneous = [point[0], point[1], point[2], 1.0]
    return [sum(matrix[row][column] * homogeneous[column] for column in range(4)) for row in range(3)]


def camera_position(matrix: list[list[float]] | None) -> list[float] | None:
    """Return translation from a 4x4 transform."""

    if matrix is None:
        return None
    return [matrix[0][3], matrix[1][3], matrix[2][3]]


def pairwise_distances(points: list[list[float]]) -> list[float]:
    """Return all pairwise Euclidean distances."""

    distances: list[float] = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distances.append(euclidean_distance(points[i], points[j]))
    return distances


def distance_stats(distances: list[float]) -> dict[str, float]:
    """Return compact stats for distances."""

    if not distances:
        return {"count": 0.0, "min": 0.0, "mean": 0.0, "max": 0.0}
    return {"count": float(len(distances)), "min": min(distances), "mean": mean(distances), "max": max(distances)}


def count_values(values: Iterable[Any]) -> dict[str, int]:
    """Count stringified values."""

    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "missing")
        counts[key] = counts.get(key, 0) + 1
    return counts


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Return Euclidean distance between two 3D points."""

    return math.sqrt(sum((float(a[index]) - float(b[index])) ** 2 for index in range(3)))


def load_json_dict(path: Path) -> dict[str, Any]:
    """Load a JSON object."""

    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}.")
    return data


def load_optional_json_dict(path: Path) -> dict[str, Any]:
    """Load a JSON object if it exists."""

    if not path.exists():
        return {}
    return load_json_dict(path)


def load_json_list(path: Path) -> list[dict[str, Any]]:
    """Load a JSON list of objects."""

    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}, got {type(data).__name__}.")
    return [item for item in data if isinstance(item, dict)]


def mean(values: Iterable[float]) -> float:
    """Return arithmetic mean, or zero for an empty iterable."""

    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def matrix4(value: Any) -> list[list[float]] | None:
    """Parse a 4x4 matrix."""

    return parse_matrix(value, rows=4, columns=4)


def matrix3(value: Any) -> list[list[float]] | None:
    """Parse a 3x3 matrix."""

    return parse_matrix(value, rows=3, columns=3)


def parse_matrix(value: Any, rows: int, columns: int) -> list[list[float]] | None:
    """Parse a numeric matrix from JSON lists."""

    if not isinstance(value, list) or len(value) != rows:
        return None
    parsed: list[list[float]] = []
    for row in value:
        vector_value = vector(row, expected_len=columns)
        if vector_value is None:
            return None
        parsed.append(vector_value)
    return parsed


def xyz(value: Any) -> list[float] | None:
    """Parse a finite 3D vector."""

    return vector(value, expected_len=3)


def vector(value: Any, expected_len: int) -> list[float] | None:
    """Parse a finite numeric vector."""

    if not isinstance(value, list) or len(value) != expected_len:
        return None
    try:
        parsed = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return parsed if all(math.isfinite(item) for item in parsed) else None


def as_float(value: Any) -> float:
    """Coerce value to float with zero fallback."""

    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def as_int(value: Any) -> int:
    """Coerce value to int with zero fallback."""

    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def format_float(value: Any) -> str:
    """Format a numeric value for Markdown."""

    if value is None:
        return "n/a"
    return f"{as_float(value):.4f}"


def format_vector(value: list[float] | None) -> str:
    """Format a vector for compact Markdown tables."""

    if value is None:
        return "n/a"
    return "[" + ", ".join(f"{item:.3f}" for item in value) + "]"


def slug(value: str) -> str:
    """Return the same slug convention used by multiview debug artifacts."""

    text = "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
    return text[:40] or "view"


if __name__ == "__main__":
    raise SystemExit(main())
