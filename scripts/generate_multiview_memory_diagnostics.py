"""Generate lightweight diagnostics for multi-view object memory merging."""

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


DEFAULT_MERGE_DISTANCES = [0.05, 0.08, 0.12, 0.16, 0.24, 0.32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze whether multi-view memory objects are merging cleanly.")
    parser.add_argument("--benchmark-dir", type=Path, required=True, help="Fusion benchmark output directory.")
    parser.add_argument(
        "--merge-distances",
        nargs="*",
        type=float,
        default=DEFAULT_MERGE_DISTANCES,
        help="Candidate merge-distance thresholds to simulate.",
    )
    parser.add_argument("--output-md", type=Path, default=None, help="Markdown diagnostics output path.")
    parser.add_argument("--output-json", type=Path, default=None, help="JSON diagnostics output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    diagnostics = build_diagnostics(args.benchmark_dir, args.merge_distances)

    output_json = args.output_json or args.benchmark_dir / "memory_diagnostics.json"
    output_md = args.output_md or args.benchmark_dir / "memory_diagnostics.md"
    write_json(diagnostics, output_json)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(diagnostics), encoding="utf-8")

    print(f"Wrote memory diagnostics JSON: {output_json}")
    print(f"Wrote memory diagnostics MD:   {output_md}")
    print(render_console_summary(diagnostics))
    return 0


def build_diagnostics(benchmark_dir: str | Path, merge_distances: Iterable[float]) -> dict[str, Any]:
    """Build diagnostics from a fusion benchmark directory."""

    benchmark_path = Path(benchmark_dir)
    rows = load_json_list(benchmark_path / "benchmark_rows.json")
    summary = load_json_dict(benchmark_path / "benchmark_summary.json")
    thresholds = sorted({float(distance) for distance in merge_distances if float(distance) > 0.0})
    if not thresholds:
        raise ValueError("At least one positive merge distance is required.")

    run_diagnostics = [diagnose_run(row, thresholds, benchmark_path) for row in rows]
    aggregate = aggregate_run_diagnostics(run_diagnostics, thresholds)
    aggregate["conclusion"] = build_conclusion(aggregate, summary)
    return {
        "benchmark_dir": str(benchmark_path),
        "detector_backend": summary.get("detector_backend"),
        "skip_clip": summary.get("skip_clip"),
        "view_preset": summary.get("view_preset", "none"),
        "merge_distances": thresholds,
        "benchmark_summary": summary.get("aggregate_metrics", {}),
        "aggregate": aggregate,
        "runs": run_diagnostics,
    }


def diagnose_run(row: dict[str, Any], thresholds: list[float], benchmark_dir: Path) -> dict[str, Any]:
    """Build diagnostics for one benchmark row."""

    artifacts = Path(str(row.get("artifacts") or ""))
    if not artifacts.is_absolute():
        artifacts = PROJECT_ROOT / artifacts
    memory_state_path = artifacts / "memory_state.json"
    if not memory_state_path.exists():
        fallback = benchmark_dir / artifacts / "memory_state.json"
        memory_state_path = fallback if fallback.exists() else memory_state_path

    memory_state = load_json_dict(memory_state_path)
    observations = extract_candidate_observations(memory_state)
    xyzs = [obs["world_xyz"] for obs in observations]
    all_distances = pairwise_distances(xyzs)
    same_label_distances = pairwise_distances_by_label(observations)
    sweep = {str(distance): simulate_merge_count(observations, distance) for distance in thresholds}

    return {
        "query": row.get("query"),
        "seed": row.get("seed"),
        "artifacts": str(artifacts),
        "memory_state_path": str(memory_state_path),
        "num_views": _as_int(row.get("num_views")),
        "num_observations_added": _as_int(row.get("num_observations_added")),
        "num_candidate_observations": len(observations),
        "num_memory_objects": _as_int(row.get("num_memory_objects")),
        "selected_object_id": row.get("selected_object_id"),
        "selected_overall_confidence": _as_float(row.get("selected_overall_confidence")),
        "pairwise_distance_stats": distance_stats(all_distances),
        "same_label_pairwise_distance_stats": distance_stats(same_label_distances),
        "simulated_memory_objects_by_merge_distance": sweep,
        "labels": sorted({obs["label"] for obs in observations if obs["label"]}),
    }


def extract_candidate_observations(memory_state: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract valid 3D candidate observations from a memory_state artifact."""

    observations: list[dict[str, Any]] = []
    for view in memory_state.get("views", []):
        if not isinstance(view, dict):
            continue
        view_id = str(view.get("view_id") or "")
        ranked_candidates = view.get("reranked_candidates", [])
        candidates_3d = view.get("candidates_3d", [])
        if not isinstance(ranked_candidates, list) or not isinstance(candidates_3d, list):
            continue
        for index, candidate_3d in enumerate(candidates_3d):
            if not isinstance(candidate_3d, dict):
                continue
            world_xyz = _xyz(candidate_3d.get("world_xyz"))
            if world_xyz is None:
                continue
            ranked = ranked_candidates[index] if index < len(ranked_candidates) and isinstance(ranked_candidates[index], dict) else {}
            observations.append(
                {
                    "view_id": view_id,
                    "rank": index,
                    "label": str(ranked.get("phrase") or ""),
                    "world_xyz": world_xyz,
                    "num_points": _as_int(candidate_3d.get("num_points")),
                    "depth_valid_ratio": _as_float(candidate_3d.get("depth_valid_ratio")),
                    "det_score": _as_float(ranked.get("det_score")),
                    "fused_2d_score": _as_float(ranked.get("fused_2d_score")),
                }
            )
    return observations


def simulate_merge_count(observations: list[dict[str, Any]], merge_distance: float) -> int:
    """Simulate the current greedy spatial merge rule for a given threshold."""

    object_centers: list[list[float]] = []
    object_counts: list[int] = []
    for observation in observations:
        xyz = observation["world_xyz"]
        match_index = nearest_within_threshold(object_centers, xyz, merge_distance)
        if match_index is None:
            object_centers.append(list(xyz))
            object_counts.append(1)
            continue
        count = object_counts[match_index]
        object_centers[match_index] = [
            (object_centers[match_index][axis] * count + xyz[axis]) / (count + 1)
            for axis in range(3)
        ]
        object_counts[match_index] = count + 1
    return len(object_centers)


def nearest_within_threshold(
    centers: list[list[float]],
    xyz: list[float],
    threshold: float,
) -> int | None:
    """Return the nearest center index within a threshold."""

    matches = [
        (euclidean_distance(center, xyz), index)
        for index, center in enumerate(centers)
        if euclidean_distance(center, xyz) <= threshold
    ]
    if not matches:
        return None
    return sorted(matches)[0][1]


def aggregate_run_diagnostics(run_diagnostics: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    """Aggregate diagnostics across runs."""

    if not run_diagnostics:
        return {
            "total_runs": 0,
            "mean_num_candidate_observations": 0.0,
            "mean_num_memory_objects": 0.0,
            "mean_same_label_pairwise_distance": 0.0,
            "mean_simulated_memory_objects_by_merge_distance": {str(distance): 0.0 for distance in thresholds},
        }

    return {
        "total_runs": len(run_diagnostics),
        "mean_num_candidate_observations": mean(run["num_candidate_observations"] for run in run_diagnostics),
        "mean_num_memory_objects": mean(run["num_memory_objects"] for run in run_diagnostics),
        "mean_same_label_pairwise_distance": mean(
            run["same_label_pairwise_distance_stats"]["mean"] for run in run_diagnostics
        ),
        "mean_simulated_memory_objects_by_merge_distance": {
            str(distance): mean(
                run["simulated_memory_objects_by_merge_distance"][str(distance)] for run in run_diagnostics
            )
            for distance in thresholds
        },
    }


def build_conclusion(aggregate: dict[str, Any], benchmark_summary: dict[str, Any]) -> str:
    """Return a small rule-based diagnostic conclusion."""

    metrics = benchmark_summary.get("aggregate_metrics", {})
    mean_views = _as_float(metrics.get("mean_num_views"))
    mean_memory_objects = _as_float(metrics.get("mean_num_memory_objects"))
    mean_same_label_distance = _as_float(aggregate.get("mean_same_label_pairwise_distance"))
    sweep = aggregate.get("mean_simulated_memory_objects_by_merge_distance", {})
    default_objects = _as_float(sweep.get("0.08", mean_memory_objects))
    relaxed_objects = _as_float(sweep.get(_largest_distance_key(sweep), default_objects))

    if mean_views >= 3.0 and mean_memory_objects > 2.0 and mean_same_label_distance > 0.5 and relaxed_objects > 2.0:
        return (
            "Multi-view capture is working, but same-label 3D target estimates are far apart across views. "
            "This points to cross-view 3D lifting, camera-pose alignment, or detector consistency as the next bottleneck, "
            "not only a conservative merge threshold."
        )
    if mean_views >= 3.0 and mean_memory_objects > 2.0 and relaxed_objects < default_objects:
        return (
            "Multi-view capture is working, but object memory still fragments targets. "
            "The merge-distance sweep suggests the current merge threshold is conservative; "
            "inspect 3D lifting consistency before increasing it globally."
        )
    if mean_views >= 3.0 and mean_memory_objects <= 2.0:
        return "Multi-view capture and memory merging look reasonably stable for this benchmark."
    return "Diagnostics are inconclusive; run with at least three views and multiple seeds for a stronger signal."


def render_markdown(diagnostics: dict[str, Any]) -> str:
    """Render diagnostics as Markdown."""

    aggregate = diagnostics["aggregate"]
    lines = [
        "# Multi-View Memory Diagnostics",
        "",
        f"- Benchmark: `{diagnostics['benchmark_dir']}`",
        f"- Detector backend: `{diagnostics.get('detector_backend')}`",
        f"- View preset: `{diagnostics.get('view_preset')}`",
        f"- Runs: {aggregate['total_runs']}",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| mean_num_candidate_observations | {_format_float(aggregate['mean_num_candidate_observations'])} |",
        f"| mean_num_memory_objects | {_format_float(aggregate['mean_num_memory_objects'])} |",
        f"| mean_same_label_pairwise_distance | {_format_float(aggregate['mean_same_label_pairwise_distance'])} |",
        "",
        "## Merge-Distance Sweep",
        "",
        "| merge_distance | mean_simulated_memory_objects |",
        "| ---: | ---: |",
    ]
    for distance, value in aggregate["mean_simulated_memory_objects_by_merge_distance"].items():
        lines.append(f"| {distance} | {_format_float(value)} |")

    lines.extend(
        [
            "",
            "## Per-Run",
            "",
            "| query | seed | views | observations | memory_objects | same_label_mean_dist | objects_at_0.08 | objects_at_0.24 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in diagnostics["runs"]:
        sweep = run["simulated_memory_objects_by_merge_distance"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(run["query"]),
                    str(run["seed"]),
                    str(run["num_views"]),
                    str(run["num_candidate_observations"]),
                    str(run["num_memory_objects"]),
                    _format_float(run["same_label_pairwise_distance_stats"]["mean"]),
                    str(sweep.get("0.08", "n/a")),
                    str(sweep.get("0.24", "n/a")),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Conclusion", "", aggregate["conclusion"], ""])
    return "\n".join(lines)


def render_console_summary(diagnostics: dict[str, Any]) -> str:
    """Return a concise terminal summary."""

    aggregate = diagnostics["aggregate"]
    return "\n".join(
        [
            "Multi-view memory diagnostics complete",
            f"  Runs:                 {aggregate['total_runs']}",
            f"  Candidate obs/run:    {_format_float(aggregate['mean_num_candidate_observations'])}",
            f"  Memory objects/run:   {_format_float(aggregate['mean_num_memory_objects'])}",
            f"  Same-label dist mean: {_format_float(aggregate['mean_same_label_pairwise_distance'])}",
            f"  Conclusion:           {aggregate['conclusion']}",
        ]
    )


def pairwise_distances(points: list[list[float]]) -> list[float]:
    """Return all pairwise Euclidean distances."""

    distances: list[float] = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distances.append(euclidean_distance(points[i], points[j]))
    return distances


def pairwise_distances_by_label(observations: list[dict[str, Any]]) -> list[float]:
    """Return pairwise distances only among observations with the same label."""

    distances: list[float] = []
    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            if observations[i]["label"] and observations[i]["label"] == observations[j]["label"]:
                distances.append(euclidean_distance(observations[i]["world_xyz"], observations[j]["world_xyz"]))
    return distances


def distance_stats(distances: list[float]) -> dict[str, float]:
    """Return compact summary statistics for distances."""

    if not distances:
        return {"count": 0, "min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "count": float(len(distances)),
        "min": min(distances),
        "mean": mean(distances),
        "max": max(distances),
    }


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Return 3D Euclidean distance."""

    return math.sqrt(sum((float(a[index]) - float(b[index])) ** 2 for index in range(3)))


def load_json_dict(path: Path) -> dict[str, Any]:
    """Load a JSON object."""

    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}.")
    return data


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


def _xyz(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        xyz = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return xyz if all(math.isfinite(item) for item in xyz) else None


def _as_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_float(value: float) -> str:
    return f"{float(value):.4f}"


def _largest_distance_key(sweep: dict[str, Any]) -> str:
    if not sweep:
        return ""
    return str(max((float(key), key) for key in sweep)[1])


if __name__ == "__main__":
    raise SystemExit(main())
