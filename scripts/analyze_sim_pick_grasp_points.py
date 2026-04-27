"""Diagnose simulated pick target points against oracle object poses."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class BenchmarkSpec:
    label: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze simulated pick target points against oracle object poses.")
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="Benchmark directory containing benchmark_rows.json/csv. Can be repeated.",
    )
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", default="rgbd")
    parser.add_argument("--control-mode", default="pd_ee_delta_pos")
    parser.add_argument("--skip-oracle", action="store_true", help="Do not reset ManiSkill to collect oracle poses.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = [_parse_benchmark_spec(raw) for raw in args.benchmark]
    if not specs:
        raise ValueError("Provide at least one --benchmark LABEL=DIR.")

    rows_by_benchmark = {spec.label: load_rows(spec.path) for spec in specs}
    seeds = sorted({int(row.get("seed", 0)) for rows in rows_by_benchmark.values() for row in rows})
    oracle_by_seed = (
        {}
        if args.skip_oracle
        else collect_oracle_targets(
            seeds=seeds,
            env_id=args.env_id,
            obs_mode=args.obs_mode,
            control_mode=args.control_mode,
        )
    )
    analyses: list[dict[str, Any]] = []
    for spec in specs:
        for row in rows_by_benchmark[spec.label]:
            analyses.append(analyze_row(benchmark=spec.label, row=row, oracle_by_seed=oracle_by_seed))

    report = {
        "benchmarks": [{"label": spec.label, "path": str(spec.path)} for spec in specs],
        "env_id": args.env_id,
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "num_rows": len(analyses),
        "oracle_available": bool(oracle_by_seed),
        "aggregate_by_benchmark": aggregate_by_key(analyses, "benchmark"),
        "aggregate_by_query": aggregate_by_key(analyses, "query"),
        "aggregate_by_success": aggregate_by_key(analyses, "pick_success"),
        "rows": analyses,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


def _parse_benchmark_spec(raw: str) -> BenchmarkSpec:
    if "=" not in raw:
        raise ValueError(f"Benchmark spec must be LABEL=DIR, got {raw!r}.")
    label, path = raw.split("=", 1)
    label = label.strip()
    path = Path(path.strip())
    if not label or not path:
        raise ValueError(f"Benchmark spec must be LABEL=DIR, got {raw!r}.")
    return BenchmarkSpec(label=label, path=path)


def load_rows(benchmark_dir: Path) -> list[dict[str, Any]]:
    rows_json = benchmark_dir / "benchmark_rows.json"
    rows_csv = benchmark_dir / "benchmark_rows.csv"
    if rows_json.exists():
        rows = json.loads(rows_json.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{rows_json} must contain a list.")
        return [dict(row) for row in rows]
    if rows_csv.exists():
        with rows_csv.open("r", encoding="utf-8", newline="") as file:
            return [dict(row) for row in csv.DictReader(file)]
    raise FileNotFoundError(f"Could not find benchmark_rows.json/csv under {benchmark_dir}.")


def collect_oracle_targets(
    seeds: list[int],
    env_id: str,
    obs_mode: str,
    control_mode: str,
) -> dict[int, list[float]]:
    from scripts.run_oracle_pick_smoke import find_oracle_target_xyz
    from src.env.maniskill_env import ManiSkillScene

    scene = ManiSkillScene(env_name=env_id, obs_mode=obs_mode, control_mode=control_mode)
    targets: dict[int, list[float]] = {}
    try:
        for seed in seeds:
            scene.reset(seed=seed)
            xyz = find_oracle_target_xyz(scene.env)
            if xyz is not None:
                targets[int(seed)] = np.asarray(xyz, dtype=np.float32).reshape(3).astype(float).tolist()
    finally:
        scene.close()
    return targets


def analyze_row(benchmark: str, row: dict[str, Any], oracle_by_seed: dict[int, list[float]]) -> dict[str, Any]:
    artifact_dir = Path(str(row.get("artifacts", "")))
    summary = _read_json_if_exists(artifact_dir / "summary.json")
    pick_result = _read_json_if_exists(artifact_dir / "pick_result.json")

    seed = int(row.get("seed", summary.get("seed", 0)))
    target_xyz = _vector3(pick_result.get("target_xyz") or summary.get("world_xyz"))
    oracle_xyz = _vector3(oracle_by_seed.get(seed))
    final_tcp_xyz = _vector3((pick_result.get("trajectory_summary") or {}).get("final_tcp_xyz"))

    metrics = {}
    if target_xyz is not None and oracle_xyz is not None:
        target_delta = target_xyz - oracle_xyz
        metrics.update(
            {
                "target_minus_oracle_xyz": target_delta.astype(float).tolist(),
                "target_xy_error_to_oracle": float(np.linalg.norm(target_delta[:2])),
                "target_z_error_to_oracle": float(target_delta[2]),
                "target_distance_to_oracle": float(np.linalg.norm(target_delta)),
                "target_z_above_oracle_gt_5cm": bool(target_delta[2] > 0.05),
                "target_xy_error_gt_5cm": bool(np.linalg.norm(target_delta[:2]) > 0.05),
            }
        )
    if final_tcp_xyz is not None and target_xyz is not None:
        final_delta = final_tcp_xyz - target_xyz
        metrics.update(
            {
                "final_tcp_minus_target_xyz": final_delta.astype(float).tolist(),
                "final_tcp_xy_error_to_target": float(np.linalg.norm(final_delta[:2])),
                "final_tcp_z_above_target": float(final_delta[2]),
            }
        )
    if final_tcp_xyz is not None and oracle_xyz is not None:
        final_oracle_delta = final_tcp_xyz - oracle_xyz
        metrics.update(
            {
                "final_tcp_xy_error_to_oracle": float(np.linalg.norm(final_oracle_delta[:2])),
                "final_tcp_z_above_oracle": float(final_oracle_delta[2]),
            }
        )

    pick_success = _bool_value(row.get("pick_success", pick_result.get("pick_success", pick_result.get("success"))))
    result = {
        "benchmark": benchmark,
        "query": row.get("query", summary.get("query")),
        "seed": seed,
        "pick_success": pick_success,
        "pick_stage": row.get("pick_stage", pick_result.get("stage")),
        "num_3d_points": _float_or_none(row.get("num_3d_points", summary.get("num_3d_points"))),
        "detector_top_phrase": row.get("detector_top_phrase", summary.get("detector_top_phrase")),
        "final_top_phrase": row.get("final_top_phrase", summary.get("final_top_phrase")),
        "target_xyz": None if target_xyz is None else target_xyz.astype(float).tolist(),
        "oracle_xyz": None if oracle_xyz is None else oracle_xyz.astype(float).tolist(),
        "final_tcp_xyz": None if final_tcp_xyz is None else final_tcp_xyz.astype(float).tolist(),
        "artifact_dir": str(artifact_dir),
    }
    result.update(metrics)
    return result


def aggregate_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key)), []).append(row)
    return {name: aggregate_rows(group_rows) for name, group_rows in sorted(grouped.items())}


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    return {
        "total_runs": total,
        "pick_success_rate": _mean_bool(row.get("pick_success") for row in rows),
        "mean_num_3d_points": _mean(row.get("num_3d_points") for row in rows),
        "mean_target_xy_error_to_oracle": _mean(row.get("target_xy_error_to_oracle") for row in rows),
        "mean_target_z_error_to_oracle": _mean(row.get("target_z_error_to_oracle") for row in rows),
        "mean_target_distance_to_oracle": _mean(row.get("target_distance_to_oracle") for row in rows),
        "target_z_above_oracle_gt_5cm_rate": _mean_bool(row.get("target_z_above_oracle_gt_5cm") for row in rows),
        "target_xy_error_gt_5cm_rate": _mean_bool(row.get("target_xy_error_gt_5cm") for row in rows),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Simulated Grasp Point Diagnosis",
        "",
        f"- Env: `{report['env_id']}`",
        f"- Control mode: `{report['control_mode']}`",
        f"- Rows: `{report['num_rows']}`",
        f"- Oracle poses available: `{report['oracle_available']}`",
        "",
        "## By Benchmark",
        "",
        _aggregate_table(report["aggregate_by_benchmark"]),
        "",
        "## By Query",
        "",
        _aggregate_table(report["aggregate_by_query"]),
        "",
        "## Success vs Failure",
        "",
        _aggregate_table(report["aggregate_by_success"]),
        "",
        "## Largest Target Height Errors",
        "",
        "| benchmark | query | seed | success | points | target_z_error | xy_error | target_xyz | oracle_xyz |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    rows = sorted(
        report["rows"],
        key=lambda row: abs(float(row.get("target_z_error_to_oracle") or 0.0)),
        reverse=True,
    )
    for row in rows[:12]:
        lines.append(
            "| {benchmark} | {query} | {seed} | {pick_success} | {points:.1f} | {zerr:.4f} | {xyerr:.4f} | `{target}` | `{oracle}` |".format(
                benchmark=row.get("benchmark"),
                query=row.get("query"),
                seed=row.get("seed"),
                pick_success=row.get("pick_success"),
                points=float(row.get("num_3d_points") or 0.0),
                zerr=float(row.get("target_z_error_to_oracle") or 0.0),
                xyerr=float(row.get("target_xy_error_to_oracle") or 0.0),
                target=_format_vector(row.get("target_xyz")),
                oracle=_format_vector(row.get("oracle_xyz")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _aggregate_table(aggregate: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| group | runs | pick success | mean points | mean xy err | mean z err | high-z rate | far-xy rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in aggregate.items():
        lines.append(
            "| {name} | {runs} | {pick:.4f} | {points:.4f} | {xy:.4f} | {z:.4f} | {high:.4f} | {far:.4f} |".format(
                name=name,
                runs=metrics["total_runs"],
                pick=float(metrics.get("pick_success_rate") or 0.0),
                points=float(metrics.get("mean_num_3d_points") or 0.0),
                xy=float(metrics.get("mean_target_xy_error_to_oracle") or 0.0),
                z=float(metrics.get("mean_target_z_error_to_oracle") or 0.0),
                high=float(metrics.get("target_z_above_oracle_gt_5cm_rate") or 0.0),
                far=float(metrics.get("target_xy_error_gt_5cm_rate") or 0.0),
            )
        )
    return "\n".join(lines)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _vector3(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float32).reshape(3)
    except Exception:
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _mean(values: Any) -> float | None:
    numbers = [_float_or_none(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    if not numbers:
        return None
    return float(sum(numbers) / len(numbers))


def _mean_bool(values: Any) -> float | None:
    bools = [value for value in values if value is not None]
    if not bools:
        return None
    return float(sum(1 for value in bools if _bool_value(value)) / len(bools))


def _format_vector(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        values = [float(item) for item in value]
    except TypeError:
        return "n/a"
    return "[" + ", ".join(f"{item:.4f}" for item in values) + "]"


if __name__ == "__main__":
    main()
