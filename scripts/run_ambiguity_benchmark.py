"""Thin wrapper for ambiguity-focused single-view pick benchmarks."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES_FILE = PROJECT_ROOT / "configs" / "ambiguity_queries.txt"

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the curated ambiguity benchmark through the existing single-view runner.")
    parser.add_argument("--queries-file", type=Path, default=DEFAULT_QUERIES_FILE, help="Ambiguity query file. Blank lines and # comments are ignored.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Integer seeds passed to the benchmark runner.")
    parser.add_argument("--num-runs", type=int, default=1, help="Fallback number of seeds when --seeds is omitted.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "ambiguity_benchmark")
    parser.add_argument(
        "--detector-backend",
        default="mock",
        choices=["auto", "hf", "transformers", "groundingdino", "original", "mock"],
    )
    parser.add_argument("--mock-box-position", default="center", choices=["center", "left", "right", "all"])
    parser.add_argument("--skip-clip", dest="skip_clip", action="store_true", default=False)
    parser.add_argument("--use-clip", dest="skip_clip", action="store_false", help="Run CLIP reranking. This is the default unless --skip-clip is set.")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", default="rgbd")
    parser.add_argument("--generate-report", action="store_true", help="Run the existing report generator after the benchmark.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")

    queries = load_ambiguity_queries(args.queries_file)
    LOGGER.info("Loaded %s ambiguity queries from %s", len(queries), args.queries_file)

    benchmark_command = build_benchmark_command(args)
    run_command(benchmark_command)

    if args.generate_report:
        run_command(build_report_command(args.output_dir))

    print(f"Ambiguity benchmark complete: {args.output_dir}")
    return 0


def load_ambiguity_queries(path: str | Path) -> list[str]:
    """Load non-empty, non-comment query lines from an ambiguity query file."""

    query_path = Path(path)
    with query_path.open("r", encoding="utf-8") as file:
        queries = [line.strip() for line in file if line.strip() and not line.lstrip().startswith("#")]
    if not queries:
        raise ValueError(f"No ambiguity queries found in {query_path}.")
    return queries


def build_benchmark_command(args: argparse.Namespace) -> list[str]:
    """Build the existing benchmark-runner command for the ambiguity query file."""

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_single_view_pick_benchmark.py"),
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
        "--output-dir",
        str(args.output_dir),
    ]
    if args.seeds:
        command.append("--seeds")
        command.extend(str(seed) for seed in args.seeds)
    if args.skip_clip:
        command.append("--skip-clip")
    else:
        command.append("--use-clip")
    return command


def build_report_command(output_dir: str | Path) -> list[str]:
    """Build the existing report-generator command for a benchmark directory."""

    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_benchmark_report.py"),
        "--benchmark-dir",
        str(output_dir),
    ]


def run_command(command: list[str]) -> None:
    """Run a child command from the project root."""

    LOGGER.info("Running: %s", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
