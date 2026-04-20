from __future__ import annotations

import argparse
from pathlib import Path
import sys

from scripts.run_ambiguity_benchmark import (
    DEFAULT_QUERIES_FILE,
    build_benchmark_command,
    build_report_command,
    load_ambiguity_queries,
)


def test_default_ambiguity_queries_load() -> None:
    queries = load_ambiguity_queries(DEFAULT_QUERIES_FILE)

    assert "red cube" in queries
    assert "object" in queries
    assert "blue cup" in queries
    assert all(not query.startswith("#") for query in queries)


def test_load_ambiguity_queries_ignores_comments_and_blanks(tmp_path: Path) -> None:
    queries_file = tmp_path / "queries.txt"
    queries_file.write_text("# comment\n\ncube\n  mug  \n", encoding="utf-8")

    assert load_ambiguity_queries(queries_file) == ["cube", "mug"]


def test_build_benchmark_command_defaults_to_clip_enabled(tmp_path: Path) -> None:
    args = _args(tmp_path, skip_clip=False)

    command = build_benchmark_command(args)

    assert command[0] == sys.executable
    assert "--queries-file" in command
    assert "--skip-clip" not in command
    assert "--use-clip" in command
    assert command[command.index("--output-dir") + 1] == str(args.output_dir)


def test_build_benchmark_command_forwards_skip_clip_and_seeds(tmp_path: Path) -> None:
    args = _args(tmp_path, skip_clip=True)

    command = build_benchmark_command(args)

    assert "--skip-clip" in command
    assert command[command.index("--seeds") + 1 : command.index("--skip-clip")] == ["0", "1"]


def test_build_report_command_targets_existing_report_generator(tmp_path: Path) -> None:
    output_dir = tmp_path / "ambiguity"

    command = build_report_command(output_dir)

    assert command[0] == sys.executable
    assert command[-2:] == ["--benchmark-dir", str(output_dir)]


def _args(tmp_path: Path, skip_clip: bool) -> argparse.Namespace:
    return argparse.Namespace(
        queries_file=DEFAULT_QUERIES_FILE,
        seeds=[0, 1],
        num_runs=1,
        output_dir=tmp_path / "ambiguity",
        detector_backend="mock",
        mock_box_position="center",
        skip_clip=skip_clip,
        depth_scale=1000.0,
        env_id="PickCube-v1",
        obs_mode="rgbd",
    )
