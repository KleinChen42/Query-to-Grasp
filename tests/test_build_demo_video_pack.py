from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.build_demo_video_pack import (
    DemoStorySpec,
    build_demo_story_specs,
    build_demo_video_pack,
    parse_story_spec,
)


def write_benchmark(source: Path, rows: list[dict[str, object]], with_media: bool = False) -> None:
    source.mkdir(parents=True)
    (source / "benchmark_summary.json").write_text(
        json.dumps(
            {
                "env_id": "StackCube-v1",
                "pick_executor": "sim_pick_place",
                "place_target_source": "oracle_cubeB_pose",
                "aggregate_metrics": {
                    "total_runs": len(rows),
                    "failed_runs": 0,
                    "pick_success_rate": 0.5,
                    "place_success_rate": 0.5,
                    "task_success_rate": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )
    with (source / "benchmark_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query",
                "seed",
                "pick_success",
                "place_success",
                "task_success",
                "place_attempted",
                "pick_stage",
                "artifacts",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    if with_media:
        media_dir = source / "runs" / "run_0001_seed_0"
        media_dir.mkdir(parents=True)
        (media_dir / "frame.png").write_bytes(b"not a real png")


def test_parse_story_spec_accepts_minimal_and_caption_forms() -> None:
    minimal = parse_story_spec("demo=outputs/demo")
    full = parse_story_spec("demo=outputs/demo::Caption.")
    with_metadata = parse_story_spec("demo=outputs/demo::Caption.::seed=7,mode=closed_loop")

    assert minimal.label == "demo"
    assert minimal.source_dir == Path("outputs/demo")
    assert "demo" in minimal.caption
    assert full.caption == "Caption."
    assert with_metadata.row_metadata == {"seed": "7", "mode": "closed_loop"}


def test_build_demo_story_specs_can_skip_defaults() -> None:
    specs = build_demo_story_specs(["demo=outputs/demo::Caption."], include_defaults=False)

    assert specs == [
        DemoStorySpec(
            label="demo",
            source_dir=Path("outputs/demo"),
            caption="Caption.",
            desired_outcomes=("success", "failure"),
        )
    ]


def test_build_demo_video_pack_selects_rows_and_requests_capture(tmp_path: Path) -> None:
    source = tmp_path / "benchmark"
    write_benchmark(
        source,
        rows=[
            {
                "query": "red cube",
                "seed": 0,
                "pick_success": True,
                "place_success": True,
                "task_success": True,
                "place_attempted": True,
                "pick_stage": "success",
                "artifacts": "",
            },
            {
                "query": "red cube",
                "seed": 1,
                "pick_success": False,
                "place_success": False,
                "task_success": False,
                "place_attempted": True,
                "pick_stage": "place_not_confirmed",
                "artifacts": "",
            },
        ],
    )

    output_dir = tmp_path / "pack"
    manifest = build_demo_video_pack(
        specs=[
            DemoStorySpec(
                label="stackcube_contrast",
                source_dir=source,
                caption="Contrast story.",
                desired_outcomes=("success", "failure"),
            )
        ],
        output_dir=output_dir,
    )

    assert manifest["included_stories"] == 1
    assert manifest["capture_request_count"] == 1
    assert [row["seed"] for row in manifest["stories"][0]["selected_rows"]] == ["0", "1"]
    assert (output_dir / "README.md").exists()
    assert "Capture Requests" in (output_dir / "README.md").read_text(encoding="utf-8")
    loaded = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert loaded["stories"][0]["metrics"]["task_success_rate"] == 0.5


def test_build_demo_video_pack_copies_existing_media(tmp_path: Path) -> None:
    source = tmp_path / "benchmark"
    write_benchmark(
        source,
        rows=[
            {
                "query": "red cube",
                "seed": 0,
                "pick_success": True,
                "place_success": True,
                "task_success": True,
                "place_attempted": True,
                "pick_stage": "success",
                "artifacts": str(source / "runs" / "run_0001_seed_0"),
            }
        ],
        with_media=True,
    )

    manifest = build_demo_video_pack(
        specs=[DemoStorySpec(label="demo", source_dir=source, caption="Demo.")],
        output_dir=tmp_path / "pack",
    )

    assert manifest["capture_request_count"] == 0
    copied = manifest["stories"][0]["copied_media"]
    assert len(copied) == 1
    assert (tmp_path / "pack" / copied[0]).exists()
    assert "slideshow_path" in manifest["stories"][0]


def test_build_demo_video_pack_prefers_execution_video(tmp_path: Path) -> None:
    source = tmp_path / "debug_capture"
    run_dir = source / "20260501_red_cube"
    video_dir = run_dir / "execution_video"
    video_dir.mkdir(parents=True)
    (video_dir / "execution_video.mp4").write_bytes(b"fake mp4")
    (run_dir / "detection_overlay.png").write_bytes(b"fake png")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "query": "red cube",
                "artifacts": str(run_dir),
                "pick_success": True,
                "task_success": True,
                "pick_stage": "success",
            }
        ),
        encoding="utf-8",
    )

    manifest = build_demo_video_pack(
        specs=[DemoStorySpec(label="debug", source_dir=source, caption="Debug capture.")],
        output_dir=tmp_path / "pack",
    )

    story = manifest["stories"][0]
    assert story["media_kind"] == "execution_video"
    assert story["slideshow_path"].endswith("execution_video.mp4")


def test_build_demo_video_pack_reads_direct_debug_summary(tmp_path: Path) -> None:
    source = tmp_path / "debug_capture"
    run_dir = source / "20260430_red_cube"
    media_dir = run_dir / "view_front"
    media_dir.mkdir(parents=True)
    (media_dir / "rgb.png").write_bytes(b"not a real png")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "query": "red cube",
                "artifacts": str(run_dir),
                "pick_success": True,
                "place_success": True,
                "task_success": True,
                "pick_stage": "success",
                "pick_target_source": "memory_grasp_world_xyz",
                "place_target_source": "oracle_cubeB_pose",
            }
        ),
        encoding="utf-8",
    )

    manifest = build_demo_video_pack(
        specs=[DemoStorySpec(label="debug", source_dir=source, caption="Debug capture.")],
        output_dir=tmp_path / "pack",
    )

    story = manifest["stories"][0]
    assert story["available_media_count"] == 1
    assert story["metrics"]["pick_success_rate"] == 1.0
    assert story["metrics"]["place_success_rate"] == 1.0
    assert story["metrics"]["task_success_rate"] == 1.0
    assert story["metrics"]["pick_target_sources"] == ["memory_grasp_world_xyz"]
    assert story["selected_rows"][0]["pick_target_source"] == "memory_grasp_world_xyz"


def test_build_demo_video_pack_applies_story_metadata(tmp_path: Path) -> None:
    source = tmp_path / "debug_capture"
    run_dir = source / "20260430_red_cube"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "query": "red cube",
                "artifacts": str(run_dir),
                "pick_success": True,
                "task_success": True,
                "pick_stage": "success",
            }
        ),
        encoding="utf-8",
    )

    manifest = build_demo_video_pack(
        specs=[
            DemoStorySpec(
                label="debug",
                source_dir=source,
                caption="Debug capture.",
                row_metadata={"seed": "5", "mode": "closed_loop"},
            )
        ],
        output_dir=tmp_path / "pack",
    )

    selected = manifest["stories"][0]["selected_rows"][0]
    assert selected["seed"] == "5"
    assert selected["mode"] == "closed_loop"
