"""Build a compact demo-video planning pack from existing benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.export_utils import write_json  # noqa: E402


MEDIA_SUFFIXES = {".gif", ".jpeg", ".jpg", ".mp4", ".png", ".webm"}


@dataclass(frozen=True)
class DemoStorySpec:
    """One benchmark source to summarize for demo/video planning."""

    label: str
    source_dir: Path
    caption: str
    desired_outcomes: tuple[str, ...] = ("success",)
    row_metadata: dict[str, str] | None = None


DEFAULT_STORIES = (
    DemoStorySpec(
        label="pickcube_full_query_success",
        source_dir=Path("outputs/h200_60071_multiview_memory_grasp_point_full_ambiguity_seed01234/tabletop_no_clip"),
        caption=(
            "PickCube full-query multi-view success: fused memory grasp points "
            "serve as executable simulated pick targets."
        ),
        desired_outcomes=("success",),
    ),
    DemoStorySpec(
        label="stackcube_query_place_success",
        source_dir=Path("outputs/h200_60071_query_stackcube_place_bridge_seed0_49/single_with_clip"),
        caption=(
            "StackCube query-pick plus oracle-place success: the cubeA pick "
            "target is query-derived, while cubeB placement remains privileged."
        ),
        desired_outcomes=("success",),
    ),
    DemoStorySpec(
        label="stackcube_tabletop_target_source_contrast",
        source_dir=Path("outputs/h200_60071_query_stackcube_place_bridge_seed0_49/tabletop_no_clip"),
        caption=(
            "StackCube tabletop target-source contrast: representative success "
            "and failure rows illustrate how target quality affects task completion."
        ),
        desired_outcomes=("success", "failure"),
    ),
    DemoStorySpec(
        label="stackcube_closed_loop_limitation",
        source_dir=Path("outputs/h200_60071_query_stackcube_place_bridge_seed0_49/closed_loop_no_clip"),
        caption=(
            "StackCube closed-loop limitation: re-observation is diagnostic, but "
            "does not guarantee higher physical task success."
        ),
        desired_outcomes=("failure", "success"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a demo video/figure planning pack.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "demo_video_pack_latest")
    parser.add_argument(
        "--story",
        action="append",
        default=[],
        help=(
            "Extra story as LABEL=DIR, LABEL=DIR::CAPTION, or LABEL=DIR::CAPTION::key=value. "
            "Rows are selected for both success and failure when available."
        ),
    )
    parser.add_argument("--skip-defaults", action="store_true", help="Only include stories passed with --story.")
    parser.add_argument("--skip-missing", action="store_true", help="Do not fail when a benchmark source is missing.")
    parser.add_argument("--max-media-per-story", type=int, default=12)
    parser.add_argument("--make-slideshows", action="store_true", help="Create one lightweight MP4 slideshow per story when OpenCV is available.")
    parser.add_argument("--slideshow-fps", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = build_demo_story_specs(args.story, include_defaults=not args.skip_defaults)
    try:
        manifest = build_demo_video_pack(
            specs=specs,
            output_dir=args.output_dir,
            skip_missing=args.skip_missing,
            max_media_per_story=args.max_media_per_story,
            make_slideshows=args.make_slideshows,
            slideshow_fps=args.slideshow_fps,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote demo video pack: {args.output_dir}")
    print(f"  README:   {args.output_dir / 'README.md'}")
    print(f"  Manifest: {args.output_dir / 'manifest.json'}")
    print(f"  Capture requests: {len(manifest['capture_requests'])}")
    return 0


def build_demo_story_specs(extra_specs: list[str], include_defaults: bool = True) -> list[DemoStorySpec]:
    """Return default plus user-provided demo story specs."""

    specs = list(DEFAULT_STORIES) if include_defaults else []
    specs.extend(parse_story_spec(value) for value in extra_specs)
    if not specs:
        raise ValueError("No demo stories requested. Use defaults or pass --story.")
    return specs


def parse_story_spec(value: str) -> DemoStorySpec:
    """Parse ``LABEL=DIR``, ``LABEL=DIR::CAPTION``, or ``LABEL=DIR::CAPTION::key=value``."""

    if "=" not in value:
        raise ValueError(f"Invalid story spec {value!r}; expected LABEL=DIR.")
    label, rest = value.split("=", 1)
    label = label.strip()
    parts = [part.strip() for part in rest.split("::")]
    source_text = parts[0] if parts else ""
    if not label or not source_text:
        raise ValueError(f"Invalid story spec {value!r}; expected non-empty LABEL and DIR.")
    caption = parts[1] if len(parts) >= 2 and parts[1] else f"User-provided demo story: {label}."
    row_metadata = parse_metadata(parts[2]) if len(parts) >= 3 and parts[2] else None
    return DemoStorySpec(
        label=label,
        source_dir=Path(source_text),
        caption=caption,
        desired_outcomes=("success", "failure"),
        row_metadata=row_metadata,
    )


def parse_metadata(value: str) -> dict[str, str]:
    """Parse comma-separated key=value metadata for selected demo rows."""

    metadata: dict[str, str] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid story metadata {item!r}; expected key=value.")
        key, item_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid story metadata {item!r}; expected non-empty key.")
        metadata[key] = item_value.strip()
    return metadata


def build_demo_video_pack(
    specs: list[DemoStorySpec],
    output_dir: Path,
    skip_missing: bool = False,
    max_media_per_story: int = 12,
    make_slideshows: bool = False,
    slideshow_fps: float = 1.0,
) -> dict[str, Any]:
    """Summarize demo stories and copy any already-available media."""

    if max_media_per_story < 0:
        raise ValueError("--max-media-per-story must be non-negative.")

    output_dir.mkdir(parents=True, exist_ok=True)
    media_root = output_dir / "media"
    media_root.mkdir(exist_ok=True)

    stories: list[dict[str, Any]] = []
    missing_sources: list[dict[str, Any]] = []
    capture_requests: list[dict[str, Any]] = []

    for spec in specs:
        source_dir = resolve_path(spec.source_dir)
        if not source_dir.exists():
            missing = {
                "label": spec.label,
                "source_dir": str(spec.source_dir),
                "resolved_source_dir": str(source_dir),
                "caption": spec.caption,
            }
            missing_sources.append(missing)
            stories.append({**missing, "exists": False})
            continue

        summary = read_json_if_exists(source_dir / "benchmark_summary.json")
        rows = read_rows_if_exists(source_dir / "benchmark_rows.csv")
        debug_summaries = [] if rows else read_debug_summaries(source_dir)
        selected_rows = select_representative_rows(rows, spec.desired_outcomes)
        if not selected_rows and debug_summaries:
            selected_rows = select_representative_rows(debug_summaries, spec.desired_outcomes)
        if spec.row_metadata:
            selected_rows = [{**row, **spec.row_metadata} for row in selected_rows]
        media_files = collect_media_files(source_dir, selected_rows)
        copied_media = copy_media_files(
            media_files[:max_media_per_story],
            output_dir=output_dir,
            media_root=media_root,
            story_label=spec.label,
        )
        slideshow_path = None
        if make_slideshows and copied_media:
            slideshow_path = write_story_slideshow(
                copied_media=copied_media,
                output_dir=output_dir,
                story_label=spec.label,
                fps=slideshow_fps,
            )

        story = {
            "label": spec.label,
            "exists": True,
            "source_dir": str(spec.source_dir),
            "resolved_source_dir": str(source_dir),
            "caption": spec.caption,
            "summary_path": path_if_exists(source_dir / "benchmark_summary.json"),
            "rows_path": path_if_exists(source_dir / "benchmark_rows.csv"),
            "metrics": extract_summary_metrics(summary) if summary else extract_debug_metrics(debug_summaries),
            "selected_rows": selected_rows,
            "available_media_count": len(media_files),
            "copied_media": copied_media,
            "slideshow_path": slideshow_path,
        }
        stories.append(story)

        if not media_files:
            capture_requests.append(make_capture_request(story))

    if missing_sources and not skip_missing:
        lines = ["Missing required demo benchmark source(s):"]
        lines.extend(f"- {item['label']}: {item['source_dir']}" for item in missing_sources)
        lines.append("Pull the missing artifacts or pass --skip-missing to build a partial pack.")
        raise FileNotFoundError("\n".join(lines))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "total_stories": len(stories),
        "included_stories": sum(1 for story in stories if story.get("exists")),
        "missing_stories": len(missing_sources),
        "capture_request_count": len(capture_requests),
        "stories": stories,
        "missing_sources": missing_sources,
        "capture_requests": capture_requests,
    }
    write_json(manifest, output_dir / "manifest.json")
    write_json(capture_requests, output_dir / "capture_requests.json")
    (output_dir / "README.md").write_text(render_demo_readme(manifest), encoding="utf-8")
    return manifest


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_debug_summaries(source_dir: Path) -> list[dict[str, Any]]:
    """Read single-run summary.json files when the source is not a benchmark dir."""

    summaries: list[dict[str, Any]] = []
    for path in sorted(source_dir.rglob("summary.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        row = compact_debug_summary(summary, path=path)
        summaries.append(row)
    return summaries


def compact_debug_summary(summary: dict[str, Any], path: Path) -> dict[str, Any]:
    """Normalize a single debug summary into the row shape used by the pack."""

    return {
        "query": summary.get("query", ""),
        "seed": summary.get("seed", ""),
        "pick_success": summary.get("pick_success", ""),
        "place_success": summary.get("place_success", ""),
        "task_success": summary.get("task_success", ""),
        "grasp_attempted": summary.get("grasp_attempted", ""),
        "place_attempted": summary.get("place_attempted", ""),
        "pick_stage": summary.get("pick_stage", ""),
        "pick_target_source": summary.get("pick_target_source", summary.get("target_used_for_pick", "")),
        "place_target_source": summary.get("place_target_source", ""),
        "target_used_for_pick": summary.get("target_used_for_pick", ""),
        "artifacts": summary.get("artifacts", str(path.parent)),
    }


def select_representative_rows(rows: list[dict[str, Any]], desired_outcomes: tuple[str, ...]) -> list[dict[str, Any]]:
    """Pick stable, compact row summaries for demo capture."""

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for outcome in desired_outcomes:
        row = first_row_for_outcome(rows, outcome)
        if row is None:
            continue
        key = (str(row.get("query", "")), str(row.get("seed", "")))
        if key in seen:
            continue
        seen.add(key)
        selected.append(compact_row(row, outcome=outcome))
    return selected


def first_row_for_outcome(rows: list[dict[str, Any]], outcome: str) -> dict[str, Any] | None:
    if outcome == "success":
        return next((row for row in rows if row_is_success(row)), None)
    if outcome == "failure":
        return next((row for row in rows if row_is_attempted_failure(row)), None)
    raise ValueError(f"Unsupported desired outcome: {outcome}")


def row_is_success(row: dict[str, Any]) -> bool:
    if "task_success" in row and row["task_success"] != "":
        return as_bool(row["task_success"])
    if "pick_success" in row and row["pick_success"] != "":
        return as_bool(row["pick_success"])
    return False


def row_is_attempted_failure(row: dict[str, Any]) -> bool:
    attempted = as_bool(row.get("place_attempted", row.get("grasp_attempted", "false")))
    if not attempted:
        return False
    return not row_is_success(row)


def compact_row(row: dict[str, Any], outcome: str) -> dict[str, Any]:
    keys = [
        "query",
        "seed",
        "pick_success",
        "place_success",
        "task_success",
        "pick_stage",
        "pick_target_source",
        "place_target_source",
        "target_used_for_pick",
        "artifacts",
    ]
    compact = {key: row.get(key, "") for key in keys if key in row}
    compact["selected_outcome"] = outcome
    return compact


def collect_media_files(source_dir: Path, selected_rows: list[dict[str, Any]]) -> list[Path]:
    """Find already-pulled media near the benchmark or selected run artifacts."""

    roots = [source_dir]
    for row in selected_rows:
        artifact_path = str(row.get("artifacts", "")).strip()
        if artifact_path:
            roots.append(resolve_path(Path(artifact_path)))

    media: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else root.rglob("*")
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in MEDIA_SUFFIXES:
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            media.append(candidate)
    return sorted(media)


def copy_media_files(media_files: list[Path], output_dir: Path, media_root: Path, story_label: str) -> list[str]:
    copied: list[str] = []
    story_dir = media_root / slug(story_label)
    if media_files:
        story_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(media_files, start=1):
        destination = story_dir / f"{index:03d}_{slug(source.stem)}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        copied.append(destination.relative_to(output_dir).as_posix())
    return copied


def write_story_slideshow(
    copied_media: list[str],
    output_dir: Path,
    story_label: str,
    fps: float = 1.0,
) -> str | None:
    """Create a simple MP4 slideshow from copied still images when OpenCV is present."""

    image_paths = [
        output_dir / relative
        for relative in copied_media
        if (output_dir / relative).suffix.lower() in {".jpeg", ".jpg", ".png"}
    ]
    if not image_paths:
        return None
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return None

    first = cv2.imread(str(image_paths[0]))
    if first is None:
        return None
    height, width = first.shape[:2]
    destination = output_dir / "media" / slug(story_label) / f"{slug(story_label)}.mp4"
    fps = fps if fps > 0 else 1.0
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        return None
    try:
        for path in image_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    return destination.relative_to(output_dir).as_posix() if destination.exists() else None


def extract_summary_metrics(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    aggregate = summary.get("aggregate_metrics", {})
    keys = [
        "total_runs",
        "failed_runs",
        "pick_success_rate",
        "place_success_rate",
        "task_success_rate",
        "grasp_attempted_rate",
        "place_attempted_rate",
        "closed_loop_resolution_rate",
        "closed_loop_still_needed_rate",
    ]
    metrics = {key: aggregate[key] for key in keys if key in aggregate}
    for key in ["env_id", "pick_executor", "grasp_target_mode", "place_target_source", "skip_clip"]:
        if key in summary:
            metrics[key] = summary[key]
    return metrics


def extract_debug_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate compact metrics from one or more direct debug summaries."""

    if not rows:
        return {}
    total = len(rows)
    pick_success = sum(1 for row in rows if as_bool(row.get("pick_success", "")))
    place_values = [row for row in rows if row.get("place_success", "") != "" and row.get("place_success") is not None]
    task_values = [row for row in rows if row.get("task_success", "") != "" and row.get("task_success") is not None]
    metrics: dict[str, Any] = {
        "total_runs": total,
        "pick_success_rate": pick_success / total,
    }
    if place_values:
        metrics["place_success_rate"] = sum(1 for row in place_values if as_bool(row.get("place_success", ""))) / len(
            place_values
        )
    if task_values:
        metrics["task_success_rate"] = sum(1 for row in task_values if as_bool(row.get("task_success", ""))) / len(
            task_values
        )
    pick_sources = sorted({str(row.get("pick_target_source", "")) for row in rows if row.get("pick_target_source")})
    place_sources = sorted({str(row.get("place_target_source", "")) for row in rows if row.get("place_target_source")})
    if pick_sources:
        metrics["pick_target_sources"] = pick_sources
    if place_sources:
        metrics["place_target_sources"] = place_sources
    return metrics


def make_capture_request(story: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": story["label"],
        "source_dir": story["source_dir"],
        "reason": "No local image or video artifacts were found for this story.",
        "recommended_rows": story.get("selected_rows", []),
        "caption": story.get("caption", ""),
    }


def render_demo_readme(manifest: dict[str, Any]) -> str:
    lines = [
        "# Query-to-Grasp Demo Video Pack",
        "",
        "This pack organizes representative benchmark rows for conference video and figure production.",
        "It does not introduce new experimental claims; missing media are listed as capture requests.",
        "",
        "## Demo Stories",
        "",
        "| story | source | key metrics | selected rows | media | caption |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for story in manifest["stories"]:
        if not story.get("exists"):
            continue
        metrics = render_metrics(story.get("metrics", {}))
        rows = render_selected_rows(story.get("selected_rows", []))
        media_text = str(story.get("available_media_count", 0))
        if story.get("slideshow_path"):
            media_text = f"{media_text}; `{story['slideshow_path']}`"
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_table_cell(story["label"]),
                    f"`{escape_table_cell(story['source_dir'])}`",
                    escape_table_cell(metrics),
                    escape_table_cell(rows),
                    media_text,
                    escape_table_cell(story.get("caption", "")),
                ]
            )
            + " |"
        )

    if manifest["capture_requests"]:
        lines.extend(["", "## Capture Requests", "", "| story | suggested rows | reason |", "| --- | --- | --- |"])
        for request in manifest["capture_requests"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        escape_table_cell(request["label"]),
                        escape_table_cell(render_selected_rows(request.get("recommended_rows", []))),
                        escape_table_cell(request["reason"]),
                    ]
                )
                + " |"
            )

    if manifest["missing_sources"]:
        lines.extend(["", "## Missing Sources", "", "| story | source |", "| --- | --- |"])
        for source in manifest["missing_sources"]:
            lines.append(f"| {escape_table_cell(source['label'])} | `{escape_table_cell(source['source_dir'])}` |")

    lines.extend(
        [
            "",
            "## Suggested Paper Use",
            "",
            "- Use PickCube success media as the positive executable-target example.",
            "- Use StackCube query-pick plus oracle-place media as the partial task-success bridge.",
            "- Use StackCube tabletop and closed-loop contrast media for the retrieval-to-execution limitation figure.",
            "- Keep oracle placement and privileged target sources clearly labeled in captions.",
            "",
        ]
    )
    return "\n".join(lines)


def render_metrics(metrics: dict[str, Any]) -> str:
    preferred = [
        "env_id",
        "total_runs",
        "failed_runs",
        "pick_success_rate",
        "place_success_rate",
        "task_success_rate",
        "pick_target_sources",
        "pick_executor",
        "place_target_source",
        "place_target_sources",
    ]
    parts = [f"{key}={metrics[key]}" for key in preferred if key in metrics]
    return "; ".join(parts) if parts else "n/a"


def render_selected_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "none"
    parts = []
    for row in rows:
        seed = row.get("seed", "?")
        outcome = row.get("selected_outcome", "?")
        query = row.get("query", "")
        task = row.get("task_success", row.get("pick_success", ""))
        parts.append(f"{outcome}: seed {seed} {query} task={task}".strip())
    return "; ".join(parts)


def path_if_exists(path: Path) -> str | None:
    return str(path) if path.exists() else None


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def slug(value: str) -> str:
    text = "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
    return text or "artifact"


def escape_table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
