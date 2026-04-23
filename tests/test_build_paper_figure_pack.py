from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_paper_figure_pack import (
    ArtifactSpec,
    DEFAULT_ARTIFACTS,
    build_artifact_specs,
    build_paper_figure_pack,
    parse_artifact_spec,
)


def test_parse_artifact_spec_accepts_minimal_and_full_forms() -> None:
    minimal = parse_artifact_spec("table=outputs/table.md")
    full = parse_artifact_spec("trace=outputs/trace.md::traces::Selection explanation.")

    assert minimal.label == "table"
    assert minimal.source_path == Path("outputs/table.md")
    assert minimal.category == "extra"
    assert "table" in minimal.caption
    assert full.label == "trace"
    assert full.category == "traces"
    assert full.caption == "Selection explanation."


def test_build_artifact_specs_can_skip_defaults() -> None:
    specs = build_artifact_specs(["trace=outputs/trace.md::traces::caption"], include_defaults=False)

    assert len(specs) == 1
    assert specs[0].label == "trace"


def test_default_artifacts_include_architecture_note() -> None:
    labels = {spec.label for spec in DEFAULT_ARTIFACTS}

    assert "implemented_architecture" in labels


def test_build_paper_figure_pack_copies_artifacts_and_writes_readme(tmp_path: Path) -> None:
    source = tmp_path / "source table.md"
    source.write_text("# Table\n\n| a | b |\n", encoding="utf-8")
    output_dir = tmp_path / "pack"

    manifest = build_paper_figure_pack(
        specs=[
            ArtifactSpec(
                label="Main Table",
                source_path=source,
                category="Tables",
                caption="Caption with | separator.",
            )
        ],
        output_dir=output_dir,
    )

    copied = output_dir / "tables" / "main_table.md"
    readme = output_dir / "README.md"
    manifest_path = output_dir / "manifest.json"

    assert copied.exists()
    assert copied.read_text(encoding="utf-8").startswith("# Table")
    assert readme.exists()
    assert "Caption with \\| separator." in readme.read_text(encoding="utf-8")
    assert manifest_path.exists()
    assert manifest["included_count"] == 1
    assert manifest["missing_count"] == 0
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["entries"][0]["copied_path"] == "tables/main_table.md"


def test_build_paper_figure_pack_missing_behavior(tmp_path: Path) -> None:
    missing = tmp_path / "missing.md"
    output_dir = tmp_path / "pack"
    spec = ArtifactSpec(label="Missing", source_path=missing, category="tables", caption="Missing artifact.")

    with pytest.raises(FileNotFoundError, match="Missing required artifact"):
        build_paper_figure_pack(specs=[spec], output_dir=output_dir)

    manifest = build_paper_figure_pack(specs=[spec], output_dir=output_dir, skip_missing=True)

    assert manifest["included_count"] == 0
    assert manifest["missing_count"] == 1
    assert "Missing Artifacts" in (output_dir / "README.md").read_text(encoding="utf-8")
