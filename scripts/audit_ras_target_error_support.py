"""Audit whether current RAS artifacts support target-error-to-success plots.

This script intentionally does not infer target error from incomplete fields.
It checks the available per-run CSV schemas and reports whether each artifact
contains both a predicted target coordinate and a privileged/oracle reference
coordinate for the same action target.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


PREDICTED_TARGET_FIELDS = {
    "target_world_xyz",
    "pick_target_xyz",
    "target_xyz",
    "selected_target_xyz",
    "grasp_world_xyz",
}
ORACLE_TARGET_FIELDS = {
    "oracle_object_pose_xyz",
    "oracle_pick_xyz",
    "oracle_target_xyz",
    "oracle_pose_xyz",
}
SUCCESS_FIELDS = {"pick_success", "place_success", "task_success", "raw_env_success"}


@dataclass
class CsvAudit:
    path: str
    rows: int
    columns: list[str]
    predicted_target_fields: list[str]
    oracle_target_fields: list[str]
    success_fields: list[str]
    supports_pick_error: bool
    note: str


def audit_csv(path: Path, root: Path) -> CsvAudit:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        rows = sum(1 for _ in reader)

    column_set = set(columns)
    predicted = sorted(column_set & PREDICTED_TARGET_FIELDS)
    oracle = sorted(column_set & ORACLE_TARGET_FIELDS)
    success = sorted(column_set & SUCCESS_FIELDS)
    supports = bool(predicted and oracle and success)

    if supports:
        note = "contains predicted target, oracle target, and success fields"
    else:
        missing = []
        if not predicted:
            missing.append("predicted pick target xyz")
        if not oracle:
            missing.append("oracle/reference pick target xyz")
        if not success:
            missing.append("success metric")
        note = "missing " + ", ".join(missing)

    return CsvAudit(
        path=str(path.relative_to(root)),
        rows=rows,
        columns=columns,
        predicted_target_fields=predicted,
        oracle_target_fields=oracle,
        success_fields=success,
        supports_pick_error=supports,
        note=note,
    )


def find_rows(root: Path) -> list[Path]:
    candidates = []
    for pattern in ("**/benchmark_rows.csv", "**/runs.jsonl", "**/*rows*.csv"):
        candidates.extend(root.glob(pattern))
    unique = sorted({p.resolve() for p in candidates if p.is_file()})
    return [Path(p) for p in unique]


def write_markdown(output_path: Path, audits: list[CsvAudit], supports_any: bool) -> None:
    lines = [
        "# Target-Error Correlation Support Audit",
        "",
        "This audit checks whether existing lightweight RAS artifacts contain the",
        "fields needed to compute target-error-to-success curves without inventing",
        "missing coordinates.",
        "",
        "## Decision",
        "",
    ]
    if supports_any:
        lines.append("At least one CSV appears to contain enough fields for target-error analysis.")
    else:
        lines.extend(
            [
                "Current lightweight rows do **not** support a paper-facing target-error",
                "plot. The available CSVs include success metrics, but they do not include",
                "both predicted pick target coordinates and oracle/reference pick target",
                "coordinates in the same per-run row.",
                "",
                "Therefore no `table_error_bins_with_ci.tex` or",
                "`figure_error_bins_success.pdf` is generated in this pass.",
            ]
        )
    lines.extend(
        [
            "",
            "## Required Fields For A Future Supported Plot",
            "",
            "- `target_world_xyz` or `pick_target_xyz`",
            "- `oracle_object_pose_xyz` or `oracle_pick_xyz`",
            "- `target_error_m` computed from those two coordinates",
            "- `pick_success` and, for pick-place rows, `place_success` / `task_success`",
            "- `target_source`, `env_id`, `seed`, and `failure_type`",
            "",
            "## CSV Schema Audit",
            "",
            "| CSV | Rows | Predicted target fields | Oracle target fields | Success fields | Decision |",
            "|---|---:|---|---|---|---|",
        ]
    )
    for audit in audits:
        lines.append(
            "| {path} | {rows} | {pred} | {oracle} | {success} | {note} |".format(
                path=audit.path.replace("\\", "/"),
                rows=audit.rows,
                pred=", ".join(audit.predicted_target_fields) or "-",
                oracle=", ".join(audit.oracle_target_fields) or "-",
                success=", ".join(audit.success_fields) or "-",
                note=audit.note,
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="outputs/ras_revision_aggressive_20260511",
        help="Artifact root to scan.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ras_revision_aggressive_20260511/error_correlation",
        help="Directory for audit outputs.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = [p for p in find_rows(root) if p.suffix.lower() == ".csv"]
    audits = [audit_csv(path, root) for path in csv_paths]
    supports_any = any(a.supports_pick_error for a in audits)

    summary = {
        "artifact_root": str(root),
        "csv_count": len(audits),
        "supports_target_error_correlation": supports_any,
        "generated_table": None,
        "generated_figure": None,
        "reason": (
            "At least one CSV has predicted/oracle target fields."
            if supports_any
            else "Current lightweight rows lack both predicted pick target xyz and oracle/reference pick target xyz."
        ),
        "audits": [asdict(a) for a in audits],
    }
    (output_dir / "summary_error_correlation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_markdown(output_dir / "summary_error_correlation.md", audits, supports_any)

    print(json.dumps({"output_dir": str(output_dir), "supports": supports_any}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
