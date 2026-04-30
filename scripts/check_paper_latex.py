"""Structural checks for the Query-to-Grasp LaTeX paper scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


REQUIRED_SECTIONS = (
    "Introduction",
    "Related Work",
    "Method",
    "Experimental Setup",
    "Results",
    "Limitations",
    "Conclusion",
)

REQUIRED_METRICS = ("1.0000", "0.6200", "0.5200", "0.9400", "0.0000")

REQUIRED_TERMS = (
    "PickCube-v1",
    "StackCube-v1",
    "task\\_success",
    "outputs/paper\\_figure\\_pack\\_latest",
    "memory\\_grasp\\_world\\_xyz",
    "task\\_guard\\_selected\\_object\\_world\\_xyz",
    "oracle\\_object\\_pose",
)

REQUIRED_BIB_KEYS = (
    "radford2021learning",
    "liu2023groundingdino",
    "shridhar2022cliport",
    "shridhar2022peract",
    "ahn2022saycan",
    "jiang2023vima",
    "brohan2022rt1",
    "brohan2023rt2",
    "gu2023maniskill2",
)

FORBIDDEN_PATTERNS = (
    (
        re.compile(
            r"\b(real[- ]robot|physical robot)[^.\n]{0,80}"
            r"(success|successful|achieved|demonstrated|deployment|deployed)",
            re.IGNORECASE,
        ),
        "Unsupported real-robot success/deployment claim.",
    ),
    (
        re.compile(
            r"\bwe\s+(achieve|demonstrate|show|validate)[^.\n]{0,80}"
            r"(real[- ]robot|physical robot)",
            re.IGNORECASE,
        ),
        "Unsupported real-robot execution claim.",
    ),
    (
        re.compile(
            r"\blearned controller[^.\n]{0,80}(achieves|solves|is trained|succeeds)",
            re.IGNORECASE,
        ),
        "Unsupported learned-controller claim.",
    ),
    (
        re.compile(
            r"StackCube[^.\n]{0,120}(stacking|stack placement|task completion)"
            r"[^.\n]{0,80}(achieved|successful|solved|completed)",
            re.IGNORECASE,
        ),
        "Unsupported StackCube completion claim.",
    ),
    (
        re.compile(
            r"StackCube[^.\n]{0,120}(task[_ ]success|task success rate)"
            r"[^.\n]{0,80}(1\.0|100%|achieved|successful|nonzero)",
            re.IGNORECASE,
        ),
        "Unsupported StackCube task-success claim.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the Query-to-Grasp LaTeX paper scaffold.")
    parser.add_argument("--tex", type=Path, default=Path("paper/main.tex"))
    parser.add_argument("--bib", type=Path, default=Path("paper/references.bib"))
    return parser.parse_args()


def check_paper_latex(tex_path: Path, bib_path: Path) -> list[str]:
    """Return a list of structural paper-check errors."""

    errors: list[str] = []
    if not tex_path.exists():
        return [f"Missing TeX file: {tex_path}"]
    if not bib_path.exists():
        return [f"Missing BibTeX file: {bib_path}"]

    tex = tex_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8")

    if "\\bibliography{references}" not in tex and "\\addbibresource{references.bib}" not in tex:
        errors.append("main.tex must reference references.bib.")

    for section in REQUIRED_SECTIONS:
        pattern = re.compile(rf"\\section\*?\{{{re.escape(section)}\}}")
        if not pattern.search(tex):
            errors.append(f"Missing required section: {section}")

    for metric in REQUIRED_METRICS:
        if metric not in tex:
            errors.append(f"Missing required metric: {metric}")

    for term in REQUIRED_TERMS:
        if term not in tex:
            errors.append(f"Missing required paper term/artifact reference: {term}")

    for key in REQUIRED_BIB_KEYS:
        if f"{{{key}," not in bib:
            errors.append(f"Missing BibTeX entry: {key}")
        if key not in tex:
            errors.append(f"BibTeX key is not cited in main.tex: {key}")

    for pattern, message in FORBIDDEN_PATTERNS:
        match = pattern.search(tex)
        if match:
            errors.append(f"{message} Matched: {match.group(0)!r}")

    return errors


def main() -> int:
    args = parse_args()
    errors = check_paper_latex(args.tex, args.bib)
    if errors:
        print("Paper LaTeX check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Paper LaTeX check passed: {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
