"""Structural checker for the Query-to-Grasp RAS journal manuscript."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, default=Path("paper_ras") / "main.tex")
    parser.add_argument("--bib", type=Path, default=Path("paper") / "references.bib")
    parser.add_argument("--highlights", type=Path, default=Path("paper_ras") / "highlights.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    tex = read_text(args.tex, errors)
    bib = read_text(args.bib, errors)
    highlights = read_text(args.highlights, errors)
    if errors:
        return report(errors)

    check_front_matter(tex, highlights, errors)
    check_sections(tex, errors)
    check_claim_boundaries(tex, errors)
    check_table_inputs(tex, errors)
    check_citations(tex, bib, errors)
    return report(errors)


def read_text(path: Path, errors: list[str]) -> str:
    if not path.exists():
        errors.append(f"Missing file: {path}")
        return ""
    return path.read_text(encoding="utf-8")


def check_front_matter(tex: str, highlights: str, errors: list[str]) -> None:
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not abstract_match:
        errors.append("Missing abstract environment.")
    else:
        words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", abstract_match.group(1))
        if len(words) > 250:
            errors.append(f"Abstract has {len(words)} words; RAS target is <=250.")

    keyword_match = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", tex, re.S)
    if not keyword_match:
        errors.append("Missing keyword environment.")
    else:
        keywords = [item.strip() for item in keyword_match.group(1).split("\\sep") if item.strip()]
        if not 1 <= len(keywords) <= 7:
            errors.append(f"Expected 1-7 keywords, found {len(keywords)}.")

    highlight_lines = [line[2:].strip() for line in highlights.splitlines() if line.startswith("- ")]
    if not 3 <= len(highlight_lines) <= 5:
        errors.append(f"Expected 3-5 highlights, found {len(highlight_lines)}.")
    for line in highlight_lines:
        if len(line) > 85:
            errors.append(f"Highlight exceeds 85 characters: {line}")


def check_sections(tex: str, errors: list[str]) -> None:
    required = [
        "Introduction",
        "Related Work",
        "Problem Formulation",
        "Method",
        "Experimental Design",
        "Results",
        "Discussion",
        "Limitations and Future Work",
        "Conclusion",
    ]
    for section in required:
        if f"\\section{{{section}}}" not in tex:
            errors.append(f"Missing required RAS section: {section}")


def check_claim_boundaries(tex: str, errors: list[str]) -> None:
    required_phrases = [
        "controlled simulated conditions",
        "physical robot deployment",
        "privileged diagnostic",
        "runtime compatibility failures",
        "executor-mismatch diagnostic",
        "not counted as negative manipulation results",
    ]
    for phrase in required_phrases:
        if phrase not in tex:
            errors.append(f"Missing claim-boundary phrase: {phrase}")

    banned_patterns = [
        r"we\s+deploy\s+on\s+a\s+real\s+robot",
        r"learned\s+controller\s+achieves",
        r"robust\s+YCB\s+manipulation",
        r"solves\s+StackCube",
    ]
    for pattern in banned_patterns:
        if re.search(pattern, tex, re.I):
            errors.append(f"Potential overclaim matched pattern: {pattern}")


def check_table_inputs(tex: str, errors: list[str]) -> None:
    inputs = re.findall(r"\\input\{([^}]+)\}", tex)
    texlive_inputs = {"glyphtounicode"}
    for raw_path in inputs:
        if raw_path in texlive_inputs:
            continue
        path = (Path("paper_ras") / raw_path).resolve()
        if not path.exists():
            errors.append(f"Missing LaTeX input: {raw_path}")

    graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    for raw_path in graphics:
        path = (Path("paper_ras") / raw_path).resolve()
        if not path.exists():
            errors.append(f"Missing figure input: {raw_path}")

    required_inputs = [
        "table_external_crop_with_ci.tex",
        "table_target_ladder_with_ci.tex",
        "table_noisy_oracle_with_ci.tex",
        "table_noncube_gate_with_ci.tex",
        "table_sensor_stress_compact.tex",
        "table_sensor_stress_with_ci.tex",
        "table_failure_taxonomy.tex",
    ]
    for filename in required_inputs:
        if filename not in tex:
            errors.append(f"Missing RAS table input: {filename}")


def check_citations(tex: str, bib: str, errors: list[str]) -> None:
    cites: set[str] = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", tex):
        cites.update(item.strip() for item in match.group(1).split(",") if item.strip())
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing = sorted(cites - keys)
    if missing:
        errors.append("Missing BibTeX keys: " + ", ".join(missing))


def report(errors: list[str]) -> int:
    if errors:
        print("RAS manuscript check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("RAS manuscript check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
