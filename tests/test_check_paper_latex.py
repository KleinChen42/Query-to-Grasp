from __future__ import annotations

from pathlib import Path

from scripts.check_paper_latex import check_paper_latex


def test_current_paper_latex_passes() -> None:
    assert check_paper_latex(Path("paper/main.tex"), Path("paper/references.bib")) == []


def test_checker_rejects_real_robot_success_claim(tmp_path: Path) -> None:
    tex_path = tmp_path / "main.tex"
    bib_path = tmp_path / "references.bib"
    tex = Path("paper/main.tex").read_text(encoding="utf-8")
    bib = Path("paper/references.bib").read_text(encoding="utf-8")
    tex_path.write_text(tex + "\nWe demonstrate real robot success.\n", encoding="utf-8")
    bib_path.write_text(bib, encoding="utf-8")

    errors = check_paper_latex(tex_path, bib_path)

    assert any("real-robot" in error for error in errors)


def test_checker_rejects_stackcube_completion_claim(tmp_path: Path) -> None:
    tex_path = tmp_path / "main.tex"
    bib_path = tmp_path / "references.bib"
    tex = Path("paper/main.tex").read_text(encoding="utf-8")
    bib = Path("paper/references.bib").read_text(encoding="utf-8")
    tex_path.write_text(tex + "\nStackCube stacking completion is achieved.\n", encoding="utf-8")
    bib_path.write_text(bib, encoding="utf-8")

    errors = check_paper_latex(tex_path, bib_path)

    assert any("StackCube completion" in error for error in errors)
