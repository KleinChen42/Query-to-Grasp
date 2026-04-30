# Query-to-Grasp Paper Draft

This directory contains the first ICRA/IROS-style LaTeX scaffold for the
Query-to-Grasp conference paper.

## Files

- `main.tex`: compact conference-style manuscript scaffold.
- `references.bib`: first-pass BibTeX scaffold for the related-work buckets.
- `README.md`: notes for maintaining and checking the paper source.

## Structural Check

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path
python scripts/check_paper_latex.py --tex paper/main.tex --bib paper/references.bib
```

The checker verifies that required sections, core metrics, artifact references,
and bibliography links are present. It also rejects obvious unsupported claims
such as real-robot success or StackCube task completion.

## Optional Compile

If a TeX toolchain is installed:

```bash
latexmk -pdf paper/main.tex
```

or:

```bash
pdflatex paper/main.tex
bibtex main
pdflatex paper/main.tex
pdflatex paper/main.tex
```

Compilation is optional for this scaffold checkpoint; the required acceptance
test is the structural checker plus paper-pack inclusion.

## Current Claim Boundary

- PickCube-v1 is the main successful simulated pick benchmark.
- StackCube-v1 is a pick-only compatibility diagnostic.
- StackCube `task_success_rate` remains `0.0000`.
- No real-robot execution, learned controller, or StackCube stack completion is
  claimed.

## Paper Pack

Refresh copied artifacts with:

```powershell
$env:PYTHONPATH=(Get-Location).Path
python scripts/build_paper_figure_pack.py --output-dir outputs/paper_figure_pack_latest --skip-missing
```
