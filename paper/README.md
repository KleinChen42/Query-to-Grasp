# Query-to-Grasp Paper Draft

This directory contains the first IROS/ICRA-style LaTeX scaffold for the
Query-to-Grasp H200-scale simulated manipulation paper.

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
- StackCube-v1 is the main cross-task bridge benchmark.
- Accepted query-driven StackCube rows remain pick-only until the placement
  bridge is validated.
- Oracle StackCube pick-place is a privileged upper-bound baseline, not a
  deployable language-conditioned stacker.
- No real-robot execution, learned controller, or StackCube stack completion is
  claimed.

## Paper Pack

Refresh copied artifacts with:

```powershell
$env:PYTHONPATH=(Get-Location).Path
python scripts/build_paper_figure_pack.py --output-dir outputs/paper_figure_pack_latest --skip-missing
```
