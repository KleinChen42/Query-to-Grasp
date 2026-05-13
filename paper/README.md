# Query-to-Grasp Paper Draft

This directory contains the IROS/ICRA-style LaTeX scaffold for the
Query-to-Grasp simulated manipulation paper.

## Files

- `main.tex`: compact conference-style manuscript scaffold.
- `IEEEtran.cls`: IEEE conference class file copied from the provided template.
- `references.bib`: first-pass BibTeX scaffold for the related-work buckets.
- `README.md`: notes for maintaining and checking the paper source.
- `figures/execution_evidence_montage.pdf`: representative continuous
  ManiSkill execution evidence figure for the main paper.
- `figures/pipeline_overview.pdf`, `figures/geometry_memory_ablation.pdf`,
  `figures/target_source_results.pdf`, and
  `figures/stackcube_failure_taxonomy.pdf`: supporting paper figures.
- `figures/external_crop_200seed_results.pdf`: 200-seed external RGB-D
  crop baseline figure generated from frozen H200 summaries.

## Structural Check

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path
python scripts/check_paper_latex.py --tex paper/main.tex --bib paper/references.bib
```

The checker verifies that required sections, author metadata, core metrics,
supplemental-video references, and bibliography links are present. It also
rejects obvious unsupported claims such as real-robot success or StackCube task
completion.

## Optional Compile

If a TeX toolchain is installed:

```bash
latexmk -pdf -cd paper/main.tex
```

or:

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Compilation is optional for this scaffold checkpoint; the required acceptance
test is the structural checker plus paper-pack inclusion.

## Overleaf Upload Set

Upload these tracked files for the current conference draft:

- `main.tex`
- `references.bib`
- `IEEEtran.cls`
- `figures/execution_evidence_montage.pdf`
- `figures/pipeline_overview.pdf`
- `figures/geometry_memory_ablation.pdf`
- `figures/target_source_results.pdf`
- `figures/stackcube_failure_taxonomy.pdf`
- `figures/external_crop_200seed_results.pdf`

## Current Claim Boundary

Current author block:

- Zhuo Chen, Chalmers University of Technology, `zhuoc@chalmers.se`.
- No funding footnote is included.

- PickCube-v1 is the main successful simulated pick benchmark.
- StackCube-v1 is the main cross-task bridge benchmark.
- The accepted query-pick/oracle-place StackCube bridge reports partial
  task-success evidence with query-derived cubeA targets and privileged
  oracle cubeB placement targets.
- Oracle StackCube pick-place is a privileged upper-bound baseline, not a
  deployable language-conditioned stacker.
- No real-robot execution, learned controller, or fully non-oracle StackCube
  stack completion is claimed.

## Paper Pack

Refresh copied artifacts with:

```powershell
$env:PYTHONPATH=(Get-Location).Path
python scripts/generate_external_crop_200seed_summary.py --output-dir outputs/paper_revision_results_summary_latest --figure-dir paper/figures
python scripts/build_paper_figure_pack.py --output-dir outputs/paper_figure_pack_latest --skip-missing
```

## Submission Package Audit

Before freezing a submission snapshot, regenerate the video and audit artifacts
in this order:

```powershell
$env:PYTHONPATH=(Get-Location).Path
python scripts/run_demo_execution_capture_pack.py `
  --output-dir outputs/h200_60071_demo_execution_capture_native720_latest `
  --demo-pack-output-dir outputs/demo_video_pack_latest `
  --sensor-width 720 --sensor-height 720 `
  --width 1920 --height 1080
python scripts/build_supplemental_video.py `
  --input outputs/demo_video_pack_latest/manifest.json `
  --output-dir outputs/supplemental_video_latest `
  --width 1920 --height 1080
python scripts/audit_paper_submission_package.py --output-dir outputs/paper_submission_audit_latest
python scripts/build_paper_figure_pack.py --output-dir outputs/paper_figure_pack_latest --skip-missing
python scripts/check_paper_latex.py --tex paper/main.tex --bib paper/references.bib
```

The frozen supplemental video uses native ManiSkill RGB frames requested with
`sensor_configs = {"width": 720, "height": 720}` and writes a `1920x1080`
conference MP4. Generated video outputs remain untracked.

The audit writes `audit_report.md/json` and the frozen main results table under
`outputs/paper_submission_audit_latest`. These are generated submission
artifacts and should not be committed directly.
