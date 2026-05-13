# Query-to-Grasp RAS Journal Draft

This directory contains the Robotics and Autonomous Systems journal rewrite.
It is separate from the conference-style draft in `paper/`.

## Files

- `main.tex`: RAS-oriented manuscript scaffold.
- `highlights.md`: Elsevier highlights, each under 85 characters.
- `references.bib`: local bibliography copy for standalone compilation.
- `figures/`: local figure files used by `main.tex`.
- `tables/`: local LaTeX table fragments and CSV provenance files.

## Data Sources

The current RAS tables and figures are generated from frozen local summaries:

```powershell
$env:PYTHONPATH=(Get-Location).Path
python scripts/generate_paper_revision_results_summary.py --output-dir outputs/paper_revision_results_summary_latest
python scripts/generate_external_crop_200seed_summary.py --output-dir outputs/paper_revision_results_summary_latest --figure-dir paper/figures
python scripts/generate_ras_tables_figures.py
python scripts/generate_ras_ycb_expansion_results.py
python scripts/build_ras_pipeline_figure.py
python scripts/generate_ras_target_error_from_artifacts.py --root <complete-run-root> --output-dir outputs/ras_revision_aggressive_20260511/error_correlation
python scripts/generate_ras_sensor_stress_results.py --input-root outputs/ras_sensor_stress_v2_20260512_lightweight
python scripts/generate_ras_pipeline_figure.py
python scripts/check_ras_manuscript.py --tex paper_ras/main.tex --bib paper_ras/references.bib --highlights paper_ras/highlights.md
```

Generated RAS artifacts:

- `tables/table_external_crop_with_ci.csv`
- `tables/table_target_ladder_with_ci.csv`
- `tables/table_noisy_oracle_with_ci.csv`
- `tables/table_noncube_gate_with_ci.csv`
- `tables/table_ycb_expansion_with_ci.csv`
- `tables/table_ycb_chunk_consistency.csv`
- `tables/table_ycb_failure_taxonomy.csv`
- `tables/table_error_bins_with_ci.csv`
- `tables/table_place_error_bins_with_ci.csv`
- `tables/table_sensor_stress_with_ci.csv`
- `tables/table_sensor_stress_compact.tex`
- `tables/table_failure_taxonomy.tex`
- `figures/ras_crop_baseline_ci.pdf`
- `figures/ras_target_ladder.pdf`
- `figures/ras_noisy_oracle_sensitivity.pdf`
- `figures/ras_ycb_noncube_ladder.pdf`
- `figures/ras_ycb_chunk_consistency.pdf`
- `figures/figure_target_error_mechanism.pdf`
- `figures/figure_place_error_success.pdf`
- `figures/figure_error_bins_success.pdf`
- `figures/figure_sensor_stress_pick_success.pdf`
- `figures/figure_sensor_stress_task_success.pdf`
- `figures/figure_sensor_stress_depth_support.pdf`
- `figures/pipeline_overview_vector.pdf`
- `figures/pipeline_overview_vector.svg`
- `figures/execution_evidence_montage.pdf` (supplementary/illustrative only; not used as the main architecture figure)

The target-error-to-success analysis is generated from complete per-run
`pick_result.json` and `summary.json` artifacts on the H200 output tree. The
lightweight CSV rows alone are insufficient because they do not contain both the
executed pick target and the matched oracle target.

## Standalone Overleaf Upload

Upload the full `paper_ras/` directory contents:

- `main.tex`
- `references.bib`
- `highlights.md`
- all files under `figures/`
- all files under `tables/`

The manuscript no longer depends on `../paper/` or `../outputs/` paths.

## Claim Boundary

This journal version must remain a simulated diagnostic systems paper. It does
not claim real-robot execution, a learned controller, a learned grasp generator,
general YCB/EGAD manipulation, or robust relation-heavy StackCube stacking.

## RAS Submission Components

The current scaffold includes:

- abstract under 250 words
- 1 to 7 keywords
- highlights
- data availability statement
- declaration of generative AI use
- funding statement
- CRediT statement

Final submission will require checking the latest Elsevier/RAS author guide and
compiling with an Elsevier-compatible LaTeX setup.
