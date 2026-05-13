# Artifact Manifest — Query-to-Grasp

This document lists the reproducibility artifacts included in the
Query-to-Grasp repository for the RAS submission.

---

## 1. Paper

| File | Description |
|------|-------------|
| `paper_ras/main.tex` | Main LaTeX source |
| `paper_ras/references.bib` | Bibliography |
| `paper_ras/elsarticle.cls` | Elsevier article class |
| `paper_ras/elsarticle-num.bst` | Elsevier bibliography style |

---

## 2. Tables

| File | Description |
|------|-------------|
| `paper_ras/tables/table_external_crop_200_with_ci.tex` | External RGB-D crop baseline (200 seeds) |
| `paper_ras/tables/table_external_crop_with_ci.tex` | External RGB-D crop baseline |
| `paper_ras/tables/table_noisy_oracle_with_ci.tex` | Noisy-oracle sensitivity |
| `paper_ras/tables/table_noncube_gate_with_ci.tex` | Non-cube PickSingleYCB diagnostic |
| `paper_ras/tables/table_target_ladder_with_ci.tex` | Target-source ladder |
| `paper_ras/tables/table_sensor_stress_with_ci.tex` | Full sensor-stress table (appendix) |
| `paper_ras/tables/table_sensor_stress_compact.tex` | Compact sensor-stress table |
| `paper_ras/tables/table_failure_taxonomy.tex` | Failure taxonomy |
| `paper_ras/tables/table_error_bins_with_ci.tex` | Target-error bins |
| `paper_ras/tables/table_place_error_bins_with_ci.tex` | Place-error bins |
| `paper_ras/tables/table_success_failure_error_stats.tex` | Success/failure error statistics |
| `paper_ras/tables/table_ycb_chunk_consistency.tex` | YCB chunk consistency |
| `paper_ras/tables/table_ycb_expansion_with_ci.tex` | YCB expansion |
| `paper_ras/tables/table_ycb_failure_taxonomy.tex` | YCB failure taxonomy |

---

## 3. Figures

| File | Description |
|------|-------------|
| `paper_ras/figures/pipeline_overview_vector.pdf` | Pipeline overview (vector) |
| `paper_ras/figures/execution_evidence_montage.pdf` | Execution evidence montage |
| `paper_ras/figures/ras_crop_baseline_ci.pdf` | Crop baseline with CI |
| `paper_ras/figures/ras_noisy_oracle_sensitivity.pdf` | Noisy-oracle sensitivity |
| `paper_ras/figures/ras_target_ladder.pdf` | Target-source ladder |
| `paper_ras/figures/ras_ycb_noncube_ladder.pdf` | YCB non-cube ladder |
| `paper_ras/figures/ras_ycb_chunk_consistency.pdf` | YCB chunk consistency |
| `paper_ras/figures/ras_ycb_expansion_ladder.pdf` | YCB expansion ladder |
| `paper_ras/figures/geometry_memory_ablation.pdf` | Camera-frame alignment ablation |
| `paper_ras/figures/figure_error_bins_success.pdf` | Target-error bins |
| `paper_ras/figures/figure_error_by_target_source.pdf` | Error by target source |
| `paper_ras/figures/figure_error_distribution_success_failure.pdf` | Error distribution |
| `paper_ras/figures/figure_place_error_success.pdf` | Place-error diagnostic |
| `paper_ras/figures/figure_target_error_mechanism.pdf` | Target-error mechanism |
| `paper_ras/figures/figure_sensor_stress_pick_success.pdf` | Sensor stress pick success |
| `paper_ras/figures/figure_sensor_stress_task_success.pdf` | Sensor stress task success |
| `paper_ras/figures/figure_sensor_stress_depth_support.pdf` | Sensor stress depth support |

---

## 4. Summaries

| File | Description |
|------|-------------|
| `paper_ras/tables/*.csv` | CSV summaries used to generate paper tables |
| `paper_ras/tables/ras_tables_manifest.json` | Table generation manifest |

---

## 5. Scripts (Reproducibility)

| File | Description |
|------|-------------|
| `scripts/generate_ras_tables_figures.py` | Generate all RAS paper tables and figures |
| `scripts/generate_paper_figures.py` | Generate paper figures from summaries |
| `scripts/generate_ras_sunday_results.py` | Generate Sunday results summary |
| `scripts/generate_ras_sensor_stress_results.py` | Generate sensor-stress results |
| `scripts/generate_ras_target_error_from_artifacts.py` | Target-error analysis |
| `scripts/generate_ras_place_error_from_artifacts.py` | Place-error analysis |
| `scripts/generate_ras_ycb_expansion_results.py` | YCB expansion results |
| `scripts/generate_external_crop_200seed_summary.py` | External crop 200-seed summary |
| `scripts/generate_benchmark_report.py` | Benchmark report generation |
| `scripts/generate_paper_ablation_table.py` | Paper ablation table |
| `scripts/generate_per_query_diagnostics_table.py` | Per-query diagnostics |
| `scripts/build_paper_figure_pack.py` | Build paper figure pack |
| `scripts/build_execution_evidence_figure.py` | Build execution evidence |

---

## 6. Exclusions

The following are **not** included in this public repository:

- **Raw massive outputs**: Per-run observations, depth maps, detection overlays
  (tens of thousands of files per benchmark configuration).
- **Model weights**: GroundingDINO and CLIP weights (obtain via HuggingFace).
- **Restricted external datasets/assets**: ManiSkill and YCB assets (obtain
  from official sources).
- **Private SSH/cluster launch scripts**: H200 connection, synchronization,
  and remote execution scripts are excluded for security and portability.
- **Temporary archives**: `.zip` and `.tgz` patch/paper archives.
- **Private development notes**: Chinese scratch notes, Codex task specs.
