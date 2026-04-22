# Query-to-Grasp Paper Milestone Log

Last updated: 2026-04-22

Purpose: keep a concise, paper-oriented record of key implementation milestones,
experiment reports, quantitative findings, and next decisions. This file is
tracked by Git; generated artifacts remain under `outputs/` and should not be
committed.

## Current Thesis

Working title:

`Query-to-Grasp via Confidence-Aware 3D Semantic Fusion`

Current evidence supports the following narrower near-term claim:

> A language-queryable single-view perception-to-3D-target pipeline can be made
> runnable and benchmarkable with HF GroundingDINO, optional CLIP reranking, RGB-D
> lifting, and placeholder pick execution. The next research bottleneck is not
> CLIP reranking quality, but candidate multiplicity and cross-view 3D geometric
> consistency.

## Artifact Map

| artifact | purpose | location |
| --- | --- | --- |
| Paper ablation table | HF no-CLIP, HF with-CLIP, ambiguity no-CLIP, ambiguity with-CLIP | `outputs/h200_60071_paper_baseline/outputs/paper_ablation_table.md` |
| Per-query diagnostics table | Query-level detector/rerank/runtime diagnostics | `outputs/h200_60071_paper_baseline/outputs/per_query_diagnostics_table.md` |
| Full tabletop_3 fusion comparison | Single-view vs virtual 3-view fusion comparison | `outputs/h200_60071_tabletop3_full/fusion_comparison_table_tabletop3_full.md` |
| Full tabletop_3 fusion summary | Machine-readable multi-view fusion benchmark summary | `outputs/h200_60071_tabletop3_full/benchmark_summary.json` |
| Multi-view memory diagnostics | Merge-distance sweep and cross-view 3D consistency diagnosis | `outputs/h200_60071_tabletop3_full/memory_diagnostics.md` |
| Remote camera probe | ManiSkill camera availability for `PickCube-v1` | H200: `outputs/camera_view_probe_pickcube/camera_view_report.json` |

## Milestone Timeline

| milestone | status | evidence | interpretation |
| --- | --- | --- | --- |
| Stable single-view mock pipeline | Done | mock detector + skip-CLIP smoke and benchmark flow | Safe fallback path exercises the full pipeline without large model downloads. |
| HF GroundingDINO path | Done | HF no-CLIP benchmark, 6 runs | Real detector path is runnable on H200. |
| Runtime and per-query reporting | Done | benchmark summaries include runtime and per-query metrics | Outputs are now suitable for paper tables and debugging. |
| CLIP usefulness diagnostics | Done | candidate multiplicity metrics and top-1-change metrics | CLIP has no measured effect because reranking rarely has room to alter top-1. |
| Ambiguity benchmark | Done | ambiguity benchmarks, 33 runs each no-CLIP/with-CLIP | Broader queries increase raw candidates, but CLIP still does not change top-1. |
| Paper ablation table | Done | `paper_ablation_table.md/csv` | Baseline results can be copied into a draft experiment table. |
| Camera view probe | Done | `check_maniskill_camera_views.py` and H200 probe | `PickCube-v1` exposes only `base_camera` by default. |
| Virtual tabletop_3 multi-view capture | Done | `--view-preset tabletop_3` smoke and benchmark | True distinct RGB-D views can be captured by moving `base_camera`. |
| Multi-view memory diagnostics | Done | `memory_diagnostics.md/json` | Multi-view capture works, but same-label 3D estimates are far apart across views. |

## Key Quantitative Results

### HF Single-View and Ambiguity Ablation

Source: `outputs/h200_60071_paper_baseline/outputs/paper_ablation_table.md`

| setting | runs | mean_raw_num_detections | mean_num_ranked_candidates | fraction_top1_changed_by_rerank | fraction_with_3d_target | pick_success_rate | mean_runtime_seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HF no CLIP | 6 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 52.6174 |
| HF with CLIP | 6 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 39.3762 |
| Ambiguity no CLIP | 33 | 1.4242 | 1.4242 | 0.0000 | 0.9394 | 0.0000 | 51.6914 |
| Ambiguity with CLIP | 33 | 1.4242 | 1.4242 | 0.0000 | 0.9394 | 0.0000 | 65.3235 |

Paper note:

> In the current HF setting, CLIP reranking produces no measurable top-1 change.
> Even ambiguity-focused queries increase candidate multiplicity only modestly
> and still produce `fraction_top1_changed_by_rerank = 0.0`. Therefore, CLIP
> should be treated as an optional diagnostic module rather than the primary
> source of improvement until detector candidate multiplicity improves.

Placeholder pick note:

`pick_success_rate = 0.0` is expected because `SafePlaceholderPickExecutor`
validates targets and returns structured outputs without executing real robot
control.

### Single-View vs tabletop_3 Fusion

Source: `outputs/h200_60071_tabletop3_full/fusion_comparison_table_tabletop3_full.md`

| setting | runs | primary_rate | mean_runtime_seconds | mean_num_views | mean_num_memory_objects | mean_num_observations_added | mean_selected_overall_confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HF single no CLIP | 6 | 1.0000 | 52.6174 | 1.0000 | n/a | n/a | n/a |
| HF tabletop_3 fusion no CLIP | 6 | 1.0000 | 87.1678 | 3.0000 | 3.3333 | 3.6667 | 0.5017 |

Interpretation:

The virtual multi-view preset successfully captures three RGB-D viewpoints and
produces selected memory objects in all runs. However, it does not yet provide a
clean fused object hypothesis; the object memory fragments into roughly three
objects per run.

### Multi-View Memory Diagnostics

Source: `outputs/h200_60071_tabletop3_full/memory_diagnostics.md`

| metric | value |
| --- | ---: |
| runs | 6 |
| mean_num_candidate_observations | 3.6667 |
| mean_num_memory_objects | 3.3333 |
| mean_same_label_pairwise_distance | 1.0693 |

Merge-distance sweep:

| merge_distance | mean_simulated_memory_objects |
| ---: | ---: |
| 0.05 | 3.6667 |
| 0.08 | 3.3333 |
| 0.12 | 3.1667 |
| 0.16 | 3.0000 |
| 0.24 | 3.0000 |
| 0.32 | 3.0000 |

Paper note:

> Multi-view capture is working, but same-label 3D target estimates are far apart
> across views. Since relaxing the merge threshold to `0.32 m` still leaves about
> three objects per run, the next bottleneck is likely cross-view 3D lifting,
> camera-pose alignment, or detector consistency, not merely a conservative
> memory merge threshold.

## Commands Worth Preserving

HF single-view no-CLIP:

```bash
PYTHONPATH=$PWD python scripts/run_single_view_pick_benchmark.py \
  --queries "red cube" "blue mug" \
  --seeds 0 1 2 \
  --detector-backend hf \
  --skip-clip \
  --depth-scale 1000 \
  --output-dir outputs/benchmark_hf_no_clip
```

HF ambiguity benchmark:

```bash
PYTHONPATH=$PWD python scripts/run_ambiguity_benchmark.py \
  --queries-file configs/ambiguity_queries.txt \
  --seeds 0 1 2 \
  --detector-backend hf \
  --skip-clip \
  --depth-scale 1000 \
  --output-dir outputs/ambiguity_hf_no_clip \
  --generate-report
```

tabletop_3 multi-view fusion benchmark:

```bash
PYTHONPATH=$PWD python scripts/run_multiview_fusion_benchmark.py \
  --queries "red cube" "blue mug" \
  --seeds 0 1 2 \
  --detector-backend hf \
  --skip-clip \
  --depth-scale 1000 \
  --view-preset tabletop_3 \
  --output-dir outputs/multiview_fusion_tabletop3_hf_no_clip
```

Memory diagnostics:

```bash
PYTHONPATH=$PWD python scripts/generate_multiview_memory_diagnostics.py \
  --benchmark-dir outputs/multiview_fusion_tabletop3_hf_no_clip
```

Fusion comparison table:

```bash
PYTHONPATH=$PWD python scripts/generate_fusion_comparison_table.py \
  --single-view "HF single no CLIP=outputs/benchmark_hf_no_clip" \
  --fusion "HF tabletop_3 fusion no CLIP=outputs/multiview_fusion_tabletop3_hf_no_clip" \
  --output-md outputs/fusion_comparison_table_tabletop3_full.md \
  --output-csv outputs/fusion_comparison_table_tabletop3_full.csv
```

## Current Bottlenecks

1. Detector candidate multiplicity is still low in most exact-object settings.
2. CLIP reranking has no current top-1 effect because candidate sets are too small.
3. Virtual multi-view capture works, but same-label 3D estimates are not
   geometrically consistent across views.
4. Memory merge threshold tuning alone is unlikely to solve fragmentation.
5. Real robot control is still intentionally absent; placeholder pick success is
   not an end-to-end grasp metric.

## Next Recommended Milestone

Build a cross-view geometry sanity report for one `tabletop_3` run:

1. For each view, record selected candidate box, camera extrinsic, camera_xyz,
   world_xyz, depth_valid_ratio, and num_points.
2. Compare same-label `world_xyz` across views.
3. Verify whether the issue is depth scaling, extrinsic convention, detector box
   inconsistency, or using median crop depth over background-heavy boxes.
4. Only after that decide whether to adjust lifting, camera pose handling,
   segmentation support, or memory merge logic.

Candidate paper framing:

> The current prototype establishes a runnable language-query-to-3D-target
> baseline and exposes the practical systems bottleneck for confidence-aware
> 3D fusion: reliable cross-view geometric consistency, not merely adding more
> views or reranking candidates.
