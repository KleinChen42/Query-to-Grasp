# Query-to-Grasp Paper Milestone Log

Last updated: 2026-04-23

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
> lifting, and placeholder pick execution. Corrected virtual multi-view capture
> can reduce object-memory fragmentation when camera-frame conventions are
> handled explicitly. CLIP reranking remains optional until detector candidate
> multiplicity improves.

## Artifact Map

| artifact | purpose | location |
| --- | --- | --- |
| Paper ablation table | HF no-CLIP, HF with-CLIP, ambiguity no-CLIP, ambiguity with-CLIP | `outputs/h200_60071_paper_baseline/outputs/paper_ablation_table.md` |
| Per-query diagnostics table | Query-level detector/rerank/runtime diagnostics | `outputs/h200_60071_paper_baseline/outputs/per_query_diagnostics_table.md` |
| Full tabletop_3 fusion comparison | Single-view vs virtual 3-view fusion comparison | `outputs/h200_60071_tabletop3_full/fusion_comparison_table_tabletop3_full.md` |
| Full tabletop_3 fusion summary | Machine-readable multi-view fusion benchmark summary | `outputs/h200_60071_tabletop3_full/benchmark_summary.json` |
| Multi-view memory diagnostics | Merge-distance sweep and cross-view 3D consistency diagnosis | `outputs/h200_60071_tabletop3_full/memory_diagnostics.md` |
| Cross-view geometry sanity report | Per-view boxes, camera/world coordinates, extrinsic source, transform sanity | `outputs/h200_60071_tabletop3_full/cross_view_geometry_report.md` |
| Extrinsic convention comparison | Direct `cam2world_gl`, converted `cam2world_gl`, and `extrinsic_cv` comparison | `outputs/h200_60071_extrinsic_convention/extrinsic_convention_report.md` |
| CV-fixed tabletop_3 fusion comparison | Single-view vs virtual 3-view fusion after camera convention fix | `outputs/h200_60071_tabletop3_cvfix/fusion_comparison_table_tabletop3_cvfix.md` |
| CV-fixed memory diagnostics | Post-fix merge-distance sweep and cross-view 3D consistency diagnosis | `outputs/h200_60071_tabletop3_cvfix/memory_diagnostics.md` |
| CV-fixed cross-view geometry sanity report | Post-fix transform sanity and cross-view geometry report | `outputs/h200_60071_tabletop3_cvfix/cross_view_geometry_report_tabletop3_cvfix.md` |
| CV-fixed CLIP ablation table | Single-view/fusion and no-CLIP/with-CLIP comparison after camera convention fix | `outputs/h200_60071_tabletop3_cvfix_clip_ablation/fusion_comparison_table_tabletop3_cvfix_clip_ablation.md` |
| CV-fixed with-CLIP fusion diagnostics | Post-fix tabletop_3 fusion benchmark with CLIP enabled | `outputs/h200_60071_tabletop3_with_clip_cvfix/memory_diagnostics.md` |
| Selection trace example | Paper-friendly explanation of corrected multi-view target selection | `outputs/h200_60071_selection_trace_red_cube_seed0/selection_trace.md` |
| Re-observation decision example | Rule-based confidence-aware re-observation decision smoke | `outputs/h200_60071_reobserve_smoke/reobserve_decision.json` |
| Implemented architecture note | Method diagram and artifact map for the current implemented pipeline | `docs/architecture_query_to_grasp.md` |
| Ambiguity tabletop_3 fusion stress table | Corrected fusion no-CLIP/with-CLIP comparison on ambiguity queries, seeds 0-2 | `outputs/h200_60071_ambiguity_tabletop3_seed012/outputs/fusion_comparison_table_ambiguity_tabletop3_hf_seed012.md` |
| Ambiguity tabletop_3 re-observation report | Per-query open-loop policy behavior on ambiguity queries, seeds 0-2 | `outputs/h200_60071_ambiguity_tabletop3_seed012/outputs/reobserve_policy_report_ambiguity_tabletop3_hf_seed012.md` |
| Closed-loop re-observation smoke | Opt-in extra-view before/after diagnostics for one mock ambiguity run | H200: `outputs/h200_smoke_closed_loop_reobserve_mock/closed_loop_reobserve.json` |
| Closed-loop ambiguity HF comparison | HF no-CLIP/with-CLIP ambiguity benchmark with one suggested extra virtual view | `outputs/h200_60071_closed_loop_ambiguity_seed012/outputs/fusion_comparison_table_ambiguity_tabletop3_hf_closed_loop.md` |
| Closed-loop ambiguity policy report | Initial vs final policy trigger rates and per-query reason counts | `outputs/h200_60071_closed_loop_ambiguity_seed012/outputs/reobserve_policy_report_ambiguity_tabletop3_hf_closed_loop.md` |
| Paper figure pack | Captioned collection of current paper/demo artifacts | `outputs/paper_figure_pack_latest/README.md` |
| Paper draft outline | Claim, method, experiment, limitation, and next-code scaffold | `docs/paper_draft_outline.md` |
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
| Cross-view geometry sanity report | Done | `cross_view_geometry_report.md/json` | World coordinates are internally consistent with the selected extrinsic, but the selected source is `cam2world_gl`. |
| Extrinsic convention comparison | Done | `extrinsic_convention_report.md/json` | `cam2world_gl @ OpenCV-to-OpenGL` reduces same-label cross-view distance from `1.0693 m` to `0.0518 m`. |
| RGB-D lifting camera convention fix | Done | `mask_projector.py` + H200 cvfix benchmark | Default lifting now converts OpenCV-projected points before applying ManiSkill `cam2world_gl`. |
| CV-fixed tabletop_3 fusion | Done | `h200_60071_tabletop3_cvfix` reports | Multi-view memory fragmentation drops from `3.3333` to `1.3333` objects/run. |
| CV-fixed CLIP fusion ablation | Done | `fusion_comparison_table_tabletop3_cvfix_clip_ablation.md/csv` | CLIP increases selected confidence but not selection rate or cross-view geometry in this benchmark. |
| Multi-view selection trace | Done | `selection_trace.json/md` per debug run | Target selection is now explainable through label pool, confidence terms, views, votes, and deterministic tie-breaks. |
| Paper figure pack | Done | `build_paper_figure_pack.py` and `outputs/paper_figure_pack_latest` | Key tables, geometry reports, diagnostics, traces, and milestone notes can be gathered with one command. |
| Formal target selector module | Done | `src/policy/target_selector.py` | Selection and trace rendering are now reusable policy code, not debug-script-only helpers. |
| Paper draft outline | Done | `docs/paper_draft_outline.md` | Current claims, experiments, limitations, and next coding milestone are explicitly scoped. |
| Rule-based re-observation policy | Done | `src/policy/reobserve_policy.py` and `reobserve_decision.json` smoke | Multi-view runs now emit confidence-aware re-observation decisions without automatically moving cameras. |
| Architecture and README refresh | Done | `docs/architecture_query_to_grasp.md` and `README.md` | The repo now has a clean external quickstart and a paper-ready method diagram source matching implemented behavior. |
| Ambiguity tabletop_3 fusion stress, seeds 0-2 | Done | `fusion_comparison_table_ambiguity_tabletop3_hf_seed012.md` and `reobserve_policy_report_ambiguity_tabletop3_hf_seed012.md` | Broader queries increase object-memory fragmentation and trigger policy uncertainty; CLIP raises selected confidence and lowers trigger rate without fixing geometry. |
| Minimal closed-loop re-observation path | Done | `closed_loop_reobserve.json` H200 mock smoke | The debug and benchmark runners can now opt into one suggested extra virtual view and report before/after policy, confidence, memory, and selected-target metrics. |
| Closed-loop ambiguity HF benchmark, seeds 0-2 | Done | `fusion_comparison_table_ambiguity_tabletop3_hf_closed_loop.md` and `reobserve_policy_report_ambiguity_tabletop3_hf_closed_loop.md` | The extra-view loop executes, but final policy trigger rate does not decrease; current suggested views do not resolve the dominant uncertainty. |
| Closed-loop delta diagnostics | Done | H200 mock smoke with `closed_loop_resolution_rate`, `closed_loop_still_needed_rate`, and selected-support deltas | Future closed-loop runs now expose whether an extra view changed selected object, confidence, selected view support, memory size, policy reason, or resolved re-observation. |
| Support-aware reobserve suggestion policy | Done | H200 mock smokes for ambiguity-driven and geometry-driven reasons | Re-observation suggestions now depend on the failure mode: missing support views are preferred for ambiguity/support issues, while `top_down`-style views are preferred for geometry issues. |

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

### Cross-View Geometry Sanity

Source: `outputs/h200_60071_tabletop3_full/cross_view_geometry_report.md`

| metric | value |
| --- | ---: |
| runs | 6 |
| mean_top_rank_pairwise_distance | 1.1301 |
| mean_same_label_pairwise_distance | 1.0693 |
| mean_world_recompute_error | 0.0284 |
| extrinsic source | `sensor_param.base_camera.cam2world_gl` |

Interpretation:

The stored `world_xyz` values are largely consistent with applying the selected
extrinsic to `camera_xyz`; the mean recomputation difference is only `0.0284 m`.
This is small relative to the `1.0693 m` same-label cross-view spread and can be
explained by estimating medians in camera/world point sets separately.

Paper note:

> The geometric inconsistency is not primarily caused by a serialization or
> memory bookkeeping error. The next targeted check should compare ManiSkill's
> `cam2world_gl` against `extrinsic_cv`/OpenCV conventions, because RGB-D lifting
> currently projects points in an OpenCV-style pinhole frame while the selected
> extrinsic source is OpenGL-labeled.

### Extrinsic Convention Comparison

Source: `outputs/h200_60071_extrinsic_convention/extrinsic_convention_report.md`

| convention | mean_same_label_pairwise_distance | mean_top_rank_pairwise_distance | mean_world_z |
| --- | ---: | ---: | ---: |
| `cam2world_gl_direct` | 1.0693 | 1.1301 | 0.9303 |
| `cam2world_gl_cv_to_gl` | 0.0518 | 0.0304 | 0.1697 |

Interpretation:

The RGB-D projection path produces OpenCV-style camera points (`x` right, `y`
down, `z` forward), while ManiSkill's selected `cam2world_gl` matrix expects
OpenGL-style camera points (`x` right, `y` up, `z` backward). Applying the fixed
OpenCV-to-OpenGL camera-frame conversion before `cam2world_gl` makes same-label
cross-view estimates geometrically consistent.

Paper note:

> A practical multi-view fusion bottleneck was not the memory merge threshold,
> but the camera-frame convention between RGB-D lifting and simulator-provided
> camera poses. Correcting this convention reduced same-label cross-view spread
> by roughly `20.6x` (`1.0693 m` to `0.0518 m`).

### tabletop_3 Fusion After Camera Convention Fix

Source: `outputs/h200_60071_tabletop3_cvfix/fusion_comparison_table_tabletop3_cvfix.md`

| setting | runs | primary_rate | mean_runtime_seconds | mean_num_views | mean_num_memory_objects | mean_num_observations_added | mean_selected_overall_confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HF single no CLIP | 6 | 1.0000 | 52.6174 | 1.0000 | n/a | n/a | n/a |
| HF tabletop_3 fusion no CLIP cvfix | 6 | 1.0000 | 96.9404 | 3.0000 | 1.3333 | 3.6667 | 0.5282 |

Post-fix memory diagnostics:

| metric | before fix | after fix |
| --- | ---: | ---: |
| mean_num_memory_objects | 3.3333 | 1.3333 |
| mean_same_label_pairwise_distance | 1.0693 | 0.0518 |
| mean_top_rank_pairwise_distance | 1.1301 | 0.0304 |

Merge-distance sweep after the fix:

| merge_distance | mean_simulated_memory_objects |
| ---: | ---: |
| 0.05 | 1.8333 |
| 0.08 | 1.3333 |
| 0.12 | 1.1667 |
| 0.16 | 1.0000 |
| 0.24 | 1.0000 |
| 0.32 | 1.0000 |

Paper note:

> After camera-convention correction, virtual three-view capture produces a
> compact object memory in the HF/no-CLIP setting. This is the first evidence
> that the multi-view path can support the project thesis, although the current
> benchmark still uses placeholder pick execution and a small object/query set.

### Corrected Multi-View CLIP Ablation

Source:
`outputs/h200_60071_tabletop3_cvfix_clip_ablation/fusion_comparison_table_tabletop3_cvfix_clip_ablation.md`

| setting | runs | primary_rate | runtime_s | views | memory_objects | observations | same_label_dist | selected_confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HF single no CLIP | 6 | 1.0000 | 52.6174 | 1.0000 | n/a | n/a | n/a | n/a |
| HF single with CLIP | 6 | 1.0000 | 39.3762 | 1.0000 | n/a | n/a | n/a | n/a |
| HF tabletop_3 fusion no CLIP cvfix | 6 | 1.0000 | 96.9404 | 3.0000 | 1.3333 | 3.6667 | 0.0518 | 0.5282 |
| HF tabletop_3 fusion with CLIP cvfix | 6 | 1.0000 | 233.6068 | 3.0000 | 1.3333 | 3.6667 | 0.0518 | 0.7091 |

Interpretation:

Corrected multi-view fusion gives compact object memory with or without CLIP.
In this small HF tabletop benchmark, CLIP improves the selected memory object's
confidence (`0.5282` to `0.7091`) but does not change the selected-object rate,
memory fragmentation, or cross-view same-label distance. Runtime increases by
about `2.41x` relative to no-CLIP fusion.

Paper note:

> The strongest current evidence for the project hypothesis comes from corrected
> multi-view geometry, not CLIP reranking. CLIP can be kept as a confidence term,
> but should not be claimed as the main source of retrieval improvement until
> candidate multiplicity and top-1 changes are observed.

### Ambiguity tabletop_3 Fusion Stress, Seeds 0-2

Source:
`outputs/h200_60071_ambiguity_tabletop3_seed012/outputs/fusion_comparison_table_ambiguity_tabletop3_hf_seed012.md`

Queries:

- `configs/ambiguity_queries.txt`
- Seeds: `0 1 2`
- View preset: corrected `tabletop_3`
- Detector: HF GroundingDINO

| setting | runs | selected_frac | runtime_s | views | memory_objects | observations | same_label_dist | selected_confidence | reobserve_trigger_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ambiguity tabletop_3 HF no CLIP seeds0-2 | 33 | 1.0000 | 86.0037 | 3.0000 | 2.1515 | 3.7576 | 0.1353 | 0.4985 | 0.6667 |
| Ambiguity tabletop_3 HF with CLIP seeds0-2 | 33 | 1.0000 | 89.6845 | 3.0000 | 2.1515 | 3.7576 | 0.1353 | 0.6806 | 0.4242 |

Policy reason counts:

| setting | reason_counts |
| --- | --- |
| no CLIP | `confident_enough: 11; insufficient_view_support: 11; low_overall_confidence: 8; ambiguous_top_candidates: 3` |
| with CLIP | `confident_enough: 19; insufficient_view_support: 11; ambiguous_top_candidates: 3` |

Interpretation:

Broader ambiguity queries produce a more difficult fusion setting than the
exact-object benchmark: mean memory objects increase from `1.3333` in the
corrected exact-query tabletop_3 benchmark to `2.1818`, and same-label spread
increases to `0.1353 m`. CLIP does not reduce memory fragmentation or geometric
spread, but it raises selected-object confidence from `0.4985` to `0.6806` and
reduces the policy trigger rate from `0.6667` to `0.4242`.

Paper note:

> In ambiguity-focused multi-view fusion, CLIP behaves more like a confidence
> stabilizer than a geometric correction mechanism. The dominant triggered
> reason is `insufficient_view_support`, suggesting that the next closed-loop
> step should test whether an additional viewpoint can increase support for
> fragmented targets.

### Selection Trace Example

Source: `outputs/h200_60071_selection_trace_red_cube_seed0/selection_trace.md`

Scenario:

- Query: `red cube`
- Seed: `0`
- View preset: `tabletop_3`
- Detector: HF GroundingDINO
- CLIP: skipped

Selection result:

| rank | selected | object_id | top_label | overall | semantic | geometry | views | observations | world_xyz |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | yes | `obj_0000` | `red cube` | 0.5081 | 1.0000 | 1.0000 | 3 | 4 | `[-0.001, 0.069, 0.003]` |
| 2 | no | `obj_0001` | `red cube` | 0.4342 | 1.0000 | 1.0000 | 1 | 1 | `[-0.007, -0.039, 0.003]` |

Interpretation:

The selector first used the query-derived label pool (`red cube`, `cube`,
`block`), found two memory objects containing `red cube`, and selected
`obj_0000` because it had higher fused confidence and stronger view support
(`3` views, `4` observations). This trace is suitable for a paper figure or demo
debug panel because it exposes both the semantic filter and deterministic
tie-break order.

### Minimal Closed-Loop Re-Observation Smoke

Source: H200 `outputs/h200_smoke_closed_loop_reobserve_mock/closed_loop_reobserve.json`

Scenario:

- Query: `object`
- Seed: `0`
- Detector: mock
- View preset: `tabletop_3`
- CLIP: skipped
- Closed-loop extra-view budget: `1`

| metric | before | after |
| --- | ---: | ---: |
| should_reobserve | 1 | 1 |
| selected_overall_confidence | 0.6015 | 0.6015 |
| num_memory_objects | 2 | 2 |
| num_view_results | 3 | 4 |

Extra view:

| view_id | detections | ranked | 3d candidates | observations added |
| --- | ---: | ---: | ---: | ---: |
| `left` | 1 | 1 | 1 | 1 |

Interpretation:

The minimal closed-loop infrastructure works: when the policy requests another
view, the debug and benchmark runners can capture one suggested virtual view,
rerun perception, update memory, and write before/after diagnostics. This mock
smoke intentionally does not prove policy benefit; the reason remains
`ambiguous_top_candidates`, so the next paper-relevant step is the HF ambiguity
closed-loop benchmark.

### Closed-Loop Ambiguity HF Benchmark, Seeds 0-2

Source:
`outputs/h200_60071_closed_loop_ambiguity_seed012/outputs/fusion_comparison_table_ambiguity_tabletop3_hf_closed_loop.md`
and
`outputs/h200_60071_closed_loop_ambiguity_seed012/outputs/reobserve_policy_report_ambiguity_tabletop3_hf_closed_loop.md`

Queries:

- `configs/ambiguity_queries.txt`
- Seeds: `0 1 2`
- View preset: corrected `tabletop_3`
- Detector: HF GroundingDINO
- Closed-loop budget: one suggested extra virtual view

| setting | runs | views | memory objects | observations | same-label dist | selected confidence | initial trigger | final trigger | execution rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ambiguity HF no CLIP closed-loop | 33 | 3.6667 | 2.1515 | 4.6061 | 0.1276 | 0.4985 | 0.6667 | 0.6667 | 0.6667 |
| Ambiguity HF with CLIP closed-loop | 33 | 3.4242 | 2.1515 | 4.5152 | 0.1284 | 0.6806 | 0.4242 | 0.4242 | 0.4242 |

Policy reason counts after closed loop:

| setting | final reason counts |
| --- | --- |
| no CLIP | `confident_enough: 11; insufficient_view_support: 11; low_overall_confidence: 7; ambiguous_top_candidates: 4` |
| with CLIP | `confident_enough: 19; insufficient_view_support: 11; ambiguous_top_candidates: 3` |

Extra-view execution:

| setting | executed runs | suggested views |
| --- | ---: | --- |
| no CLIP | 22 | `left: 8; front: 7; top_down: 4; right: 3` |
| with CLIP | 14 | `front: 7; left: 6; right: 1` |

Interpretation:

The closed-loop infrastructure is now validated on the full ambiguity benchmark:
all benchmark runs completed, extra views were captured when requested, and
before/after policy metrics were written. However, the one-extra-view policy did
not reduce trigger rate: no-CLIP remains `0.6667` before and after, while
with-CLIP remains `0.4242` before and after. CLIP again raises selected-object
confidence, but does not improve memory fragmentation, geometric spread, or
closed-loop uncertainty resolution.

Paper note:

> A single rule-suggested extra virtual view is not sufficient to resolve the
> dominant ambiguity in the current HF tabletop setting. This is a useful
> negative result: the project now has a runnable closed-loop perception
> baseline, and the next algorithmic improvement should target support-aware
> view selection or memory update criteria rather than merely enabling another
> observation pass.

### Closed-Loop Delta Diagnostics

Current implementation:

- `closed_loop_reobserve.json` now includes deltas for selected confidence,
  selected view support, selected observation count, memory-object count, policy
  reason changes, selected-object changes, and whether re-observation was
  resolved.
- `benchmark_rows.json/csv` expose compact per-run delta fields.
- `benchmark_summary.json` aggregates resolution rate, still-needed rate,
  selected-object-change rate, reason-change rate, and mean selected-support
  deltas.
- `generate_reobserve_policy_report.py` and
  `generate_fusion_comparison_table.py` surface the new aggregate fields.

H200 mock smoke:

| metric | value |
| --- | ---: |
| closed_loop_execution_rate | 1.0000 |
| closed_loop_resolution_rate | 0.0000 |
| closed_loop_still_needed_rate | 1.0000 |
| mean_closed_loop_delta_selected_num_views | 0.0000 |
| mean_closed_loop_delta_selected_overall_confidence | 0.0000 |

Interpretation:

The diagnostic layer is now in place. The mock smoke still remains unresolved,
which is expected, but future HF closed-loop reruns can directly show whether an
extra view increased selected support or changed policy outcomes. This is the
measurement layer needed before changing the re-observation algorithm itself.

### Support-Aware Reobserve Suggestion Policy

Current implementation:

- `insufficient_view_support` and `ambiguous_top_candidates` now prioritize
  missing views that do not yet support the selected object.
- `low_geometry_confidence` and `too_few_3d_points` now prioritize geometry
  views such as `top_down` and `closer_oblique`.
- `reobserve_decision.json` now records `suggested_view_plan` with a
  `priority_reason` for each suggested view.

H200 mock validation:

| scenario | triggered reason | suggested views | priority reasons |
| --- | --- | --- | --- |
| default ambiguity smoke | `ambiguous_top_candidates` | `left` | `increase_selected_view_support` |
| forced geometry smoke | `low_geometry_confidence` | `top_down`, `closer_oblique` | `improve_geometry_evidence` |

Interpretation:

The policy has moved from static fallback suggestions to reason-aware view
selection. This is the first direct functionality improvement after the
closed-loop negative result: the system can now distinguish between "look from
an unused support view" and "look from a geometry-friendly view." The next step
is to rerun a compact closed-loop benchmark and see whether selected-support
deltas or resolution rate improve.

### Support-Variant Compact Ambiguity Benchmark

Source:
H200 `outputs/h200_60071_support_variant_ambiguity_compact_seed0`

Setup:

- Queries: `cube`, `block`, `container`, `object`
- Seed: `0`
- Conditions: HF no-CLIP and HF with-CLIP
- Closed-loop policy: support-aware with `closer_front/left/right` variants

| setting | initial trigger | final trigger | resolution rate | selected support delta | selected object change rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no CLIP | 1.0000 | 1.0000 | 0.0000 | 0.2500 | 0.2500 |
| with CLIP | 0.7500 | 0.7500 | 0.0000 | 0.0000 | 0.2500 |

Interpretation:

The `closer_*` support-view variants are a real functional step forward:
compared with the previous compact closed-loop rerun, the no-CLIP condition now
shows positive selected-support movement (`mean_closed_loop_delta_selected_num_views = 0.25`).
However, this still does not resolve uncertainty in the compact benchmark, and
with-CLIP remains support-flat. This suggests the next bottleneck is not only
which extra pose to capture, but how the new observation is associated back into
the selected object memory.

### Selected-Object Association Compact Ambiguity Benchmark

Source:
H200 `outputs/h200_60071_assoc_diag_ambiguity_compact_seed0_v2`

Setup:

- Queries: `cube`, `block`, `container`, `object`
- Seed: `0`
- Conditions: HF no-CLIP and HF with-CLIP
- Closed-loop policy: support-aware with `closer_front/left/right` variants
- New diagnostics: initial-selected association and support-gain rates

| setting | initial trigger | final trigger | resolution rate | selected assoc rate | selected support gain rate | mean before-selected delta views | mean before-selected delta obs | selected object change rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no CLIP | 1.0000 | 1.0000 | 0.0000 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| with CLIP | 0.7500 | 0.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2500 |

Interpretation:

The new diagnostic makes the next bottleneck explicit. In the compact no-CLIP
condition, one quarter of runs now merge the extra observation back into the
initially selected object and increase that object's view support. In the
with-CLIP condition, the same support-aware policy still executes, but the
association rate remains `0.0` while selected-object change rate stays `0.25`.
This strongly suggests the next algorithmic target is not "add another view,"
but "preserve selected-object continuity when the extra observation is folded
back into memory."

### Extra-View Absorber Trace Compact Ambiguity Benchmark

Source:
H200 `outputs/h200_60071_absorber_trace_ambiguity_compact_seed0`

Setup:

- Queries: `cube`, `block`, `container`, `object`
- Seed: `0`
- Conditions: HF no-CLIP and HF with-CLIP
- Closed-loop policy: support-aware with `closer_front/left/right` variants
- New diagnostics: which object actually absorbed the extra-view observation

| setting | initial trigger | final trigger | resolution rate | initial selected absorber rate | final selected absorber rate | third-object rate | mean absorber count | selected object change rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no CLIP | 1.0000 | 1.0000 | 0.0000 | 0.2500 | 0.5000 | 0.5000 | 1.2500 | 0.2500 |
| with CLIP | 0.7500 | 0.7500 | 0.0000 | 0.0000 | 0.2500 | 0.5000 | 1.0000 | 0.2500 |

Interpretation:

This trace resolves the ambiguity around the previous association metric. In
the compact no-CLIP condition, extra views sometimes land in the final selected
object even when they do not preserve the initial selection. In the with-CLIP
condition, only one quarter of runs land in the final selected object, while
half of runs involve a third object entirely. That points the next coding
milestone toward selected-object continuity and CLIP-aware memory association,
not toward adding more views or more policy branches.

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

tabletop_3 multi-view fusion with CLIP:

```bash
PYTHONPATH=$PWD python scripts/run_multiview_fusion_benchmark.py \
  --queries "red cube" "blue mug" \
  --seeds 0 1 2 \
  --detector-backend hf \
  --use-clip \
  --depth-scale 1000 \
  --view-preset tabletop_3 \
  --camera-name base_camera \
  --output-dir outputs/multiview_fusion_tabletop3_hf_with_clip_cvfix
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

Extrinsic convention comparison:

```bash
PYTHONPATH=$PWD python scripts/compare_extrinsic_conventions.py \
  --queries "red cube" "blue mug" \
  --seeds 0 1 2 \
  --detector-backend hf \
  --skip-clip \
  --depth-scale 1000 \
  --view-preset tabletop_3 \
  --camera-name base_camera \
  --output-dir outputs/extrinsic_convention_tabletop3_hf_no_clip
```

Post-fix cross-view geometry sanity report:

```bash
PYTHONPATH=$PWD python scripts/generate_cross_view_geometry_report.py \
  --benchmark-dir outputs/multiview_fusion_tabletop3_hf_no_clip_cvfix \
  --output-json outputs/cross_view_geometry_report_tabletop3_cvfix.json \
  --output-md outputs/cross_view_geometry_report_tabletop3_cvfix.md
```

Corrected fusion CLIP ablation table:

```bash
PYTHONPATH=$PWD python scripts/generate_fusion_comparison_table.py \
  --single-view "HF single no CLIP=outputs/benchmark_hf_no_clip" \
  --single-view "HF single with CLIP=outputs/benchmark_hf_with_clip" \
  --fusion "HF tabletop_3 fusion no CLIP cvfix=outputs/multiview_fusion_tabletop3_hf_no_clip_cvfix" \
  --fusion "HF tabletop_3 fusion with CLIP cvfix=outputs/multiview_fusion_tabletop3_hf_with_clip_cvfix" \
  --output-md outputs/fusion_comparison_table_tabletop3_cvfix_clip_ablation.md \
  --output-csv outputs/fusion_comparison_table_tabletop3_cvfix_clip_ablation.csv
```

Selection trace smoke:

```bash
PYTHONPATH=$PWD python scripts/run_multiview_fusion_debug.py \
  --query "red cube" \
  --seed 0 \
  --detector-backend hf \
  --skip-clip \
  --depth-scale 1000 \
  --view-preset tabletop_3 \
  --camera-name base_camera \
  --output-dir outputs/selection_trace_tabletop3_red_cube_seed0_cvfix
```

Re-observation decision smoke:

```bash
PYTHONPATH=$PWD python scripts/run_multiview_fusion_debug.py \
  --query "red cube" \
  --seed 0 \
  --detector-backend mock \
  --mock-box-position center \
  --skip-clip \
  --depth-scale 1000 \
  --view-preset tabletop_3 \
  --camera-name base_camera \
  --output-dir outputs/reobserve_smoke_mock_tabletop3
```

Re-observation policy report:

```bash
PYTHONPATH=$PWD python scripts/generate_reobserve_policy_report.py \
  --benchmark HF_no_CLIP_reobserve=outputs/multiview_fusion_tabletop3_hf_no_clip_reobserve_v2 \
  --benchmark HF_with_CLIP_reobserve=outputs/multiview_fusion_tabletop3_hf_with_clip_reobserve_v2 \
  --output-md outputs/reobserve_policy_report_tabletop3_hf_reobserve_v2.md \
  --output-json outputs/reobserve_policy_report_tabletop3_hf_reobserve_v2.json
```

Closed-loop re-observation smoke:

```bash
PYTHONPATH=$PWD python scripts/run_multiview_fusion_benchmark.py \
  --queries object \
  --seeds 0 \
  --detector-backend mock \
  --mock-box-position center \
  --skip-clip \
  --depth-scale 1000 \
  --view-preset tabletop_3 \
  --camera-name base_camera \
  --enable-closed-loop-reobserve \
  --closed-loop-max-extra-views 1 \
  --fail-on-child-error \
  --output-dir outputs/h200_smoke_closed_loop_reobserve_mock
```

Closed-loop ambiguity HF benchmark:

```bash
PYTHONPATH=$PWD python scripts/run_multiview_fusion_benchmark.py \
  --queries-file configs/ambiguity_queries.txt \
  --seeds 0 1 2 \
  --detector-backend hf \
  --skip-clip \
  --depth-scale 1000 \
  --view-preset tabletop_3 \
  --camera-name base_camera \
  --enable-closed-loop-reobserve \
  --closed-loop-max-extra-views 1 \
  --fail-on-child-error \
  --output-dir outputs/ambiguity_tabletop3_hf_no_clip_closed_loop
```

Closed-loop ambiguity policy report:

```bash
PYTHONPATH=$PWD python scripts/generate_reobserve_policy_report.py \
  --benchmark Ambiguity_tabletop3_HF_no_CLIP_closed_loop=outputs/ambiguity_tabletop3_hf_no_clip_closed_loop \
  --benchmark Ambiguity_tabletop3_HF_with_CLIP_closed_loop=outputs/ambiguity_tabletop3_hf_with_clip_closed_loop \
  --output-md outputs/reobserve_policy_report_ambiguity_tabletop3_hf_closed_loop.md \
  --output-json outputs/reobserve_policy_report_ambiguity_tabletop3_hf_closed_loop.json
```

Latest H200 policy-metric rerun:

| benchmark | runs | selected_frac | mean_confidence | reobserve_trigger_rate | reason_counts |
| --- | ---: | ---: | ---: | ---: | --- |
| HF tabletop_3 fusion no CLIP reobserve v2 | 6 | 1.0000 | 0.5282 | 0.0000 | confident_enough: 6 |
| HF tabletop_3 fusion with CLIP reobserve v2 | 6 | 1.0000 | 0.7091 | 0.0000 | confident_enough: 6 |

Interpretation:

- Corrected HF fusion remains stable after adding the policy artifact.
- CLIP increases selected-object confidence in this small benchmark, but does
  not change selected-object rate or re-observation decisions.
- The current default policy does not trigger on the corrected exact-object HF
  benchmark because fused selections are considered confident enough.

Paper figure pack:

```bash
PYTHONPATH=$PWD python scripts/build_paper_figure_pack.py \
  --output-dir outputs/paper_figure_pack_latest
```

## Current Bottlenecks

1. Detector candidate multiplicity is still low in most exact-object settings.
2. CLIP reranking has no current top-1 effect because candidate sets are too small.
3. Corrected virtual multi-view capture now gives compact memory in the small
   HF no-CLIP and with-CLIP benchmarks, but the claim still needs broader query
   and seed coverage before becoming a final result.
4. Memory merge behavior is now plausible at `0.08 m`, but selection traces need
   to be easier to inspect for paper figures.
5. The RGB-D lifting camera convention bug is fixed for `cam2world_gl`, but
   future geometry diagnostics should continue to log the convention explicitly.
6. The closed-loop re-observation path executes on ambiguity HF benchmarks, but
   one suggested extra virtual view does not yet reduce final uncertainty.
7. Real robot control is still intentionally absent; placeholder pick success is
   not an end-to-end grasp metric.

## Next Recommended Milestone

Improve selected-object continuity for closed-loop extra observations,
especially in the HF with-CLIP condition:

1. Add a compact continuity rule or diagnostic that explicitly prefers merging
   extra-view observations back into the currently selected object when the
   geometric evidence is compatible.
2. Inspect whether CLIP-weighted candidates are shifting the extra observation
   into a third memory slot even when the re-observation view is support-aware.
3. Keep real robot control and web demo out of scope until closed-loop
   perception improves a measurable policy or memory metric.

Candidate paper framing:

> The current prototype establishes a runnable language-query-to-3D-target
> baseline and demonstrates that confidence-aware 3D fusion depends on careful
> camera-frame convention handling. After correcting the RGB-D lifting convention,
> virtual three-view fusion produces compact object memory in the HF benchmark.
> CLIP increases fused confidence but does not improve geometric consistency or
> selected-object rate in the current small setting.
