# Query-to-Grasp Paper Milestone Log

Last updated: 2026-04-30

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

Scope update:

- The current publishable v1 is a target-retrieval and active re-observation
  system for grasp preparation. It should not claim real-robot execution.
- Placeholder pick remains useful as a target-validation artifact, but
  `pick_success_rate = 0.0` is expected and should not be interpreted as a
  manipulation result.
- The next quality-upgrade phase has started: an opt-in simulated ManiSkill
  grasp baseline now executes real low-level simulated actions, while keeping
  retrieval and grasp-control metrics reported separately.

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
| Selected-object continuity policy report | Compact ambiguity closed-loop rerun after preferred-merge continuity rule | H200: `outputs/h200_60071_selected_continuity_ambiguity_compact_seed0/reobserve_policy_report_selected_continuity.md` |
| Post-selection continuity policy report | Compact ambiguity rerun after adding final-selection continuity bias | `outputs/h200_60071_post_selection_continuity_ambiguity_compact_seed0/reobserve_policy_report_post_selection_continuity.md` |
| Post-selection margin sweep (no CLIP) | Compact ambiguity margin sweep for post-selection continuity without CLIP | `outputs/h200_60071_post_selection_margin_sweep_compact_seed0/no_clip/margin_sweep_summary.md` |
| Post-selection margin sweep (with CLIP) | Compact ambiguity margin sweep for post-selection continuity with CLIP | `outputs/h200_60071_post_selection_margin_sweep_compact_seed0/with_clip/margin_sweep_summary.md` |
| Absorber-aware continuity policy report | Accepted compact ambiguity closed-loop rerun after supported near-gap and absorber-aware continuity fixes | `outputs/h200_60071_absorber_aware_continuity_compact_seed01234/reports/reobserve_policy_report.md` |
| Full ambiguity absorber-aware validation | Full ambiguity query file, seeds 0-4, no-CLIP and with-CLIP | `outputs/h200_60071_absorber_aware_full_ambiguity_seed01234/reports/reobserve_policy_report.md` |
| Attribute residual diagnosis | Trace-level diagnosis of full-validation residuals without rerunning benchmarks | `outputs/h200_60071_attribute_residual_diagnostics_existing/reports/residual_diagnosis.md` |
| Attribute trace-field targeted validation | Targeted H200 rerun confirming additive attribute/point/support trace fields | `outputs/h200_60071_attribute_trace_fields_targeted/trace_exports/trace_field_validation.json` |
| Simulated top-down grasp smoke | Oracle, mock-query, and HF `red cube` sanity checks for opt-in ManiSkill low-level control | `outputs/h200_hf_sim_pick_red_cube_seed012/benchmark_summary.json` |
| Simulated top-down compact grasp baseline | Compact query set, seeds 0-4, no-CLIP and with-CLIP, with real simulated grasp attempts | `outputs/h200_60071_sim_topdown_singleview_compact_seed01234/reports/sim_topdown_singleview_report.md` |
| Simulated grasp-point diagnosis | Target-vs-oracle pose analysis for exact successes and compact failures | `outputs/h200_60071_sim_topdown_singleview_compact_seed01234/reports/grasp_point_diagnosis.md` |
| Single-view shifted-crop sim grasp | Accepted refined single-view grasp target with compact `PickCube-v1` success | `outputs/h200_60071_grasp_shifted_crop_compact_seed01234_v2/with_clip/benchmark_summary.json` |
| Multi-view sim-grasp bridge ablation | Opt-in `sim_topdown` execution from fused tabletop_3 and closed-loop selected objects | `outputs/h200_60071_multiview_sim_pick_bridge_ablation_seed01234` |
| Fused-memory grasp point ablation | Accepted multi-view `sim_topdown` path using fused `memory_grasp_world_xyz` instead of semantic centers | `outputs/h200_60071_multiview_memory_grasp_point_ablation_seed01234` |
| StackCube task-aware grasp guard | Expanded `StackCube-v1` multi-view refined pick validation using semantic selected-object centers | `outputs/h200_60071_stackcube_task_guard_expanded_seed0_49` |
| StackCube expanded failure analysis | Failure taxonomy explaining why closed-loop improves uncertainty but not StackCube pick success | `outputs/h200_60071_stackcube_task_guard_expanded_seed0_49/reports/stackcube_guard_failure_report.md` |
| Oracle target-source ablation | Privileged object-pose upper-bound for the same `sim_topdown` controller on PickCube and StackCube | `outputs/h200_60071_oracle_pick_ablation/reports/oracle_target_source_comparison.md` |
| Multi-task simulated-grasp section draft | Paper-ready prose for PickCube/StackCube simulated pick evidence and limitations | `docs/paper_multitask_sim_grasp_section.md` |
| Full manuscript draft skeleton | Coherent first-pass Markdown paper draft assembled from the outline, results, and multi-task section | `docs/paper_manuscript_draft.md` |
| Related-work citation plan | Candidate citation buckets, search terms, and metadata verification checklist for the conference draft | `docs/paper_related_work_citation_plan.md` |
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
| Selected-object continuity rule | Done | H200 compact ambiguity rerun with `--enable-selected-object-continuity` | Preferred-merge continuity improves selected-object association in compact ambiguity stress tests, but closed-loop resolution remains `0.0` and extra views still sometimes attach to third objects. |
| Post-selection continuity rule | Done | H200 compact ambiguity rerun with `--enable-post-reobserve-selection-continuity` | The new final-selection continuity hook is instrumented and runnable, but `selection_continuity_apply_rate = 0.0` in the current compact ambiguity setting, so the current margin/eligibility logic is too conservative to change outcomes. |
| Post-selection continuity margin sweep | Done | H200 compact ambiguity sweep across margins `0.03, 0.05, 0.08, 0.12` | Raising the margin from `0.03` to `0.05+` activates post-selection continuity in one compact ambiguity case (`apply_rate = 0.2500` per benchmark), but closed-loop resolution stays `0.0`. The next bottleneck is no longer margin gating alone; it is how confidence and uncertainty update after the extra view is merged. |
| Absorber-aware closed-loop continuity | Done | `outputs/h200_60071_absorber_aware_continuity_compact_seed01234/reports/reobserve_policy_report.md` | The accepted compact H200 rerun resolves all final policy triggers (`still_needed_rate = 0.0000`) while preserving the third-object acceptance gate (`third_object_rate = 0.1000`). |
| Full ambiguity absorber-aware validation | Done | `outputs/h200_60071_absorber_aware_full_ambiguity_seed01234/reports/reobserve_policy_report.md` | The accepted policy completes the full ambiguity file with `55/55` successful runs in both modes. Residual uncertainty concentrates in attribute-style queries such as `red block` and `red cube`, so the next step is diagnostic, not immediate tuning. |
| Attribute-query residual diagnosis | Done | `outputs/h200_60071_attribute_residual_diagnostics_existing/reports/residual_diagnosis.md` | Existing full-run traces isolate three residual types: 3D point insufficiency for `red cube` seed 3, selected-view support conflict for `red block` seed 3, and same-phrase attribute ambiguity plus third-object absorption for `red block` seed 0. |
| Attribute trace-field validation | Done | `outputs/h200_60071_attribute_trace_fields_targeted/trace_exports/trace_field_validation.json` | Targeted H200 traces confirm all new diagnostic fields are present. Residual attribute-style queries have `attribute_coverage = 1.0`, so the next bottleneck is same-phrase memory fragmentation and point/view support, not missing parsed color evidence. |
| Minimal simulated grasp executor | Done | `e453d1f` and H200 oracle/mock/HF smoke outputs | `SimulatedTopDownPickExecutor` executes opt-in `pd_ee_delta_pos` actions and reports `grasp_attempted`, `pick_success`, `task_success`, and raw trajectory diagnostics while preserving placeholder as the default. |
| Simulated top-down compact grasp baseline | Done | `outputs/h200_60071_sim_topdown_singleview_compact_seed01234/reports/sim_topdown_singleview_report.md` | The compact single-view benchmark completes `20/20` runs per mode with zero failures and `grasp_attempted_rate = 1.0`, but `pick_success_rate = 0.1000`, exposing grasp-point/target-center alignment as the next bottleneck. |
| Simulated grasp-point diagnosis | Done | `outputs/h200_60071_sim_topdown_singleview_compact_seed01234/reports/grasp_point_diagnosis.md` | Exact `red cube` successes have mean target-oracle distance `0.0171 m`, while compact failures average `0.3329 m` from oracle and almost always have high/far target points. |
| Multi-view simulated pick bridge | Done | `493f63b` and `outputs/h200_60071_multiview_sim_pick_bridge_ablation_seed01234` | Fused tabletop_3 and closed-loop selected objects now drive `sim_topdown` metrics with `0` child failures, but compact pick success remains `0.0000` because fused memory currently exposes semantic object centers rather than shifted-crop grasp points. |
| Fused-memory grasp point path | Done | `2403755` and `outputs/h200_60071_multiview_memory_grasp_point_ablation_seed01234` | Propagating per-view grasp points into memory and picking from `memory_grasp_world_xyz` lifts compact multi-view and closed-loop pick success from `0.0000` to `1.0000` in all four H200 compact modes. |
| StackCube task-aware grasp target guard | Done | `f5810ff`, `1c10aa6`, and `outputs/h200_60071_stackcube_task_guard_expanded_seed0_49` | `StackCube-v1` refined multi-view picking now uses the semantic selected-object center. The expanded 50-seed validation reaches tabletop pick success `0.6200` and closed-loop pick success `0.5200` with `0` failures, while a PickCube regression remains `1.0000` with `memory_grasp_world_xyz`. |
| Multi-task simulated-grasp section draft | Done | `docs/paper_multitask_sim_grasp_section.md` | The PickCube full-ambiguity, StackCube expanded guard, and StackCube failure taxonomy results are now written as manuscript-style prose with explicit pick-vs-task-success framing. |
| Full manuscript draft skeleton | Done | `docs/paper_manuscript_draft.md` | A first complete Markdown paper skeleton now covers abstract, introduction, method, setup, results, multi-task simulated grasp evaluation, limitations, conclusion, and artifact appendix. |
| Conference manuscript draft v0.2 | Done | `docs/paper_manuscript_draft.md` and `docs/paper_related_work_citation_plan.md` | The draft now replaces the related-work placeholder with citation-scaffold subsections, adds figure/table callouts tied to the paper pack, and frames the main results as geometry correction, fused PickCube grasp targets, and StackCube cross-task limitations. |
| Oracle target-source ablation tooling | Done | `scripts/run_oracle_pick_benchmark.py` and `outputs/h200_60071_oracle_pick_ablation/reports/oracle_target_source_comparison.md` | The same `sim_topdown` controller reaches PickCube oracle pick success `1.0000` and StackCube oracle pick success `0.9400` over 50 seeds each, creating a privileged upper-bound row for target-source comparisons. |

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

### Closed-Loop Continuity Diagnostic

Source:

H200: `outputs/h200_60071_selected_continuity_ambiguity_compact_seed0/reobserve_policy_report_selected_continuity.md`

Previous compact ambiguity closed-loop baseline (before the continuity rule):

| setting | selected_assoc_rate | final_selected_absorber_rate | third_object_rate |
| --- | ---: | ---: | ---: |
| no CLIP | 0.2500 | 0.5000 | 0.5000 |
| with CLIP | 0.0000 | 0.2500 | 0.5000 |

After enabling `--enable-selected-object-continuity`:

| setting | selected_assoc_rate | final_selected_absorber_rate | third_object_rate | preferred_merge_rate | resolution_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no CLIP | 0.5000 | 0.5000 | 0.2500 | 0.5000 | 0.0000 |
| with CLIP | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.0000 |

Interpretation:

The preferred-merge continuity rule measurably improves whether extra-view
observations return to the initially selected object, especially in the no-CLIP
setting. However, the final policy trigger rate still does not decrease and the
resolution rate remains `0.0`. This means continuity helps memory association,
but is not yet sufficient to resolve the dominant uncertainty that drives
re-observation.

Paper note:

> Closed-loop ambiguity results suggest that continuity-aware memory updates are
> useful but incomplete. The next coding step should target the remaining gap
> between association and resolution, likely by improving the selected-object
> continuity rule under CLIP or tightening how extra views compete with third
> objects in memory.

### Post-Selection Continuity Diagnostic

Source:

`outputs/h200_60071_post_selection_continuity_ambiguity_compact_seed0/reobserve_policy_report_post_selection_continuity.md`

After adding `--enable-post-reobserve-selection-continuity` with
`--post-reobserve-selection-margin 0.03`:

| setting | selected_assoc_rate | third_object_rate | preferred_merge_rate | selection_continuity_apply_rate | resolution_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no CLIP | 0.5000 | 0.2500 | 0.5000 | 0.0000 | 0.0000 |
| with CLIP | 0.2500 | 0.2500 | 0.2500 | 0.0000 | 0.0000 |

Row-level diagnostics show why the new rule had no effect:

- `block`: continuity was eligible, but rejected because `confidence_gap_exceeds_margin`
- `object`: the preferred object was already selected
- `cube` / `container`: the preferred object never received the extra-view observation

Interpretation:

The added post-selection hook is not wrong; it is simply inactive in the
current compact benchmark. The next experiment should therefore be a tiny margin
sweep or confidence-gap analysis, not another large control-flow rewrite.

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

### Absorber-Aware Continuity Compact Benchmark

Source:
H200 `outputs/h200_60071_absorber_aware_continuity_compact_seed01234`

Setup:

- Queries: `cube`, `block`, `container`, `object`
- Seeds: `0 1 2 3 4`
- Conditions: HF no-CLIP and HF with-CLIP
- Closed-loop policy: support-aware extra view, selected-object continuity,
  supported near-gap policy floor, and absorber-aware post-reobserve continuity

| setting | runs | failed | final trigger | resolution rate | still-needed rate | third-object rate | delta confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| support_no_clip baseline | 20 | 0 | 0.3500 | 0.3000 | 0.3500 | 0.3000 | 0.0182 |
| support_with_clip baseline | 20 | 0 | 0.3500 | 0.3000 | 0.3500 | 0.3000 | 0.0180 |
| selection_sanity_no_clip baseline | 20 | 0 | 0.1000 | 0.5500 | 0.1000 | 0.1000 | 0.0344 |
| selection_sanity_with_clip baseline | 20 | 0 | 0.1000 | 0.5500 | 0.1000 | 0.1000 | 0.0344 |
| stageaware_rejected_no_clip | 20 | 0 | 0.0000 | 0.6500 | 0.0000 | 0.1500 | 0.0322 |
| stageaware_rejected_with_clip | 20 | 0 | 0.0000 | 0.6500 | 0.0000 | 0.1500 | 0.0321 |
| absorber_aware_no_clip | 20 | 0 | 0.0000 | 0.6500 | 0.0000 | 0.1000 | 0.0344 |
| absorber_aware_with_clip | 20 | 0 | 0.0000 | 0.6500 | 0.0000 | 0.1000 | 0.0344 |

Interpretation:

The supported near-gap policy floor resolves the two remaining compact cube
cases, but by itself it allowed one extra third-object absorber case. The
absorber-aware post-selection guard keeps the resolution gain while returning
third-object involvement to the accepted `0.1000` rate. This is the current
stable closed-loop perception baseline for paper tables.

### Full Ambiguity Absorber-Aware Validation

Source:
H200 `outputs/h200_60071_absorber_aware_full_ambiguity_seed01234`

Setup:

- Queries: `configs/ambiguity_queries.txt`
- Seeds: `0 1 2 3 4`
- Conditions: HF no-CLIP and HF with-CLIP
- Closed-loop policy: same accepted absorber-aware policy as the compact
  baseline

| setting | runs | failed | final trigger | resolution rate | still-needed rate | third-object rate | delta confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| absorber_aware_full_no_clip | 55 | 0 | 0.0909 | 0.3273 | 0.0909 | 0.1091 | 0.0186 |
| absorber_aware_full_with_clip | 55 | 0 | 0.0545 | 0.3273 | 0.0545 | 0.0727 | 0.0184 |

Residual summary:

| query | no-CLIP final trigger | with-CLIP final trigger | note |
| --- | ---: | ---: | --- |
| `red block` | 0.4000 | 0.4000 | Dominant remaining attribute-query residual. |
| `red cube` | 0.2000 | 0.2000 | Residual includes a `too_few_3d_points` case. |
| `cup` | 0.2000 | 0.0000 | With-CLIP removes the residual trigger in this full run. |
| `mug` | 0.2000 | 0.0000 | With-CLIP removes the residual trigger in this full run. |
| `cube` | 0.0000 | 0.0000 | Fully resolves, but still has third-object absorbers in some seeds. |

Interpretation:

The full ambiguity validation passes the broad execution gate: both modes
complete `55/55` runs with zero failures and positive selected-confidence
deltas. The compact acceptance result still holds, but the full query set shows
that residual uncertainty is concentrated in attribute-style queries, especially
`red block` and `red cube`. With-CLIP does not change the resolution rate, but
it reduces final still-needed rate from `0.0909` to `0.0545` and third-object
involvement from `0.1091` to `0.0727`. The next step should inspect residual
selection traces, label votes, final reobserve reasons, point counts, and
whether color attributes are represented in memory before adding any new policy
logic.

### Attribute Residual Diagnosis

Source:
`outputs/h200_60071_attribute_residual_diagnostics_existing/reports/residual_diagnosis.md`

Method:

- Reused the existing full-validation child artifacts.
- Inspected parsed queries, selection traces, label votes, final reobserve
  reasons, point counts, view support, extra-view absorbers, and continuity
  traces.
- Did not rerun benchmarks or change code.

| case | final reason | classification |
| --- | --- | --- |
| `red cube`, seed 3 | `too_few_3d_points` | 3D point insufficiency: selected object has 3 views but only `56.3` mean points. |
| `red block`, seed 3 | `insufficient_view_support` | Selected-view support conflict: final selected object has 1 view despite strong point support. |
| `red block`, seed 0 | `ambiguous_top_candidates` | Attribute evidence is represented only as the phrase label `red block`, with a third absorber also involved. |
| no-CLIP `cup`/`mug`, seed 3 | `ambiguous_top_candidates` | Third-object absorption without an attribute term. |

Interpretation:

The next patch should not be a broad policy retune. The traces show a mixed
failure set: one geometry/point-count case, one view-support conflict, and one
attribute-style same-phrase ambiguity. Source inspection matches this diagnosis:
the selector tries normalized prompt, target name, and synonyms, while object
memory stores only `label_votes`; color attributes are not maintained as an
independent memory term.

### Attribute Trace-Field Targeted Validation

Source:
`outputs/h200_60071_attribute_trace_fields_targeted`

Setup:

- no-CLIP: `red block`, `red cube`, `cup`, `mug` with seeds `0,3`
- with-CLIP: `red block`, `red cube` with seeds `0,3`
- Same accepted absorber-aware closed-loop settings as the full validation
- Extra validation artifact:
  `outputs/h200_60071_attribute_trace_fields_targeted/trace_exports/trace_field_validation.json`

Status:

| step | status |
| --- | --- |
| `unit_policy_tests` | 0 |
| `targeted_no_clip` | 0 |
| `targeted_with_clip` | 0 |
| `export_trace_samples` | 0 |
| `finished_at` | ok |

Trace validation:

| mode | rows | missing required fields |
| --- | ---: | --- |
| no-CLIP | 8 | none |
| with-CLIP | 4 | none |

Key trace readout:

| case | attribute coverage | same-phrase competitors | point/view signal |
| --- | ---: | ---: | --- |
| no-CLIP `red block`, seed 0 | 1.0000 | 1 | Both top objects have full `red block` labels and strong support. |
| with-CLIP `red block`, seed 0 | 1.0000 | 1 | Same near-tie pattern with a very small confidence gap. |
| `red block`, seed 3 | 1.0000 | 4 | Selected object has one view; strongest multi-view alternative has only `56.3` mean points. |
| `red cube`, seed 3 | 1.0000 | 3 | Selected object has three views but only `56.3` mean points. |

Interpretation:

The targeted trace fields refine the earlier diagnosis. The residual
attribute-style queries are not failing because the parsed color attribute is
missing from the selected trace: each targeted `red block` and `red cube`
residual reports `attribute_coverage = 1.0`. The remaining issue is that
multiple memory objects carry the same full phrase label, while different
objects trade off view count against point support. The next behavior patch
should therefore target same-phrase fragmentation or point/view support
accounting, not broad attribute-aware selection.

### Simulated Top-Down Grasp Baseline

Source:
`outputs/h200_60071_sim_topdown_singleview_compact_seed01234`

Setup:

- Environment: `PickCube-v1`
- Executor: opt-in `SimulatedTopDownPickExecutor`
- Control mode: `pd_ee_delta_pos`
- Queries: `cube`, `block`, `container`, `object`
- Seeds: `0 1 2 3 4`
- Conditions: HF no-CLIP and HF with-CLIP

Smoke validation:

| check | result |
| --- | --- |
| H200 focused unit tests | `21 passed` |
| oracle target, seed 0 | `pick_success = true` |
| mock query smoke | completed with `grasp_attempted = true` |
| HF `red cube`, seeds 0-2 | `pick_success_rate = 1.0000` |
| placeholder regression smoke | `placeholder_not_executed`, `grasp_attempted = false` |

Compact baseline:

| setting | runs | failed | 3D target | grasp attempted | pick success | task success | mean 3D points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sim_topdown_no_clip | 20 | 0 | 1.0000 | 1.0000 | 0.1000 | 0.0000 | 2778.6000 |
| sim_topdown_with_clip | 20 | 0 | 1.0000 | 1.0000 | 0.1000 | 0.0000 | 2778.6000 |

Per-query pick success:

| query | no-CLIP | with-CLIP | note |
| --- | ---: | ---: | --- |
| `block` | 0.4000 | 0.4000 | Success only in seeds 1 and 2, where lifted point support is small and centered. |
| `cube` | 0.0000 | 0.0000 | Broad `cube` detections lift many points but the top-down target is not grasp-confirmed. |
| `container` | 0.0000 | 0.0000 | Query is runnable but does not produce successful grasp confirmation. |
| `object` | 0.0000 | 0.0000 | Generic target is runnable but not yet a reliable grasp point. |

Interpretation:

This is the first honest downstream control baseline. The system now converts a
selected 3D target into real ManiSkill low-level actions and logs grasp metrics
separately from task placement success. The positive oracle and exact
`red cube` smoke tests show that the controller can grasp when the target point
is suitable. The compact query result shows the next bottleneck: broad
single-view detections often lift a semantic object region whose median/center
is not a reliable top-down grasp point. CLIP does not change this outcome in the
current compact setting.

Follow-up grasp-point diagnosis:

Source:
`outputs/h200_60071_sim_topdown_singleview_compact_seed01234/reports/grasp_point_diagnosis.md`

| group | runs | pick success | mean target-oracle distance | mean xy error | mean z error | high-z rate | far-xy rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact_red_cube | 3 | 1.0000 | 0.0171 | 0.0058 | -0.0160 | 0.0000 | 0.0000 |
| compact failures | 36 | 0.0000 | 0.3329 | 0.1864 | 0.2714 | 0.9444 | 1.0000 |
| compact successes | 4 | 1.0000 | 0.0169 | 0.0052 | -0.0160 | 0.0000 | 0.0000 |

Interpretation:

The controller is not the main failure source in the compact simulated-grasp
baseline. Successful runs, including exact `red cube` and compact `block` seeds
1/2, choose target points close to the oracle cube pose. Failed compact runs
usually select points far from the cube and high above it. The next patch should
therefore diagnose or refine 3D target selection for graspability, such as
filtering broad-box points to a plausible tabletop workspace or adding an
explicit grasp-point candidate stage, before changing low-level controller
timings.

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

Simulated top-down single-view compact benchmark:

```bash
PYTHONPATH=$PWD python scripts/run_single_view_pick_benchmark.py \
  --queries-file configs/ambiguity_queries_compact.txt \
  --seeds 0 1 2 3 4 \
  --detector-backend hf \
  --skip-clip \
  --depth-scale 1000 \
  --control-mode pd_ee_delta_pos \
  --pick-executor sim_topdown \
  --output-dir outputs/sim_topdown_singleview_compact_no_clip
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
3. Corrected virtual multi-view capture gives compact memory in the small HF
   benchmarks, and the full absorber-aware validation now confirms zero-failure
   execution across the complete ambiguity query file for seeds `0..4`.
4. Memory merge behavior is now plausible at `0.08 m`, but selection traces need
   to be easier to inspect for paper figures.
5. The RGB-D lifting camera convention bug is fixed for `cam2world_gl`, but
   future geometry diagnostics should continue to log the convention explicitly.
6. The accepted compact closed-loop re-observation path now reduces final
   policy uncertainty in the diagnostic ambiguity benchmark; full-query
   residuals now point to same-phrase memory fragmentation and point/view
   support tradeoffs as the next bottleneck.
7. Re-observation remains a virtual-camera perception loop rather than learned
   view planning or robot motion.
8. Real robot control is still intentionally absent; simulated pick success and
   ManiSkill task success must be reported separately from any real-robot claim.
9. The opt-in simulated grasp executor is now connected. The first broad-query
   compact baseline was low, but the accepted shifted-crop refined grasp target
   improves compact success from `0.1000` to `1.0000`; the current compact
   simulated-grasp bottleneck is no longer target height or lateral point
   selection for `PickCube-v1`.
10. The fused-memory grasp point path is now accepted. Multi-view and
   closed-loop compact picks use `memory_grasp_world_xyz` in all runs and reach
   `pick_success_rate = 1.0000`, so the next bottleneck is broader task
   coverage rather than the PickCube fused target source.

## Next Recommended Milestone

Broaden the simulated manipulation baseline without destabilizing the accepted
PickCube path:

1. Refresh paper reports/figure pack to include fused-memory grasp-point results.
2. Add one additional ManiSkill pick-style task or object configuration with the
   same opt-in `sim_topdown` reporting contract.
3. Keep fused semantic centers unchanged for retrieval and reporting while
   treating grasp points as downstream execution targets.
4. Keep detector backends, fusion weights, training, web demo, controller timing,
   and real robot deployment out of scope for this grasp-baseline phase.

Latest accepted simulated grasp refinements:

| benchmark | runs | failed | pick success | mean XY error | far-XY rate | high-Z rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact red cube refined sim top-down | 3 | 0 | 1.0000 | 0.0049 | 0.0000 | 0.0000 |
| Compact no CLIP refined sim top-down | 20 | 0 | 0.3500 | 0.1005 | 0.6500 | 0.0000 |
| Compact with CLIP refined sim top-down | 20 | 0 | 0.3500 | 0.1005 | 0.6500 | 0.0000 |
| Compact no CLIP shifted-crop refined sim top-down | 20 | 0 | 1.0000 | 0.0065 | 0.0000 | 0.0000 |
| Compact with CLIP shifted-crop refined sim top-down | 20 | 0 | 1.0000 | 0.0065 | 0.0000 | 0.0000 |

Compared with the previous compact refined baseline, mean XY error improved
from `0.1239 m` to `0.1005 m`, far-XY rate improved from `0.9000` to `0.6500`,
and high-Z rate stayed at `0.0000`. The shifted-crop fallback then reduced mean
XY error to `0.0065 m`, far-XY rate to `0.0000`, and compact pick success to
`1.0000` for both no-CLIP and with-CLIP compact runs. This is strong evidence
that broad single-view failures were caused by detector boxes whose upper region
missed the graspable object support; the semantic center remains unchanged, but
the refined grasp target can use a conservative downward crop when the original
box has no elevated object-like support.

Latest accepted multi-view sim-grasp bridge:

| benchmark | runs | failed | grasp attempted | pick success | closed-loop resolution | still-needed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact red cube tabletop_3 no CLIP | 3 | 0 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| Compact tabletop_3 no CLIP | 20 | 0 | 1.0000 | 0.0000 | 0.0000 | 0.6500 |
| Compact tabletop_3 with CLIP | 20 | 0 | 1.0000 | 0.0000 | 0.0000 | 0.6500 |
| Compact closed-loop no CLIP | 20 | 0 | 1.0000 | 0.0000 | 0.6500 | 0.0000 |
| Compact closed-loop with CLIP | 20 | 0 | 1.0000 | 0.0000 | 0.6500 | 0.0000 |

The bridge result is a successful infrastructure checkpoint, not a manipulation
success result. It shows that multi-view and closed-loop selected objects can
drive the same simulated executor and emit stable downstream metrics with
`0` child failures. It also isolates the next bottleneck: fused memory currently
offers only semantic object centers for picking, while the successful
single-view compact baseline depends on an explicit shifted-crop grasp point.

Latest accepted fused-memory grasp-point result:

| benchmark | runs | failed | grasp attempted | pick success | task success | target source | closed-loop resolution | still-needed |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Targeted exact red cube tabletop_3 no CLIP | 3 | 0 | 1.0000 | 1.0000 | 0.3333 | `memory_grasp_world_xyz` | 0.0000 | 0.0000 |
| Targeted compact broad tabletop_3 no CLIP | 20 | 0 | 1.0000 | 1.0000 | 0.1500 | `memory_grasp_world_xyz` | 0.0000 | 0.6500 |
| Compact tabletop_3 no CLIP | 20 | 0 | 1.0000 | 1.0000 | 0.1500 | `memory_grasp_world_xyz` | 0.0000 | 0.6500 |
| Compact tabletop_3 with CLIP | 20 | 0 | 1.0000 | 1.0000 | 0.1500 | `memory_grasp_world_xyz` | 0.0000 | 0.6500 |
| Compact closed-loop no CLIP | 20 | 0 | 1.0000 | 1.0000 | 0.1500 | `memory_grasp_world_xyz` | 0.6500 | 0.0000 |
| Compact closed-loop with CLIP | 20 | 0 | 1.0000 | 1.0000 | 0.1500 | `memory_grasp_world_xyz` | 0.6500 | 0.0000 |

This result upgrades the multi-view bridge from infrastructure-only to a
successful PickCube simulated-grasp baseline. The semantic fused center remains
unchanged for retrieval and confidence, while the downstream executor uses a
separately fused grasp point. The remaining paper limitation is breadth:
results are still compact-query `PickCube-v1` simulated control, not robust
multi-task manipulation or real-robot execution.

Latest broader-task simulated pick smoke:

| benchmark | env | query | runs | failed | 3D target | grasp attempted | pick success | task success | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Single-view no CLIP | `StackCube-v1` | `red cube` | 5 | 0 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | pick-only, not stack placement |
| Single-view with CLIP | `StackCube-v1` | `red cube` | 5 | 0 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | pick-only, not stack placement |

This smoke validates the same query-driven RGB-D to HF detection to refined
grasp target to `sim_topdown` control chain on a second ManiSkill task that
uses a task-specific grasp flag (`is_cubeA_grasped`). It should be reported as
broader task compatibility for picking, not as a completed stacking controller:
raw ManiSkill task success remains `0.0000` because the current executor lifts
the cube but does not place it on cubeB.

Latest overnight full-grasp validation:

| benchmark | env | runs | failed | grasp attempted | pick success | task success | closed-loop resolution | still-needed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full ambiguity tabletop_3 no CLIP | `PickCube-v1` | 55 | 0 | 1.0000 | 1.0000 | 0.1455 | 0.0000 | 0.4182 |
| Full ambiguity tabletop_3 with CLIP | `PickCube-v1` | 55 | 0 | 1.0000 | 1.0000 | 0.1455 | 0.0000 | 0.3818 |
| Full ambiguity closed-loop no CLIP | `PickCube-v1` | 55 | 0 | 1.0000 | 1.0000 | 0.1455 | 0.3273 | 0.0909 |
| Full ambiguity closed-loop with CLIP | `PickCube-v1` | 55 | 0 | 1.0000 | 1.0000 | 0.1455 | 0.3273 | 0.0545 |
| StackCube single-view no CLIP | `StackCube-v1` | 20 | 0 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| StackCube single-view with CLIP | `StackCube-v1` | 20 | 0 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| StackCube tabletop_3 no CLIP | `StackCube-v1` | 20 | 0 | 1.0000 | 0.5500 | 0.0000 | 0.0000 | 0.8000 |
| StackCube tabletop_3 with CLIP | `StackCube-v1` | 20 | 0 | 1.0000 | 0.5500 | 0.0000 | 0.0000 | 0.8000 |
| StackCube closed-loop no CLIP | `StackCube-v1` | 20 | 0 | 1.0000 | 0.5500 | 0.0000 | 0.3000 | 0.5000 |
| StackCube closed-loop with CLIP | `StackCube-v1` | 20 | 0 | 1.0000 | 0.5500 | 0.0000 | 0.3000 | 0.5000 |

This overnight validation upgrades the PickCube simulated grasp story from
compact-only to full ambiguity coverage: all four 55-run full-query multi-view
and closed-loop modes complete with `0` failures and `pick_success_rate =
1.0000`. The low `task_success_rate` is expected because `sim_topdown` measures
grasp/lift success, while ManiSkill task success can require additional task
completion conditions.

The same run also clarifies the broader-task boundary. `StackCube-v1` is stable
for query-driven single-view pick-only control across seeds `0..19`, but
tabletop_3 and closed-loop multi-view runs remain at `pick_success_rate =
0.5500`. Closed-loop reduces uncertainty triggers but does not improve grasp
success on this task, so the next technical bottleneck is task-general
multi-view fused grasp target quality, not CLIP or the top-down controller.

Latest StackCube task-aware grasp target guard:

| benchmark | env | runs | failed | grasp attempted | pick success | task success | target source | guard applied |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Expanded tabletop_3 no CLIP | `StackCube-v1` | 50 | 0 | 1.0000 | 0.6200 | 0.0000 | `task_guard_selected_object_world_xyz` | true |
| Expanded tabletop_3 with CLIP | `StackCube-v1` | 50 | 0 | 1.0000 | 0.6200 | 0.0000 | `task_guard_selected_object_world_xyz` | true |
| Expanded closed-loop no CLIP | `StackCube-v1` | 50 | 0 | 1.0000 | 0.5200 | 0.0000 | `task_guard_selected_object_world_xyz` | true |
| Expanded closed-loop with CLIP | `StackCube-v1` | 50 | 0 | 1.0000 | 0.5200 | 0.0000 | `task_guard_selected_object_world_xyz` | true |
| PickCube regression tabletop_3 no CLIP | `PickCube-v1` | 3 | 0 | 1.0000 | 1.0000 | 0.3333 | `memory_grasp_world_xyz` | false |

The semantic-vs-memory ablation showed that StackCube targeted failures improve
when multi-view refined picking uses the selected object's semantic fused
center instead of the fused memory grasp point. The accepted task-aware guard
implements that behavior only for `StackCube-v1`. The compact seeds `0..19`
first improved tabletop_3 pick-only success from the previous `0.5500` to
`0.7000`; the expanded seeds `0..49` provide the paper-facing estimate:
`0.6200` tabletop and `0.5200` closed-loop in both no-CLIP and with-CLIP modes,
with `0` child failures. Closed-loop still reduces uncertainty diagnostics but
does not produce a StackCube grasp-success gain, so this should be reported as
a limitation rather than as a tuning target. `PickCube-v1` remains protected
and continues to use `memory_grasp_world_xyz` for refined multi-view picks.
This is still a pick-only compatibility result: `task_success_rate = 0.0000`
because the executor does not stack cubeA onto cubeB.

StackCube expanded limitation/failure analysis:

| benchmark | failures | dominant classes | mean semantic-grasp XY | mean grasp XY spread |
| --- | ---: | --- | ---: | ---: |
| Expanded tabletop_3 no CLIP | 19 | wrong fused grasp observation: 14; memory fragmentation/low support: 5 | 0.0479 | 0.0145 |
| Expanded tabletop_3 with CLIP | 19 | wrong fused grasp observation: 14; memory fragmentation/low support: 5 | 0.0479 | 0.0145 |
| Expanded closed-loop no CLIP | 24 | third-object absorption: 11; wrong fused grasp observation: 8; controller/contact: 5 | 0.0514 | 0.0227 |
| Expanded closed-loop with CLIP | 24 | third-object absorption: 10; wrong fused grasp observation: 8; controller/contact: 5; memory fragmentation/low support: 1 | 0.0512 | 0.0175 |

The report classifies `86` failed grasps across the four expanded StackCube
modes. The dominant aggregate class is wrong fused grasp observation
(`44/86`), while closed-loop adds a task-specific failure mode: extra views can
be absorbed by a third object, producing `third_object_absorption` in `21`
closed-loop failures. This explains the paper limitation cleanly: closed-loop
reduces uncertainty signals, but on StackCube it can still move the effective
post-reobserve state away from a graspable cubeA pick target. The current result
is therefore a stable cross-task pick-only diagnostic, not a stacking task
completion claim.

Publication-level expectation:

- Current retrieval/re-observation version: arXiv, workshop, or diagnostic
  perception-for-manipulation submission.
- With a reliable minimal simulated grasp baseline and clean ablations:
  stronger ICRA/IROS workshop candidate and a possible full-conference story if
  the downstream grasp metric improves.
- With robust simulated grasp control, stronger baselines, and broader tasks:
  closer to RA-L/ICRA/IROS full-paper expectations.
- Real-robot validation would raise the ceiling further, but it is a separate
  project phase.

Candidate paper framing:

> The current prototype establishes a runnable language-query-to-3D-target
> baseline and demonstrates that confidence-aware 3D fusion depends on careful
> camera-frame convention handling. After correcting the RGB-D lifting convention,
> virtual three-view fusion produces compact object memory in the HF benchmark.
> The accepted absorber-aware closed-loop diagnostic further shows that a
> conservative one-extra-view policy can reduce compact ambiguity triggers when
> memory association and post-reobserve continuity are explicitly instrumented.
> Full ambiguity validation confirms stable execution across broader query
> coverage. Targeted trace diagnostics show that those residuals retain full
> parsed-attribute coverage; the next retrieval bottleneck is same-phrase memory
> fragmentation and point/view support, not benchmark reliability. The new
> opt-in simulated grasp executor connects selected 3D targets to real ManiSkill
> actions; its compact baseline shows that downstream success now depends on
> producing graspable target points, not only semantically correct object
> centers. The accepted geometry-only refined target reduces the broad-query
> high-Z failure, and the shifted-crop fallback lifts compact simulated pick
> success to `1.0000` for `PickCube-v1` compact queries. The multi-view
> sim-grasp bridge confirms that fused and closed-loop target sources can now
> produce stable downstream pick metrics. The accepted fused-memory grasp-point
> path then separates semantic object centers from downstream grasp targets and
> lifts compact and full-ambiguity PickCube multi-view and closed-loop simulated
> pick success to `1.0000`. The first StackCube validation shows that the same
> single-view pick-only chain transfers across tasks, while multi-view StackCube
> exposes task-dependent grasp target preferences. A task-aware guard
> improves StackCube multi-view pick-only success and the expanded 50-seed
> validation gives a stable cross-task estimate without regressing PickCube,
> but StackCube remains a compatibility diagnostic rather than a stacking
> completion result.
