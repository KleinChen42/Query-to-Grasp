# Paper Draft Outline

Working title:

`Query-to-Grasp: Diagnosing the Retrieval-to-Execution Gap in Open-Vocabulary RGB-D Manipulation`

## Current Claim

This project now targets an H200-scale simulated IROS/ICRA full-paper systems
claim:

> Open-vocabulary detectors can identify language-relevant objects in 2D, but
> robot manipulation requires graspable 3D action targets. Query-to-Grasp is a
> diagnostic RGB-D manipulation system that exposes the retrieval-to-execution
> gap: target-source quality, cross-view geometric consistency, and
> re-observation association determine whether language-selected objects become
> executable simulated pick or pick-place targets.

Updated paper positioning:

- The primary target is now an IROS/ICRA-style full simulated systems paper.
- H200 HF/ManiSkill runs are the authoritative experimental source; Colab/local
  mock paths are smoke and reproducibility support.
- The current placeholder pick path is valid infrastructure, but only opt-in
  simulated executors count as control evidence.
- The paper should center target-source, oracle, and task-success ablations:
  query-derived targets, fused memory grasp targets, task-aware guards, and
  privileged oracle object poses.
- The paper should read as a diagnostic systems paper, not an implementation
  log. Engineering terms such as "smoke test", "mock backend", "debug path",
  "child failure", "repo", "commit", and explicit source paths should be
  removed from manuscript prose unless they appear in a reproducibility
  appendix.

What we should not claim yet:

- Real grasp execution success.
- General cluttered-scene manipulation.
- Learned re-observation.
- Learned grasping or learned active perception.
- CLIP as the main source of retrieval improvement.
- Robust relation-heavy language grounding.
- Full non-oracle StackCube stacking completion.

Submission-level expectation:

- Current target: IROS/ICRA full paper in simulation, assuming H200-scale
  validation, strong target-source baselines, and conservative claim framing.
- RA-L remains a stretch target if task-success evidence broadens beyond the
  current StackCube bridge and the paper gains a stronger method contribution.
- With real-robot validation: materially higher ceiling, but that is a new
  project phase rather than a small extension.

## Revised Narrative Spine

The manuscript should be rewritten around four layers.

### Layer 1: Reranking Is Not the Bottleneck

CLIP should be framed as a controlled semantic reranking ablation, not as a
failed feature. In the current candidate pools, GroundingDINO usually returns
one or very few candidates, so CLIP rarely changes the top-1 selection. The
academic message is that the dominant bottleneck is downstream of 2D semantic
matching: the quality of the lifted 3D target and its suitability for execution.

### Layer 2: Geometric Consistency Enables Multi-View Memory

The OpenCV-to-OpenGL camera-frame correction should be described as a coordinate
alignment result, not as a bug fix. Persistent multi-view memory is meaningful
only when RGB-D lifting and simulator camera poses agree. Without cross-view
coordinate consistency, memory fragmentation is expected; with the correction,
the object memory becomes a valid diagnostic substrate.

### Layer 3: PickCube Demonstrates Executability

PickCube is the strongest positive result. The paper should emphasize that
fused memory grasp targets bridge semantic retrieval and executable simulated
control, reaching `1.0000` simulated pick success in the validated full-query
multi-view and closed-loop settings.

### Layer 4: StackCube Exposes the Retrieval-to-Execution Gap

StackCube should be framed as a cross-task diagnostic testbed. It is not a
claim of full language-conditioned stacking. Current StackCube evidence has
three roles:

- pick-only compatibility shows cross-task transfer of query-derived pick
  targets;
- oracle pick and oracle pick-place establish privileged upper bounds for the
  scripted controller;
- the query-pick plus oracle-place bridge tests whether query-derived cubeA
  targets can support task completion when the placement target is privileged.

The key interpretation is that remaining failures are not simply low-level
controller failures. They reflect target-source selection, task-dependent grasp
target quality, memory association, and closed-loop third-object absorption.

## Revised Section Headings

1. Introduction
2. Related Work
   - Open-Vocabulary Grounding for Manipulation
   - Language-Conditioned Robotic Manipulation
   - RGB-D Lifting and Multi-View Object Memory
   - Active Perception and Re-Observation Diagnostics
3. System Overview
4. From Language Queries to 3D Action Targets
   - Open-Vocabulary 2D Proposals and Reranking
   - Camera-Consistent RGB-D Lifting
   - Multi-View Object Memory and Target Selection
   - Re-Observation as Diagnostic Association
5. Simulated Execution and Target-Source Baselines
   - Simulated Pick Execution
   - Oracle Pick and Pick-Place Baselines
   - Query-Pick plus Oracle-Place Bridge
6. Experimental Protocol
7. Results
   - Reranking Is Not the Main Bottleneck
   - Geometric Consistency Reduces Memory Fragmentation
   - PickCube Converts Retrieval into Executable Picks
   - StackCube Reveals Cross-Task Target-Source Limits
8. Limitations and Failure Taxonomy
9. Conclusion
10. Reproducibility Appendix

## Manuscript Rewrite Targets

### Abstract

The abstract should use the new title and thesis. It should state that the paper
studies the retrieval-to-execution gap between open-vocabulary 2D detection and
graspable 3D action targets. It should preserve the established numbers:
PickCube full-query multi-view and closed-loop pick success `1.0000`;
StackCube expanded pick-only tabletop `0.6200` and closed-loop `0.5200`;
StackCube oracle pick `0.9400`; StackCube oracle pick-place task success
`0.8800`; and the accepted query-pick plus oracle-place bridge result:
single-view task success `0.7200`, tabletop_3 task success `0.5200`, and
closed-loop task success `0.4800` over 50 seeds per mode.

### Introduction

The introduction should start from the mismatch between 2D open-vocabulary
grounding and 3D executable manipulation. The system should be presented as a
diagnostic benchmark and pipeline for identifying where language-conditioned
RGB-D manipulation fails: semantic candidate pools, camera-frame consistency,
multi-view association, target-source choice, and physical execution.

### Conclusion

The conclusion should not summarize implementation milestones. It should return
to the central lesson: the hard part of query-to-grasp is not only recognizing a
named object but producing and maintaining a graspable 3D action target across
views and tasks. PickCube shows that the bridge can close under favorable
target-source conditions; StackCube shows why oracle and target-source ablations
remain necessary before claiming full language-conditioned task completion.

## Method-Section Edit Checklist

- Replace implementation-log phrasing with method concepts: proposal generation,
  RGB-D lifting, cross-view alignment, object memory, target-source selection,
  and execution baselines.
- Remove explicit source-code paths from main method prose; keep them only in
  appendix or reproducibility notes.
- Describe CLIP as an optional semantic reranking/control ablation, not as a
  central contribution.
- Describe the camera-frame conversion as coordinate-system alignment required
  for valid multi-view fusion.
- Separate semantic object centers from grasp/action targets throughout the
  method.
- Present re-observation as diagnostic association and uncertainty reduction,
  not learned active perception.
- Present privileged oracle target sources as baselines and upper bounds, not
  deployable perception results.

## Results-Section Edit Checklist

- Begin with the reranking observation: current candidate multiplicity leaves
  little room for CLIP to alter top-1, so downstream target quality matters more.
- Make the camera-frame consistency result the first major systems finding:
  geometric alignment enables meaningful multi-view memory.
- Treat PickCube as the main positive execution result and preserve the
  `1.0000` full-query pick-success claim.
- Treat StackCube as a diagnostic cross-task result, clearly separating
  pick-only, oracle placement, and query-pick plus oracle-place bridge rows.
- State that closed-loop re-observation can reduce uncertainty while still
  failing to improve StackCube manipulation success because association errors
  such as third-object absorption can affect the final target.
- Include a failure taxonomy paragraph instead of an engineering post-mortem.

## Claim Boundary Box

The paper can claim:

- open-vocabulary RGB-D target retrieval with camera-consistent 3D lifting;
- inspectable multi-view object memory and deterministic target selection;
- diagnostic re-observation that measures uncertainty and association effects;
- opt-in simulated pick and oracle pick-place baselines in ManiSkill;
- PickCube full-query simulated pick success of `1.0000`;
- StackCube pick-only compatibility and privileged oracle placement baselines;
- query-pick plus oracle-place StackCube task success as a partial privileged
  placement bridge, with the destination target supplied by oracle cubeB pose.

The paper must not claim:

- real-robot execution;
- learned grasping or learned active perception;
- robust long-horizon language-conditioned manipulation;
- robust relation-heavy language grounding;
- full non-oracle language-conditioned StackCube stacking completion;
- CLIP as the primary performance driver unless future evidence changes the
  top-1 reranking result.

## Abstract Skeleton

Language-conditioned robotic manipulation requires connecting open-vocabulary
2D perception with 3D geometric reasoning and action execution. We present
Query-to-Grasp, a modular research prototype for language-queryable target
retrieval in ManiSkill. The system parses a natural-language query, detects
candidate 2D regions with GroundingDINO, optionally reranks them with CLIP, lifts
RGB-D detections into 3D, and fuses multi-view evidence into a persistent object
memory. A safe placeholder executor validates selected targets by default, and
an opt-in simulated top-down executor connects selected 3D targets to real
ManiSkill low-level actions while reporting grasp outcomes separately from
retrieval outcomes.

Our experiments show that in the current HF GroundingDINO setting, CLIP reranking
does not change top-1 selections because detector candidate multiplicity is low.
The primary systems bottleneck is instead geometric consistency across views.
After correcting the camera-frame convention between OpenCV-style RGB-D lifting
and ManiSkill `cam2world_gl` poses, same-label cross-view spread drops from
`1.0693 m` to `0.0518 m`, and mean memory fragmentation drops from `3.3333` to
`1.3333` objects per run. These results establish a reproducible baseline for
confidence-aware 3D semantic fusion and identify the next steps toward a full
query-to-grasp system.

## Method Outline

### 1. Query Parsing

Implemented in `src/perception/query_parser.py`.

Inputs:

- Free-form query such as `red cube`.

Outputs:

- Raw query.
- Normalized prompt.
- Target name.
- Attributes.
- Synonyms.
- Conservative relation fields.

Paper framing:

The parser is intentionally lightweight and deterministic-first. It is not a
large language model contribution.

### 2. Open-Vocabulary 2D Proposals

Implemented in `src/perception/grounding_dino.py`.

Backends:

- HF GroundingDINO.
- Original adapter path.
- Mock backend for safe CI/smoke tests.

Paper framing:

GroundingDINO is used as an off-the-shelf open-vocabulary proposal model. The
paper contribution is the embodied 3D integration and diagnostics, not detector
training.

### 3. Optional CLIP Reranking

Implemented in `src/perception/clip_rerank.py`.

Current evidence:

- Exact-object HF benchmarks have `mean_raw_num_detections = 1.0`.
- Ambiguity benchmarks increase candidate count to `1.4242`, but
  `fraction_top1_changed_by_rerank = 0.0`.

Paper framing:

CLIP is retained as a confidence term and diagnostic module. Current results do
not support claiming CLIP as the primary improvement source.

### 4. RGB-D to 3D Lifting

Implemented in `src/perception/mask_projector.py`.

Key correction:

- RGB-D projection creates OpenCV-style camera points.
- ManiSkill `cam2world_gl` expects OpenGL-style camera points.
- The pipeline now applies `diag([1, -1, -1, 1])` before `cam2world_gl`.

Evidence:

- `outputs/h200_60071_extrinsic_convention/extrinsic_convention_report.md`
- `outputs/h200_60071_tabletop3_cvfix/cross_view_geometry_report_tabletop3_cvfix.md`

### 5. Multi-View Object Memory

Implemented in:

- `src/memory/object_memory_3d.py`
- `src/memory/fusion.py`

Tracked terms:

- Detection confidence.
- CLIP confidence.
- View support.
- Spatial consistency.
- Geometry confidence.
- Label votes.

Current result:

- Before camera-convention fix: `mean_num_memory_objects = 3.3333`.
- After fix: `mean_num_memory_objects = 1.3333`.

### 6. Target Selection

Implemented in:

- `src/policy/target_selector.py`

Behavior:

- Try query-derived labels in priority order.
- Select among matching memory objects by overall confidence.
- Tie-break by number of views, geometry confidence, then object id.
- Export `selection_trace.json` and `selection_trace.md`.

Evidence:

- `outputs/h200_60071_selection_trace_red_cube_seed0/selection_trace.md`

### 8. Rule-Based Re-Observation Decision

Implemented in:

- `src/policy/reobserve_policy.py`

Current behavior:

- Produces a `ReobserveDecision`.
- Uses selected confidence, top-1/top-2 confidence gap, view count, geometry
  confidence, and mean valid 3D point count.
- Writes `reobserve_decision.json` in multi-view debug runs.
- Supports an opt-in minimal closed-loop path that captures one or more
  suggested virtual views, reruns perception, and writes before/after
  diagnostics.
- This is still a virtual-camera perception loop, not learned view planning or
  low-level robot motion.

Evidence:

- `outputs/h200_60071_reobserve_smoke/reobserve_decision.json`
- H200: `outputs/h200_smoke_closed_loop_reobserve_mock/closed_loop_reobserve.json`
- H200: `outputs/h200_60071_absorber_aware_continuity_compact_seed01234/reports/reobserve_policy_report.md`
- H200: `outputs/h200_60071_absorber_aware_full_ambiguity_seed01234/reports/reobserve_policy_report.md`
- `outputs/h200_60071_attribute_residual_diagnostics_existing/reports/residual_diagnosis.md`
- `outputs/h200_60071_attribute_trace_fields_targeted/trace_exports/trace_field_validation.json`

Current closed-loop diagnostic:

- The current accepted compact H200 baseline combines support-aware extra-view
  selection, selected-object continuity, a supported near-gap policy floor, and
  absorber-aware post-reobserve continuity.
- On compact ambiguity seeds `0..4`, both no-CLIP and with-CLIP reach
  `closed_loop_resolution_rate = 0.6500`, `closed_loop_still_needed_rate =
  0.0000`, and positive selected-confidence deltas.
- The absorber-aware guard fixes the rejected stage-aware variant's third-object
  regression: `third_object_rate` returns from `0.1500` to the accepted `0.1000`
  gate while preserving the `0.6500` resolution rate.
- The full ambiguity validation completes `55/55` runs with zero failures in
  both no-CLIP and with-CLIP modes. Residual triggers concentrate in
  attribute-style queries such as `red block` and `red cube`.
- Trace-level residual diagnosis shows that the remaining cases are mixed:
  point-count insufficiency, selected-view support conflict, and same-phrase
  attribute ambiguity with third-object absorption.
- Targeted attribute trace fields show `attribute_coverage = 1.0` in the
  residual `red block` and `red cube` cases. The next bottleneck is therefore
  same-phrase fragmentation and point/view support, not missing parsed color
  evidence.
- This remains a virtual-camera perception loop, not learned view planning or
  robot camera motion.

### 7. Pick Execution

Implemented in `src/manipulation/pick_executor.py`.

Current behavior:

- Keeps `SafePlaceholderPickExecutor` as the default target-validation path.
- Adds opt-in `SimulatedTopDownPickExecutor` for `PickCube-v1` smoke and
  benchmark runs.
- Reports `grasp_attempted`, `pick_success`, `task_success`, `is_grasped`,
  executed stages, final TCP position, and raw final ManiSkill info.
- Treats `pick_success` as grasp/lift confirmation via `is_grasped`, while
  preserving raw ManiSkill task placement success as `task_success`.

Paper framing:

Use placeholder `pick_success_rate = 0.0` as expected target-validation
behavior. Use `sim_topdown` results only as simulated control evidence, and keep
retrieval and control metrics separate. The first compact result shows that a
simple top-down controller can grasp oracle/exact targets, but broad lifted
target centers are not yet reliable grasp points.

## Experiment Plan

### Experiment 1: Detector-only vs Detector + CLIP

Question:

Does CLIP reranking change top-1 or improve 3D target availability?

Current result:

| setting | runs | raw detections | top1 changed | 3D target rate |
| --- | ---: | ---: | ---: | ---: |
| HF no CLIP | 6 | 1.0000 | 0.0000 | 1.0000 |
| HF with CLIP | 6 | 1.0000 | 0.0000 | 1.0000 |
| Ambiguity no CLIP | 33 | 1.4242 | 0.0000 | 0.9394 |
| Ambiguity with CLIP | 33 | 1.4242 | 0.0000 | 0.9394 |

Artifact:

- `outputs/h200_60071_paper_baseline/outputs/paper_ablation_table.md`

### Experiment 2: Single-View vs Corrected Multi-View Fusion

Question:

Does multi-view fusion produce compact persistent object memory?

Current result:

| setting | runs | views | memory objects | same-label distance | selected confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF tabletop_3 fusion no CLIP cvfix | 6 | 3.0000 | 1.3333 | 0.0518 | 0.5282 |
| HF tabletop_3 fusion with CLIP cvfix | 6 | 3.0000 | 1.3333 | 0.0518 | 0.7091 |

Artifact:

- `outputs/h200_60071_tabletop3_cvfix_clip_ablation/fusion_comparison_table_tabletop3_cvfix_clip_ablation.md`

### Experiment 3: Camera Convention Ablation

Question:

Is cross-view fragmentation caused by the camera-frame convention?

Current result:

| convention | same-label distance | top-rank distance |
| --- | ---: | ---: |
| `cam2world_gl_direct` | 1.0693 | 1.1301 |
| `cam2world_gl_cv_to_gl` | 0.0518 | 0.0304 |

Artifact:

- `outputs/h200_60071_extrinsic_convention/extrinsic_convention_report.md`

### Experiment 4: Selection Trace Case Study

Question:

Can the system explain why the final 3D target was selected?

Current result:

- Query: `red cube`
- Selected: `obj_0000`
- Selection pool label: `red cube`
- Pool size: `2`
- Selected object has stronger view support and confidence.

Artifact:

- `outputs/h200_60071_selection_trace_red_cube_seed0/selection_trace.md`

### Experiment 5: Closed-Loop Re-Observation Continuity

Question:

When the policy triggers an extra view, does continuity-aware memory updating
help the new observation merge back into the initially selected object?

Current result:

| setting | selected_assoc_rate | final_selected_absorber_rate | third_object_rate | preferred_merge_rate | resolution_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| compact ambiguity no CLIP | 0.5000 | 0.5000 | 0.2500 | 0.5000 | 0.0000 |
| compact ambiguity with CLIP | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.0000 |

Artifact:

- H200: `outputs/h200_60071_selected_continuity_ambiguity_compact_seed0/reobserve_policy_report_selected_continuity.md`

Interpretation:

The continuity rule improves selected-object association and reduces third-object
involvement relative to the earlier compact closed-loop baseline, but it still
does not reduce the final trigger rate. This is evidence that memory association
is part of the bottleneck, not the whole bottleneck.

### Experiment 6: Post-Selection Continuity

Question:

If extra-view observations already merge back into the initially selected object,
can a small final-selection continuity bias keep that object selected and lower
closed-loop uncertainty?

Current result:

| setting | selected_assoc_rate | preferred_merge_rate | selection_continuity_apply_rate | resolution_rate |
| --- | ---: | ---: | ---: | ---: |
| compact ambiguity no CLIP | 0.5000 | 0.5000 | 0.0000 | 0.0000 |
| compact ambiguity with CLIP | 0.2500 | 0.2500 | 0.0000 | 0.0000 |

Artifact:

- `outputs/h200_60071_post_selection_continuity_ambiguity_compact_seed0/reobserve_policy_report_post_selection_continuity.md`

Interpretation:

The initial post-selection continuity hook was useful as instrumentation but
did not fire in the compact ambiguity benchmark. The later margin sweep showed
that margin gating was not the only blocker: after continuity applied, policy
uncertainty still remained flat.

### Experiment 7: Absorber-Aware Closed-Loop Re-Observation

Question:

Can a conservative supported near-gap policy floor resolve compact ambiguity
without increasing third-object extra-view absorption?

Current result:

| setting | runs | resolution_rate | still_needed_rate | third_object_rate | delta_confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| support_no_clip baseline | 20 | 0.3000 | 0.3500 | 0.3000 | 0.0182 |
| selection_sanity_no_clip baseline | 20 | 0.5500 | 0.1000 | 0.1000 | 0.0344 |
| stageaware_rejected_no_clip | 20 | 0.6500 | 0.0000 | 0.1500 | 0.0322 |
| absorber_aware_no_clip | 20 | 0.6500 | 0.0000 | 0.1000 | 0.0344 |
| absorber_aware_with_clip | 20 | 0.6500 | 0.0000 | 0.1000 | 0.0344 |

Artifact:

- `outputs/h200_60071_absorber_aware_continuity_compact_seed01234/reports/reobserve_policy_report.md`
- `outputs/h200_60071_absorber_aware_continuity_compact_seed01234/reports/ablation_table.md`

Interpretation:

The accepted absorber-aware run is the current closed-loop perception baseline:
the extra view resolves all final policy triggers in the compact benchmark while
keeping the third-object rate within the pre-set `0.10` acceptance gate.

### Experiment 8: Full Ambiguity Absorber-Aware Validation

Question:

Does the accepted absorber-aware policy remain stable on the complete ambiguity
query set, and where do residual triggers concentrate?

Current result:

| setting | runs | failed | resolution_rate | still_needed_rate | third_object_rate | delta_confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| absorber_aware_full_no_clip | 55 | 0 | 0.3273 | 0.0909 | 0.1091 | 0.0186 |
| absorber_aware_full_with_clip | 55 | 0 | 0.3273 | 0.0545 | 0.0727 | 0.0184 |

Per-query residuals:

| query | no-CLIP final trigger | with-CLIP final trigger | current reading |
| --- | ---: | ---: | --- |
| `red block` | 0.4000 | 0.4000 | Main remaining attribute-query residual. |
| `red cube` | 0.2000 | 0.2000 | Includes a 3D point insufficiency residual. |
| `cup` | 0.2000 | 0.0000 | With-CLIP removes the residual trigger. |
| `mug` | 0.2000 | 0.0000 | With-CLIP removes the residual trigger. |

Artifacts:

- `outputs/h200_60071_absorber_aware_full_ambiguity_seed01234/reports/reobserve_policy_report.md`
- `outputs/h200_60071_absorber_aware_full_ambiguity_seed01234/reports/ablation_table.md`
- `outputs/paper_figure_pack_latest/README.md`

Interpretation:

The accepted compact baseline holds as the stable closed-loop diagnostic result,
and the full ambiguity run shows that the benchmark path is reliable at broader
query coverage. With-CLIP does not improve resolution rate, but it reduces
still-needed rate and third-object involvement on the full set. The next code
decision should be based on residual traces for `red block`, `red cube`, and the
no-CLIP-only `cup`/`mug` cases, with special attention to whether color
attributes are represented in memory and selection.

Follow-up residual diagnosis:

| case | final reason | classification |
| --- | --- | --- |
| `red cube`, seed 3 | `too_few_3d_points` | 3D point insufficiency. |
| `red block`, seed 3 | `insufficient_view_support` | Selected-view support conflict. |
| `red block`, seed 0 | `ambiguous_top_candidates` | Same-phrase attribute ambiguity plus third-object absorption. |
| no-CLIP `cup`/`mug`, seed 3 | `ambiguous_top_candidates` | Third-object absorption without an attribute term. |

The selector currently tries normalized prompt, target name, and synonyms as
labels, while object memory stores `label_votes`. Parsed color attributes are
not maintained as a separate memory term, so the next implementation should
start with targeted diagnostics or one narrowly justified guard rather than a
broad attribute-aware selector rewrite.

Targeted trace-field validation:

| mode | runs | missing trace fields | key readout |
| --- | ---: | --- | --- |
| no-CLIP | 8 | none | Residual `red block`/`red cube` cases have full attribute coverage. |
| with-CLIP | 4 | none | Same attribute-coverage pattern; residuals remain point/support limited. |

The new trace fields change the interpretation: the parser and trace can expose
the color attribute, but the residual memory objects all carry the full phrase
label. The next behavior patch should focus on same-phrase memory fragmentation
or point/view support accounting, not on a broad attribute-aware selector.

### Experiment 9: Simulated Top-Down Grasp Baseline

Question:

Can the selected 3D target drive a real simulated ManiSkill grasp attempt, and
does the first downstream metric expose a new bottleneck beyond retrieval?

Current result:

| setting | runs | failed | 3D target | grasp attempted | pick success | task success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HF compact no CLIP sim_topdown | 20 | 0 | 1.0000 | 1.0000 | 0.1000 | 0.0000 |
| HF compact with CLIP sim_topdown | 20 | 0 | 1.0000 | 1.0000 | 0.1000 | 0.0000 |

Smoke checks:

| check | result |
| --- | --- |
| oracle target seed 0 | `pick_success = true` |
| mock query smoke | completed with `grasp_attempted = true` |
| HF `red cube`, seeds 0-2 | `pick_success_rate = 1.0000` |
| placeholder regression | `placeholder_not_executed`, `grasp_attempted = false` |

Artifact:

- `outputs/h200_60071_sim_topdown_singleview_compact_seed01234/reports/sim_topdown_singleview_report.md`

Interpretation:

The low-level simulated action path is connected and benchmarkable. The oracle
and exact `red cube` smoke tests show that the controller can grasp when the
target point is suitable. The compact query result shows that broad semantic
detections often lift to a valid 3D object center that is not a reliable
top-down grasp point. The next manipulation-side code should therefore diagnose
or refine grasp-point selection before claiming that better retrieval improves
downstream grasp success.

Follow-up grasp-point diagnosis:

| group | runs | pick success | target-oracle distance | xy error | z error | high-z rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact `red cube` | 3 | 1.0000 | 0.0171 | 0.0058 | -0.0160 | 0.0000 |
| compact failures | 36 | 0.0000 | 0.3329 | 0.1864 | 0.2714 | 0.9444 |
| compact successes | 4 | 1.0000 | 0.0169 | 0.0052 | -0.0160 | 0.0000 |

Artifact:

- `outputs/h200_60071_sim_topdown_singleview_compact_seed01234/reports/grasp_point_diagnosis.md`

Interpretation:

The first compact failure mode is not primarily a controller timing issue.
Successful runs choose target points close to the oracle cube pose; failed
compact runs usually choose points high above and far away from the object. The
next behavior patch should target grasp-point refinement for broad detections,
not detector backend changes or low-level action retuning.

### Experiment 10: Multi-View Sim-Grasp Ablation Bridge

Question:

Can final multi-view and closed-loop selected objects drive the same simulated
pick executor and produce downstream grasp metrics?

Current result:

| setting | runs | failed | grasp attempted | pick success | closed-loop resolution | still-needed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact `red cube` tabletop_3 no CLIP | 3 | 0 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| compact tabletop_3 no CLIP | 20 | 0 | 1.0000 | 0.0000 | 0.0000 | 0.6500 |
| compact tabletop_3 with CLIP | 20 | 0 | 1.0000 | 0.0000 | 0.0000 | 0.6500 |
| compact closed-loop no CLIP | 20 | 0 | 1.0000 | 0.0000 | 0.6500 | 0.0000 |
| compact closed-loop with CLIP | 20 | 0 | 1.0000 | 0.0000 | 0.6500 | 0.0000 |

Artifacts:

- `outputs/h200_60071_multiview_sim_pick_bridge_targeted`
- `outputs/h200_60071_multiview_sim_pick_bridge_ablation_seed01234`

Interpretation:

The bridge is stable: multi-view and closed-loop final selections can execute
`sim_topdown` and report pick metrics with `0` child failures. The negative
compact pick result is also informative. The current fused memory exposes
`selected_object_world_xyz`, a semantic center, not the shifted-crop grasp point
that made compact single-view picks reliable. The next behavior patch should
add or propagate a grasp-specific target in fused memory rather than retuning
the controller, CLIP, detector, or re-observation policy.

### Experiment 11: Fused-Memory Grasp Point Path

Question:

Does preserving per-view refined grasp points in fused memory make multi-view
and closed-loop selected objects usable as downstream simulated grasp targets?

Current result:

| setting | runs | failed | grasp attempted | pick success | task success | target source | closed-loop resolution | still-needed |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| targeted exact `red cube` tabletop_3 no CLIP | 3 | 0 | 1.0000 | 1.0000 | 0.3333 | `memory_grasp_world_xyz` | 0.0000 | 0.0000 |
| targeted compact broad tabletop_3 no CLIP | 20 | 0 | 1.0000 | 1.0000 | 0.1500 | `memory_grasp_world_xyz` | 0.0000 | 0.6500 |
| compact tabletop_3 no CLIP | 20 | 0 | 1.0000 | 1.0000 | 0.1500 | `memory_grasp_world_xyz` | 0.0000 | 0.6500 |
| compact tabletop_3 with CLIP | 20 | 0 | 1.0000 | 1.0000 | 0.1500 | `memory_grasp_world_xyz` | 0.0000 | 0.6500 |
| compact closed-loop no CLIP | 20 | 0 | 1.0000 | 1.0000 | 0.1500 | `memory_grasp_world_xyz` | 0.6500 | 0.0000 |
| compact closed-loop with CLIP | 20 | 0 | 1.0000 | 1.0000 | 0.1500 | `memory_grasp_world_xyz` | 0.6500 | 0.0000 |

Artifacts:

- `outputs/h200_60071_multiview_memory_grasp_point_targeted`
- `outputs/h200_60071_multiview_memory_grasp_point_ablation_seed01234`

Interpretation:

The fused-memory grasp point path changes the multi-view simulated-grasp result
from a stable-but-negative bridge to a successful compact PickCube baseline.
Semantic fused centers remain unchanged for retrieval and confidence, while
`sim_topdown --grasp-target-mode refined` uses `memory_grasp_world_xyz` as the
execution target. This should be framed as strong simulated PickCube evidence,
not as robust multi-task or real-robot manipulation.

### Experiment 5: Re-Observation Decision Smoke

Question:

Can the system expose whether another viewpoint would be useful before claiming
a closed-loop re-observation policy?

Current result:

- Query: `red cube`
- Detector: mock
- View preset: `tabletop_3`
- Decision: `should_reobserve = true`
- Reason: `ambiguous_top_candidates`
- Suggested view: `left`
- Confidence gap: `0.0165`

Artifact:

- `outputs/h200_60071_reobserve_smoke/reobserve_decision.json`

### Experiment 6: Re-Observation Policy Metrics

Question:

Does the open-loop rule-based policy request extra views in the corrected HF
fusion benchmark?

Current result:

- HF tabletop_3 no-CLIP reobserve v2: `reobserve_trigger_rate = 0.0000`
- HF tabletop_3 with-CLIP reobserve v2: `reobserve_trigger_rate = 0.0000`
- Reason counts: `confident_enough: 6` for both conditions
- CLIP raises selected-object confidence from `0.5282` to `0.7091`, but does not
  change selected-object rate or policy trigger behavior.

Artifacts:

- `outputs/h200_60071_reobserve_policy_v2/outputs/reobserve_policy_report_tabletop3_hf_reobserve_v2.md`
- `outputs/h200_60071_reobserve_policy_v2/outputs/fusion_comparison_table_tabletop3_hf_reobserve_v2.md`

### Experiment 7: Ambiguity Multi-View Fusion Stress

Question:

Do ambiguity-focused queries create a setting where memory fragmentation and
re-observation policy decisions become more informative?

Current seeds 0-2 result:

| setting | runs | selected_frac | memory objects | same-label distance | selected confidence | reobserve trigger |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ambiguity tabletop_3 HF no CLIP | 33 | 1.0000 | 2.1515 | 0.1353 | 0.4985 | 0.6667 |
| Ambiguity tabletop_3 HF with CLIP | 33 | 1.0000 | 2.1515 | 0.1353 | 0.6806 | 0.4242 |

Interpretation:

Ambiguity queries make the corrected fusion setting harder: object-memory
fragmentation rises relative to exact-object tabletop_3 runs. CLIP raises
selected-object confidence and reduces re-observation triggers, but does not
reduce geometric spread or memory fragmentation. The most common triggered
reason after fixing report logic is `insufficient_view_support`.

Artifacts:

- `outputs/h200_60071_ambiguity_tabletop3_seed012/outputs/fusion_comparison_table_ambiguity_tabletop3_hf_seed012.md`
- `outputs/h200_60071_ambiguity_tabletop3_seed012/outputs/reobserve_policy_report_ambiguity_tabletop3_hf_seed012.md`

### Experiment 8: Minimal Closed-Loop Re-Observation Smoke

Question:

Can the system take a policy-triggered suggested view, rerun perception, and
report before/after target and uncertainty diagnostics?

Current H200 mock result:

| setting | initial trigger | final trigger | extra views | views before | views after | confidence before | confidence after |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| mock `object`, seed 0, tabletop_3 | 1 | 1 | `left` | 3 | 4 | 0.6015 | 0.6015 |

Interpretation:

The opt-in closed-loop path executes and records the expected artifacts:
`memory_state_before_reobserve.json`, `reobserve_decision_before.json`, final
`reobserve_decision.json`, and `closed_loop_reobserve.json`. In this mock smoke,
one extra virtual view did not resolve the ambiguity; the reason remained
`ambiguous_top_candidates`. This is useful as an infrastructure milestone, but
the paper-relevant test is the HF ambiguity benchmark with closed-loop enabled.

Artifact:

- H200: `outputs/h200_smoke_closed_loop_reobserve_mock/closed_loop_reobserve.json`

### Experiment 9: Closed-Loop Ambiguity HF Benchmark

Question:

Does one rule-suggested extra virtual view reduce policy uncertainty in the
harder ambiguity benchmark?

Current result:

| setting | runs | selected_frac | views | memory objects | selected confidence | initial trigger | final trigger | execution rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ambiguity HF no CLIP closed-loop | 33 | 1.0000 | 3.6667 | 2.1515 | 0.4985 | 0.6667 | 0.6667 | 0.6667 |
| Ambiguity HF with CLIP closed-loop | 33 | 1.0000 | 3.4242 | 2.1515 | 0.6806 | 0.4242 | 0.4242 | 0.4242 |

Interpretation:

The extra-view loop executes, but does not reduce final re-observation trigger
rate. CLIP raises selected confidence and reduces the number of triggered runs,
but the triggered cases remain triggered after one additional suggested virtual
view. This is a useful negative result: the system has a runnable closed-loop
baseline, but future improvement should target support-aware view selection or
memory update criteria rather than only adding another observation pass.

Follow-up implementation:

Closed-loop runs now emit compact delta diagnostics for selected-object changes,
selected view-support changes, selected confidence changes, memory-object
changes, policy reason changes, and resolution/still-needed rates. This gives
the next support-aware view selection milestone a measurable success criterion.

Current functional improvement:

The re-observation policy now chooses suggested views based on the triggered
failure mode, and closed-loop runs now emit initial-selected association
diagnostics. Ambiguity and insufficient-support cases prefer missing selected
support views, while geometry-driven cases prefer `top_down`-style views and
record the rationale in `suggested_view_plan`.

Compact H200 rerun with association diagnostics:

| setting | initial trigger | final trigger | resolution rate | selected assoc rate | selected support gain rate | selected object change rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ambiguity compact HF no CLIP closed-loop v2 | 1.0000 | 1.0000 | 0.0000 | 0.2500 | 0.2500 | 0.2500 |
| Ambiguity compact HF with CLIP closed-loop v2 | 0.7500 | 0.7500 | 0.0000 | 0.0000 | 0.0000 | 0.2500 |

Interpretation:

This new diagnostic narrows the failure mode. The no-CLIP compact rerun shows
that one quarter of runs now merge the extra observation back into the initial
selected object and increase its view support. The with-CLIP compact rerun does
not: association stays at `0.0` while selected-object change rate remains
`0.25`. That points the next implementation step toward memory association and
selection continuity under CLIP-weighted fusion, not toward adding more views.

Follow-up absorber trace rerun:

| setting | initial trigger | final trigger | resolution rate | initial selected absorber rate | final selected absorber rate | third-object rate | mean absorber count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ambiguity compact HF no CLIP closed-loop absorber trace | 1.0000 | 1.0000 | 0.0000 | 0.2500 | 0.5000 | 0.5000 | 1.2500 |
| Ambiguity compact HF with CLIP closed-loop absorber trace | 0.7500 | 0.7500 | 0.0000 | 0.0000 | 0.2500 | 0.5000 | 1.0000 |

Interpretation:

The new absorber trace sharpens the diagnosis. In the no-CLIP compact rerun,
extra views sometimes merge into the final selected object even when they do
not preserve the initial selection. In the with-CLIP compact rerun, half of
the executed runs involve a third object altogether, and only one quarter land
in the final selected object. The next implementation step should therefore
focus on CLIP-aware memory association or selected-object continuity, not on
adding more re-observation views.

Artifacts:

- `outputs/h200_60071_closed_loop_ambiguity_seed012/outputs/fusion_comparison_table_ambiguity_tabletop3_hf_closed_loop.md`
- `outputs/h200_60071_closed_loop_ambiguity_seed012/outputs/reobserve_policy_report_ambiguity_tabletop3_hf_closed_loop.md`
- `outputs/h200_60071_assoc_diag_ambiguity_compact_seed0_v2/reobserve_policy_report_ambiguity_compact_hf_closed_loop.md`

## Figures and Tables

Current pack:

- `outputs/paper_figure_pack_latest/README.md`

Paper assets:

1. Architecture diagram: `docs/architecture_query_to_grasp.md`.
2. Main ablation table: corrected multi-view CLIP ablation.
3. Geometry validation table: extrinsic convention comparison.
4. Qualitative example: selection trace for `red cube`.
5. Policy diagnostic: `reobserve_decision.json` example.
6. Policy diagnostic table: re-observation trigger rates and reason counts.
7. Ambiguity fusion stress table: seeds 0-2 no-CLIP vs with-CLIP.
8. Closed-loop re-observation smoke: before/after diagnostics for one suggested
   virtual view.
9. Closed-loop ambiguity benchmark: initial vs final trigger rate after one
   suggested extra virtual view.
10. Full ambiguity absorber-aware validation: 55-run no-CLIP and with-CLIP
   reports plus residual attribute-query summary.
11. Attribute residual diagnosis: trace-level classification of the remaining
   full-validation cases.
12. Attribute trace-field validation: targeted residual traces with parsed
   attributes, same-phrase competitors, and point/view support flags.
13. Simulated grasp baseline: exact/oracle `PickCube-v1` success and compact
   broad-query refined-target comparison.
14. Multi-view sim-grasp bridge: tabletop_3 and closed-loop selected-object
   pick metrics with target-source diagnostics.
15. Fused-memory grasp point ablation: compact multi-view and closed-loop
   simulated grasp success using `memory_grasp_world_xyz`.
16. Full ambiguity simulated grasp validation: 55-run `PickCube-v1`
   fused-memory tabletop_3 and closed-loop pick-success results.
17. Broader-task simulated pick validation: `StackCube-v1` query-driven
   `red cube` single-view and expanded guarded tabletop_3/closed-loop
   pick-only seeds `0..49`.
18. Oracle target-source ablation: privileged object-pose `sim_topdown`
   upper-bound rows for PickCube and StackCube.
19. Limitation box: placeholder pick, low detector multiplicity, and no real
   camera-planning or robot-control loop yet.

## Limitations

Be explicit:

- Real-robot control is not implemented.
- Low-level ManiSkill simulated control is opt-in and currently includes
  scripted pick and pick-place executors. These are benchmark controllers, not
  learned or general-purpose robot policies.
- Therefore, the current paper may report simulated pick and privileged
  pick-place baselines, but should not claim robust real-world manipulation or
  non-oracle StackCube completion until the bridge validation supports it.
- Web demo is not implemented.
- Re-observation is implemented as a minimal opt-in virtual-view loop. It now
  resolves compact ambiguity triggers in the accepted H200 diagnostic benchmark,
  but is still not learned view planning or physical camera motion.
- Full-query residuals now concentrate in attribute-style queries. Attribute
  parsing exists, but the current memory and selector diagnostics need to prove
  whether color evidence is represented strongly enough before adding new
  policy logic.
- The first residual diagnosis found mixed causes rather than one clean policy
  knob: point insufficiency, selected-view support conflict, and third-object
  absorption all appear.
- Attribute diagnostics show full attribute coverage in the residual red-object
  traces, so future changes should avoid claiming a missing-color-evidence fix
  unless new data contradicts this targeted result.
- The accepted shifted-crop refined simulated grasp target improves compact
  broad-query success from `0.1000` to `1.0000` on `PickCube-v1`, but broader
  task coverage is still needed before claiming robust manipulation.
- The accepted fused-memory grasp point path also reaches
  `pick_success_rate = 1.0000` for compact and full-ambiguity multi-view and
  closed-loop `PickCube-v1`, but this remains simulated single-task evidence.
- The `StackCube-v1` validation reaches `pick_success_rate = 1.0000` for
  query-driven single-view `red cube` seeds `0..19` in no-CLIP and with-CLIP
  modes. The expanded guarded multi-view validation over seeds `0..49` reaches
  `0.6200` for tabletop_3 and `0.5200` for closed-loop.
- The expanded StackCube failure report shows the main residual classes:
  wrong fused grasp observation dominates tabletop failures, while closed-loop
  adds third-object absorption. This should be used as the limitation figure or
  table for the multi-task section.
- Accepted query-driven StackCube pick rows remain pick-only, while the
  query-pick plus oracle-place bridge reaches positive StackCube task success
  with a privileged cubeB target. The oracle placement baseline is privileged
  and must be reported as an upper bound, not as a deployable
  language-conditioned stacker.
- The resolved simulated-grasp failures were dominated by detector boxes whose
  original upper region missed the graspable object support; the semantic target
  center is preserved while the opt-in refined grasp point can use a downward
  crop fallback.
- Experiments are still simulated: the strongest accepted benchmark is
  full-query `PickCube-v1`, while `StackCube-v1` is now the key cross-task
  bridge for testing whether query-derived cubeA targets can support placement
  when cubeB is privileged.
- CLIP did not change top-1 in current benchmarks.
- Query parser supports simple attributes and conservative relations, not full
  language reasoning.
- No real robot deployment.

## Implementation Gap Checklist

Important for a stronger v1:

- [x] Formal target selector module.
- [x] Minimal rule-based re-observation policy module.
- [x] `reobserve_decision.json` artifact in multi-view runs.
- [x] Small re-observation policy report/table generator.
- [x] Minimal opt-in closed-loop re-observation artifact path.
- [x] Closed-loop HF ambiguity benchmark with initial/final policy metrics.
- [x] Compact closed-loop delta diagnostics for benchmark summaries and reports.
- [x] Support-aware suggested-view policy with reason-specific priorities.
- [x] Accepted absorber-aware compact H200 closed-loop baseline.
- [x] Full ambiguity absorber-aware H200 validation, seeds `0..4`.
- [x] Minimal simulated ManiSkill grasp baseline from selected 3D target.
- [x] Small paper/demo architecture diagram.
- [x] README cleanup and current quickstart refresh.
- [x] Grasp-success ablation comparing single-view, multi-view, and
  closed-loop re-observation.
- [x] Fused-memory grasp point path for multi-view simulated picks.
- [ ] Optional Gradio demo shell only after paper metrics are frozen.

## Latest Simulated Grasp Result

The opt-in `sim_topdown` executor is now connected to query-driven single-view
targets. The latest accepted H200 shifted-crop refinement keeps the exact
`red cube` smoke at `3/3` pick success and improves compact simulated grasp
success to `1.0000` for both no-CLIP and with-CLIP runs. Mean compact target XY
error improves from `0.1005 m` to `0.0065 m` relative to the previous refined
grasp target, far-XY rate improves from `0.6500` to `0.0000`, and high-Z rate
remains `0.0000`.

This result should be reported as an initial simulated grasp baseline, not as
robust manipulation. The multi-view bridge ablation first showed that fused
tabletop_3 and closed-loop target sources are executable and benchmarkable. The
accepted fused-memory grasp path then replaces semantic-center pick targets
with `memory_grasp_world_xyz` for refined mode and improves compact multi-view
and closed-loop pick success from `0.0000` to `1.0000`.

The first broader-task smoke on `StackCube-v1` also runs the query-driven
single-view chain for `red cube` seeds `0..4`. Both no-CLIP and with-CLIP modes
complete `5/5` runs with `pick_success_rate = 1.0000` and
`task_success_rate = 0.0000`, validating task-specific grasp detection
(`is_cubeA_grasped`) while keeping stack placement out of scope.

The overnight full validation extends this result. On `PickCube-v1`, full
ambiguity tabletop_3 and closed-loop fused-memory grasp runs complete `55/55`
with `0` failures and `pick_success_rate = 1.0000` in both no-CLIP and with-CLIP
modes. Closed-loop still improves uncertainty diagnostics: still-needed falls
from `0.4182` to `0.0909` without CLIP and from `0.3818` to `0.0545` with CLIP.

On `StackCube-v1`, single-view `red cube` seeds `0..19` remain fully successful
for pick-only control, while the expanded guarded multi-view run reaches
`pick_success_rate = 0.6200` for tabletop_3 and `0.5200` for closed-loop across
seeds `0..49`. Closed-loop reduces still-needed uncertainty but does not improve
pick success on this task. This marks task-general re-observation-to-grasp
alignment as the next bottleneck.

The latest expanded StackCube guard result shows that this bottleneck is
task-dependent rather than a global failure of fused-memory grasp points. For
`StackCube-v1`, using the selected object's semantic fused center as the
effective refined multi-view pick target reaches `pick_success_rate = 0.6200`
for tabletop_3 seeds `0..49` in both no-CLIP and with-CLIP modes. Closed-loop
reaches `0.5200`, so the guard improves the static multi-view target source but
does not yet convert re-observation into better StackCube grasp success. The
expanded estimate is slightly lower than the compact 20-seed checkpoint, but it
is stronger paper evidence because it covers 50 seeds with `0` child failures.
A PickCube regression remains `3/3` successful and continues to use
`memory_grasp_world_xyz`. This should be reported as a StackCube pick-only
compatibility guard, not as stack-placement success.

Latest oracle target-source ablation:

| benchmark | env | runs | failed | pick success | task success | target source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Oracle pick | `PickCube-v1` | 50 | 0 | 1.0000 | 0.0400 | `oracle_object_pose` |
| Oracle pick | `StackCube-v1` | 50 | 0 | 0.9400 | 0.0000 | `oracle_object_pose` |

The oracle rows are an upper-bound diagnostic for the same scripted controller,
not a deployable perception result. PickCube oracle matches the query-driven
fused-memory result, while StackCube oracle remains much higher than guarded
multi-view query-driven picking (`0.6200` tabletop, `0.5200` closed-loop). This
supports the current interpretation that StackCube residuals are primarily
target-source and association limitations rather than an absolute inability of
the controller to pick cubeA.

Latest oracle placement baseline:

| benchmark | env | runs | failed | pick success | place success | task success | target source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Oracle pick-place | `StackCube-v1` | 50 | 0 | 0.9400 | 0.8800 | 0.8800 | `oracle_cubeA_pose` -> `oracle_cubeB_pose` |

This is the first positive StackCube placement-capability result, but it is
privileged. It should be framed as an upper-bound controller/target-source
ablation, not as query-driven StackCube task completion.

Accepted query-pick plus oracle-place bridge:

| benchmark | env | runs | failed | pick success | place success | task success | place target |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Single-view no CLIP | `StackCube-v1` | 50 | passed by H200 checker | 0.8800 | 0.7200 | 0.7200 | `oracle_cubeB_pose` |
| Single-view with CLIP | `StackCube-v1` | 50 | 0 | 0.8800 | 0.7200 | 0.7200 | `oracle_cubeB_pose` |
| tabletop_3 no CLIP | `StackCube-v1` | 50 | 0 | 0.6200 | 0.5200 | 0.5200 | `oracle_cubeB_pose` |
| tabletop_3 with CLIP | `StackCube-v1` | 50 | 0 | 0.6200 | 0.5200 | 0.5200 | `oracle_cubeB_pose` |
| closed-loop no CLIP | `StackCube-v1` | 50 | 0 | 0.5200 | 0.4800 | 0.4800 | `oracle_cubeB_pose` |
| closed-loop with CLIP | `StackCube-v1` | 50 | 0 | 0.5200 | 0.4800 | 0.4800 | `oracle_cubeB_pose` |

This bridge upgrades StackCube from pick-only compatibility to a partial
task-success result: the pick target is query-derived, while the placement
target remains privileged. It supports the retrieval-to-execution-gap thesis
without claiming fully non-oracle language-conditioned stacking.

## Next Writing Milestone

Turn the polished Markdown draft into a venue-shaped manuscript without changing
detector, fusion weights, controller timing, or benchmark claims:

1. Freeze the cross-task simulated pick/place table with PickCube full/compact
   rows, StackCube expanded guarded rows, oracle pick/place rows, and bridge
   rows.
2. Use the expanded failure report as the limitation/failure figure showing
   that StackCube closed-loop still reaches only `0.5200` pick success despite
   reduced uncertainty diagnostics.
3. Preserve the target-source distinction: PickCube refined uses
   `memory_grasp_world_xyz`; StackCube refined uses
   `task_guard_selected_object_world_xyz`.
4. Keep privileged placement sources clearly labeled unless a non-oracle
   placement target path is implemented and validated.
5. Multi-task simulated-grasp prose is drafted in
   `docs/paper_multitask_sim_grasp_section.md`, and the first full manuscript
   skeleton is assembled in `docs/paper_manuscript_draft.md`.
6. Manuscript draft v0.2 now includes a related-work scaffold, figure/table
   callouts, and conference-style result framing; the citation buckets are
   tracked in `docs/paper_related_work_citation_plan.md`.
7. IEEEtran conference-style LaTeX draft v0.1 now lives in `paper/main.tex`,
   with the provided `IEEEtran.cls`, first-pass BibTeX in
   `paper/references.bib`, and structural checks in
   `scripts/check_paper_latex.py`.
8. The LaTeX draft now uses the diagnostic-systems title and four-layer
   narrative. Captions and the reproducibility appendix have been compressed
   for page budget, and core BibTeX entries have been cleaned. Next writing
   task: run a true PDF compile once a TeX toolchain is available, then polish
   page breaks and optional related-work additions.
9. The IEEE author block now uses Zhuo Chen as the sole author with Chalmers
   affiliation and no funding footnote. Detailed reproducibility commands live
   in `paper/README.md`, while the main paper only references the supplemental
   video at a high level.

## Video/Figure Evidence Plan

The core system and benchmark claims are now frozen for the IROS/ICRA simulated
full-paper v1. The next development work should therefore package existing
evidence, not tune detector, CLIP, fusion, re-observation, or controller
behavior.

`scripts/build_demo_video_pack.py` creates the current video-planning pack at
`outputs/demo_video_pack_latest`. It reads accepted benchmark summaries and
rows, selects representative success/failure seeds, indexes any already-pulled
image/video artifacts, and emits `capture_requests.json` when a story still
needs a small demo recapture.

The required supplemental-video stories are:

| story | purpose | claim boundary |
| --- | --- | --- |
| PickCube full-query success | Show `memory_grasp_world_xyz` as an executable simulated pick target. | Simulated pick only, not real robot. |
| StackCube query-pick plus oracle-place success | Show query-derived cubeA pick target driving a privileged-place task bridge. | Placement target remains oracle. |
| StackCube tabletop success/failure contrast | Show task-source quality as the cross-task bottleneck. | Diagnostic contrast, not a new benchmark. |
| StackCube closed-loop limitation | Show that re-observation diagnostics do not guarantee higher task success. | Closed-loop is diagnostic, not claimed as universally beneficial. |

Generated video-pack files are supplemental artifacts and should remain
untracked under `outputs/`. If local media are missing, only recapture the
listed representative seeds; do not rerun 50-seed benchmarks for video.

The frozen conference supplemental video is built from native ManiSkill RGB
execution captures: the demo recapture requests `sensor_configs` with
`width = 720` and `height = 720`, then assembles a `1920x1080` video from
`outputs/demo_video_pack_latest/manifest.json` into
`outputs/supplemental_video_latest`. It is a presentation artifact, not a new
experiment: captions explicitly name `memory_grasp_world_xyz` for the PickCube
executable-target result, `oracle_cubeB_pose` for the StackCube query-pick plus
oracle-place bridge, and the closed-loop StackCube clips as a diagnostic
limitation rather than a universal manipulation improvement.

The paper now includes a tracked execution-evidence montage,
`paper/figures/execution_evidence_montage.pdf`, built from continuous native
720p ManiSkill clips. The montage is the first paper figure and uses verified
success/failure metadata; the StackCube tabletop and closed-loop failure panels
come from true `place_not_confirmed` recaptures rather than mislabeled
successful clips.

## Submission Readiness Freeze

Core functionality is frozen for the simulated IROS/ICRA v1. The submission
package should now be checked with `scripts/audit_paper_submission_package.py`,
which regenerates the final main results table and verifies required artifacts,
key accepted metrics, and unsupported-claim boundaries. Any future technical
milestone such as non-oracle placement targets should be treated as a stretch
branch, not as a requirement for the current paper freeze. The current
paper-preparation focus is Overleaf compilation, page-budget editing, final
caption polish, and camera-ready citation cleanup.
