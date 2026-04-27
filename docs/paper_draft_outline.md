# Paper Draft Outline

Working title:

`Query-to-Grasp via Confidence-Aware 3D Semantic Fusion`

Shorter title option:

`Language-Queryable 3D Target Retrieval with Confidence-Aware Multi-View Fusion`

## Current Claim

This project currently supports a focused systems claim:

> A language-queryable RGB-D perception pipeline can retrieve open-vocabulary
> 3D target hypotheses in ManiSkill, and corrected multi-view geometric fusion
> substantially reduces object-memory fragmentation compared with an uncorrected
> or single-view-only baseline.

Updated paper positioning:

- The near-term paper is a target-retrieval and active re-observation paper for
  grasp preparation, with an initial opt-in simulated grasp baseline.
- The current placeholder pick path is valid infrastructure, but it should be
  framed as target validation rather than robot-control evidence.
- The simulated top-down executor now reports downstream grasp outcomes
  separately from retrieval outcomes; the first compact result is diagnostic
  rather than a mature manipulation benchmark.

What we should not claim yet:

- Real grasp execution success.
- General cluttered-scene manipulation.
- Learned re-observation.
- CLIP as the main source of retrieval improvement.
- Robust relation-heavy language grounding.

Submission-level expectation:

- Current retrieval/re-observation version: suitable for an arXiv systems paper,
  workshop paper, or perception-for-manipulation diagnostic submission.
- With a reliable minimal simulated grasp baseline and grasp-success ablations:
  substantially stronger for ICRA/IROS workshop or a possible conference paper
  if the experimental story is clean.
- With robust simulated grasp control, stronger baselines, and broader tasks:
  closer to RA-L/ICRA/IROS full-paper expectations.
- With real-robot validation: materially higher ceiling, but that is a new
  project phase rather than a small extension.

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
14. Limitation box: placeholder pick, low detector multiplicity, and no real
   camera-planning or robot-control loop yet.

## Limitations

Be explicit:

- Real-robot control is not implemented.
- Low-level ManiSkill simulated control is opt-in and currently limited to a
  simple top-down executor for `PickCube-v1`.
- Therefore, the current paper may report an initial simulated grasp baseline,
  but should not claim robust end-to-end manipulation or real grasp success.
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
- The accepted refined simulated grasp target improves compact broad-query
  success from `0.1000` to `0.3500`, but downstream manipulation remains a
  bottleneck rather than a solved contribution.
- Remaining simulated-grasp failures are dominated by lateral target error in
  broad detections, not by target height or TCP tracking to the requested point.
- Experiments are small and use `PickCube-v1` plus virtual camera poses.
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
- [ ] Grasp-success ablation comparing single-view, multi-view, and
  closed-loop re-observation.
- [ ] Optional Gradio demo shell only after paper metrics are frozen.

## Latest Simulated Grasp Result

The opt-in `sim_topdown` executor is now connected to query-driven single-view
targets. The latest accepted H200 refinement keeps the exact `red cube` smoke
at `3/3` pick success and improves compact simulated grasp success to `0.3500`
for both no-CLIP and with-CLIP runs. Mean compact target XY error improves from
`0.1239 m` to `0.1005 m`, far-XY rate improves from `0.9000` to `0.6500`, and
high-Z rate remains `0.0000`.

This result should be reported as an initial simulated grasp baseline, not as
robust manipulation. The next experiment is component-aware grasp-point
selection inside broad detections.

## Next Coding Milestone

Improve the new simulated grasp baseline without changing detector/fusion
backends:

1. Add one component-aware grasp-point refinement for broad detections whose
   elevated workspace points still span multiple object-like clusters.
2. Validate it on targeted compact simulated-grasp cases before touching
   detector, fusion, or re-observation logic.
3. Rerun the compact simulated grasp benchmark only after the component-aware
   target-point refinement has a clear targeted smoke result.
4. Keep detector backends, fusion weights, training, web demo, and real-robot
   deployment out of scope for the grasp-baseline phase.
