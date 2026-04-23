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

What we should not claim yet:

- Real grasp execution success.
- General cluttered-scene manipulation.
- Learned re-observation.
- CLIP as the main source of retrieval improvement.
- Robust relation-heavy language grounding.

## Abstract Skeleton

Language-conditioned robotic manipulation requires connecting open-vocabulary
2D perception with 3D geometric reasoning and action execution. We present
Query-to-Grasp, a modular research prototype for language-queryable target
retrieval in ManiSkill. The system parses a natural-language query, detects
candidate 2D regions with GroundingDINO, optionally reranks them with CLIP, lifts
RGB-D detections into 3D, and fuses multi-view evidence into a persistent object
memory. A safe placeholder executor validates selected targets without claiming
unverified robot-control success.

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
- H200: `outputs/h200_60071_selected_continuity_ambiguity_compact_seed0/reobserve_policy_report_selected_continuity.md`

Current closed-loop diagnostic:

- A selected-object continuity rule now prefers merging extra-view observations
  back into the initially selected memory object when geometry is compatible.
- In compact HF ambiguity reruns, this improves selected-object association from
  `0.2500` to `0.5000` in the no-CLIP setting and from `0.0000` to `0.2500` in
  the with-CLIP setting.
- However, closed-loop resolution remains `0.0`, so the current next bottleneck
  is not merely executing extra views but making those extra observations reduce
  policy uncertainty.

### 7. Placeholder Pick Execution

Implemented in `src/manipulation/pick_executor.py`.

Current behavior:

- Validates selected target coordinates.
- Returns structured result.
- Does not send low-level robot actions.

Paper framing:

Use `pick_success_rate = 0.0` as expected placeholder behavior, not as a failed
real grasp metric.

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
- Pool size: `2`
- Selected object has stronger view support and confidence.

Artifact:

- `outputs/h200_60071_selection_trace_red_cube_seed0/selection_trace.md`

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
10. Limitation box: placeholder pick, low detector multiplicity, and no real
   camera-planning or robot-control loop yet.

## Limitations

Be explicit:

- Real low-level ManiSkill robot control is not implemented.
- Web demo is not implemented.
- Re-observation is implemented only as a minimal opt-in virtual-view loop; it
  does not yet reduce ambiguity-trigger rates in the HF ambiguity benchmark and
  is not learned view planning or physical camera motion.
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
- [x] Small paper/demo architecture diagram.
- [x] README cleanup and current quickstart refresh.
- [ ] Optional Gradio demo shell.
- [ ] Optional real ManiSkill scripted pick only after control API is verified.

## Next Coding Milestone

Improve selected-object continuity in closed-loop ambiguity runs:

1. Add a compact continuity rule or diagnostic that explicitly prefers merging
   extra-view observations back into the currently selected object when the
   geometry remains compatible.
2. Inspect why the HF with-CLIP compact rerun still sends half of its extra
   views into a third object.
3. Only build demo UI after the closed-loop perception result improves a
   measurable policy or memory metric.
