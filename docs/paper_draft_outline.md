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
- Pool size: `2`
- Selected object has stronger view support and confidence.

Artifact:

- `outputs/h200_60071_selection_trace_red_cube_seed0/selection_trace.md`

## Figures and Tables

Current pack:

- `outputs/paper_figure_pack_latest/README.md`

Planned paper assets:

1. Architecture diagram: query -> detector -> CLIP optional -> RGB-D lifting ->
   object memory -> selector -> placeholder pick.
2. Main ablation table: corrected multi-view CLIP ablation.
3. Geometry validation table: extrinsic convention comparison.
4. Qualitative example: selection trace for `red cube`.
5. Limitation box: placeholder pick and low detector multiplicity.

## Limitations

Be explicit:

- Real low-level ManiSkill robot control is not implemented.
- Web demo is not implemented.
- Re-observation policy is not implemented as a closed loop.
- Experiments are small and use `PickCube-v1` plus virtual camera poses.
- CLIP did not change top-1 in current benchmarks.
- Query parser supports simple attributes and conservative relations, not full
  language reasoning.
- No real robot deployment.

## Implementation Gap Checklist

Important for a stronger v1:

- [x] Formal target selector module.
- [ ] Minimal rule-based re-observation policy module.
- [ ] `reobserve_decision.json` artifact in multi-view runs.
- [ ] Small paper/demo architecture diagram.
- [ ] README cleanup and current quickstart refresh.
- [ ] Optional Gradio demo shell.
- [ ] Optional real ManiSkill scripted pick only after control API is verified.

## Next Coding Milestone

Implement `src/policy/reobserve_policy.py` as a minimal, testable rule-based
module.

Initial inputs:

- Selected object confidence.
- Top-1 vs top-2 confidence gap.
- Minimum view count.
- Geometry confidence.
- Number of 3D points.

Initial output:

```python
{
  "should_reobserve": bool,
  "reason": str,
  "suggested_view_ids": list[str],
  "diagnostics": dict,
}
```

Keep this policy separate from automatically moving cameras at first. The first
milestone should only produce a decision artifact that can be benchmarked and
reported.
