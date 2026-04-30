# Query-to-Grasp: Language-Queryable RGB-D Retrieval, Multi-View Memory, and Simulated Pick Execution in ManiSkill

## Abstract

Language-conditioned robotic manipulation requires connecting open-vocabulary
2D perception with 3D geometric reasoning and executable action targets. We
present Query-to-Grasp, a modular ManiSkill research prototype for
language-queryable RGB-D target retrieval and simulated pick evaluation. The
system parses a natural-language query, detects candidate 2D regions with
GroundingDINO, optionally reranks candidates with CLIP, lifts RGB-D detections
into 3D, fuses multi-view evidence into persistent object memory, selects a
query-matching 3D target, and optionally executes a simulated top-down pick. A
safe placeholder executor remains the default, while an opt-in ManiSkill
controller reports downstream `pick_success` separately from task-level
`task_success`.

Our experiments show that the current bottleneck is geometric target quality,
not CLIP reranking: detector candidate multiplicity is low, and CLIP does not
change top-1 selections in the validated benchmarks. Correcting the RGB-D camera
frame convention sharply improves multi-view consistency, and fused memory
grasp points lift `PickCube-v1` full-query multi-view and closed-loop simulated
pick success to `1.0000`. A broader `StackCube-v1` pick-only validation reaches
`0.6200` tabletop and `0.5200` closed-loop pick success over 50 seeds, while
revealing task-dependent target-source preferences and closed-loop
third-object absorption. These results support Query-to-Grasp as a reproducible
diagnostic baseline for language-conditioned RGB-D retrieval and simulated
grasp target evaluation, while clearly separating perception, pick-only control,
and full task completion.

## 1. Introduction

Open-vocabulary object detectors make it increasingly easy to localize objects
from language in a single RGB image. Robotic manipulation, however, needs more
than a 2D box. A useful language-conditioned manipulation pipeline must decide
which 3D object instance is being referred to, determine whether the evidence is
geometrically consistent, expose uncertainty when observations are ambiguous,
and produce an action target that can drive a controller. Query-to-Grasp is a
systems-oriented prototype for studying this full chain in ManiSkill.

The project focuses on measurable RGB-D retrieval and grasp-target diagnostics
rather than learned grasp synthesis or real-robot deployment. Its default path
exports a structured placeholder pick result so that perception and target
selection can be benchmarked without pretending to solve low-level control. For
simulated manipulation evidence, the system provides an opt-in top-down
controller that executes real ManiSkill `pd_ee_delta_pos` actions and reports
grasp/lift success separately from environment task success.

This distinction is central to the paper. On `PickCube-v1`, selected 3D targets
can now drive successful simulated picks across compact and full ambiguity
query sets. On `StackCube-v1`, the same pipeline validates pick-only transfer
for cubeA, but does not perform the placement phase needed for stacking task
completion. We therefore report `pick_success` and `task_success` as separate
metrics and avoid claiming full end-to-end manipulation or real-robot success.

The paper makes three practical contributions:

1. A modular language-queryable RGB-D retrieval pipeline with open-vocabulary
   proposals, optional reranking, 2D-to-3D lifting, multi-view object memory,
   deterministic target selection, and structured per-run artifacts.
2. Diagnostic multi-view and closed-loop re-observation baselines that expose
   confidence, view support, memory fragmentation, selected-object continuity,
   and residual failure causes.
3. An opt-in simulated pick evaluation path that connects selected 3D targets
   to low-level ManiSkill actions and demonstrates strong `PickCube-v1` pick
   success while identifying cross-task limitations on `StackCube-v1`.

## 2. Related Work

### 2.1 Open-Vocabulary Grounding

Open-vocabulary image grounding methods make it possible to retrieve 2D regions
from category names, attributes, and free-form prompts without training a
closed-set detector for every manipulation task. Query-to-Grasp uses this family
of models as an off-the-shelf proposal layer: HF GroundingDINO supplies query
conditioned 2D detections, and CLIP can optionally rerank candidate crops. The
paper does not claim a new grounding model. Instead, it studies what happens
after an open-vocabulary 2D proposal is available: whether RGB-D lifting,
multi-view fusion, uncertainty diagnostics, and simulated control preserve a
graspable target. Table 1 should use the paper-pack ablation artifact to show
that the current bottleneck is not CLIP reranking, because candidate
multiplicity is low and top-1 changes are rare in the validated benchmarks.

### 2.2 Language-Conditioned Manipulation

Language-conditioned manipulation systems connect natural-language task
descriptions to actions through perception, planning, learned policies, or
large-model reasoning. Query-to-Grasp occupies a narrower but practical part of
this space. It does not learn a policy, synthesize grasps from language, or
perform long-horizon reasoning. It asks whether a simple query such as
`red cube` can be converted into a stable 3D target hypothesis and then into an
honest simulated pick attempt. This makes the system complementary to broader
language-to-action frameworks: it provides a reproducible diagnostic baseline
for the intermediate target-retrieval layer that those systems often depend on.

### 2.3 RGB-D and Multi-View Object Memory

RGB-D manipulation pipelines commonly rely on geometric lifting, point clouds,
multi-view fusion, or object-centric maps to move from image evidence to
actionable 3D state. Query-to-Grasp follows this object-memory framing but keeps
the memory deliberately small and inspectable. Each memory object stores label
votes, view support, confidence terms, geometry diagnostics, and a separate
grasp point when the simulated pick path is enabled. The central systems lesson
is that this memory is only meaningful when camera conventions are handled
correctly. Fig. 2 should use the geometry artifacts in the paper pack to show
that the OpenCV-to-OpenGL correction reduces cross-view spread enough for
deterministic memory fusion to become credible.

### 2.4 Active and Re-Observation Diagnostics

Active perception and next-best-view methods use additional observations to
reduce uncertainty before acting. Query-to-Grasp implements a diagnostic version
of this idea rather than a learned view planner: a rule-based policy requests a
virtual extra view when confidence, view support, geometry, or point-count
signals are weak. This design makes uncertainty visible in benchmark rows and
reports, and it exposes a useful caution for manipulation. Closed-loop
re-observation can reduce the final uncertainty signal, but StackCube shows that
lower uncertainty does not automatically imply better grasp execution. Table 5
or the limitation figure should therefore use the expanded StackCube failure
report to separate third-object absorption, wrong fused grasp observations, and
controller/contact residuals.

## 3. Method

Query-to-Grasp decomposes language-conditioned grasp target retrieval into
small, inspectable modules. A rule-based parser extracts a normalized prompt,
target name, attributes, synonyms, and conservative relation fields from the
query. GroundingDINO provides open-vocabulary 2D proposals, with a mock backend
preserved for dependency-light smoke tests. CLIP reranking can be enabled, but
current results show that it rarely changes top-1 because candidate sets are
small.

RGB-D detections are lifted into world coordinates using camera intrinsics,
depth, and ManiSkill camera poses. A key implementation correction is the
OpenCV-to-OpenGL camera convention before applying ManiSkill `cam2world_gl`.
This correction is important empirically: same-label cross-view spread drops
from `1.0693 m` to `0.0518 m`, and memory fragmentation drops from `3.3333` to
`1.3333` objects per run in the validated geometry benchmark.
Fig. 1 should show the implemented pipeline from `docs/architecture_query_to_grasp.md`;
Fig. 2 should show the camera-convention and cross-view geometry validation.

The multi-view path captures virtual tabletop views, lifts per-view detections
to 3D, and merges them into object memory. Each memory object stores semantic
labels, confidence terms, view support, geometry diagnostics, and, for the
simulated grasp path, a separately fused grasp point. Target selection remains
semantic: it chooses a query-matching memory object using label evidence,
confidence, view support, geometry confidence, and deterministic tie-breaks.
The selected semantic object center is kept distinct from any downstream grasp
point.

Re-observation is implemented as a diagnostic virtual-view loop. A rule-based
policy triggers on low confidence, small confidence gaps, insufficient view
support, geometry issues, or too few 3D points. The closed-loop path can capture
one suggested extra view, merge it back into memory, and export before/after
diagnostics. This is not learned camera planning or physical robot camera
motion; it is an instrumented way to test whether another view resolves
uncertainty without destabilizing target selection.

For action execution, Query-to-Grasp keeps `SafePlaceholderPickExecutor` as the
default. The opt-in `SimulatedTopDownPickExecutor` moves the TCP above the
selected target, descends, closes the gripper, and lifts. It reports
`grasp_attempted`, `pick_success`, `task_success`, grasp state, stage, final TCP
position, and raw simulator info. `pick_success` measures task-specific
grasp/lift confirmation, while `task_success` preserves the full ManiSkill task
success flag.

## 4. Experimental Setup

The main perception and simulated-control validations run on H200 using the HF
GroundingDINO backend. Local smoke tests use the mock detector and skip-CLIP
path when heavy dependencies are unnecessary. The primary environment is
`PickCube-v1`, where the target is a cube that can be picked by the top-down
executor. `StackCube-v1` is used as a broader-task compatibility diagnostic:
the system attempts to pick cubeA from the query `red cube`, but it does not
perform stack placement.

The benchmark suite reports both retrieval and control metrics. Retrieval
metrics include detection count, 3D target availability, memory size, selected
object confidence, re-observation trigger rate, closed-loop resolution rate,
and still-needed rate. Control metrics include grasp attempted rate,
`pick_success_rate`, `task_success_rate`, and pick-stage counts. This separation
prevents a successful retrieval or grasp from being confused with full task
completion.

Key artifact sources are:

- Single-view and ambiguity baselines:
  `outputs/h200_60071_paper_baseline/outputs`.
- Full PickCube simulated grasp comparison:
  `outputs/h200_60071_multiview_memory_grasp_point_full_ambiguity_seed01234`.
- Expanded StackCube guard validation:
  `outputs/h200_60071_stackcube_task_guard_expanded_seed0_49`.
- StackCube failure taxonomy:
  `outputs/h200_60071_stackcube_task_guard_expanded_seed0_49/reports/stackcube_guard_failure_report.md`.
- Paper pack:
  `outputs/paper_figure_pack_latest`.

The main quantitative callouts are organized as follows. Table 1 reports the
single-view and ambiguity detector/rerank baseline. Table 2 reports the
geometry and corrected fusion evidence. Table 3 reports PickCube full-ambiguity
simulated grasp metrics. Table 4 reports the cross-task PickCube/StackCube pick
comparison. Fig. 3 or Table 5 reports the expanded StackCube failure taxonomy.

## 5. Results

### 5.1 Single-View Retrieval and CLIP Reranking

The single-view HF baselines establish that the retrieval chain is stable before
multi-view fusion or simulated control is introduced. In exact-object runs,
both no-CLIP and with-CLIP modes detect one candidate per query on average and
produce a 3D target in every run. The ambiguity benchmark raises candidate
multiplicity to `1.4242`, but CLIP still does not change top-1 in the current
result set. Table 1 should use
`tables/single_view_and_ambiguity_ablation.md` from the paper pack. The main
interpretation is conservative: CLIP is useful infrastructure and a confidence
source, but the validated results do not support presenting CLIP as the primary
improvement mechanism.

### 5.2 Geometry Correction Enables Multi-View Memory

The multi-view geometry correction is the first pivotal systems result. Before
the camera-frame fix, same-label observations from different views could be
more than a meter apart in memory. After applying the OpenCV-to-OpenGL
conversion before ManiSkill `cam2world_gl`, same-label cross-view spread drops
from `1.0693 m` to `0.0518 m`, and memory fragmentation drops from `3.3333` to
`1.3333` objects per run. Table 2 should combine
`geometry/extrinsic_convention_report.md`,
`geometry/corrected_cross_view_geometry.md`, and the corrected fusion table in
the paper pack.

This result changes the nature of the benchmark. Without the convention fix,
multi-view memory is mostly an artifact of inconsistent coordinates. With the
fix, deterministic fusion, label voting, view support, and selection traces
become meaningful enough to diagnose. The memory still fails in some harder
cases, but those failures are now attributable to object support, target-source
choice, or re-observation association rather than a basic frame-convention bug.

### 5.3 Closed-Loop Re-Observation Diagnostics

Closed-loop re-observation improves uncertainty diagnostics in the accepted
absorber-aware policy path. On compact ambiguity seeds, the accepted policy
reaches `closed_loop_resolution_rate = 0.6500` and
`closed_loop_still_needed_rate = 0.0000` while keeping third-object involvement
within the accepted gate. On the full ambiguity file, both no-CLIP and with-CLIP
complete `55/55` runs with zero child failures; residual triggers concentrate
in attribute-style queries such as `red block` and `red cube`.

The result is useful because it is both positive and diagnostic. The extra-view
path can resolve compact uncertainty, but the full-query residuals show that
re-observation is not a universal cure. Attribute trace diagnostics found that
residual red-object cases already had `attribute_coverage = 1.0`; the remaining
limitations are same-phrase memory fragmentation and point/view support, not
missing parsed color evidence.

### 5.4 Fused Grasp Targets Enable PickCube Simulated Picking

The strongest downstream-control result is the transition from semantic fused
centers to fused grasp points. The initial multi-view simulated-pick bridge
showed that selected memory objects could drive the controller and emit stable
metrics, but compact pick success was `0.0000` because semantic centers were
not reliable top-down grasp targets. Propagating per-view refined grasp points
into fused memory changes the execution target to `memory_grasp_world_xyz`
while keeping semantic centers unchanged for retrieval and confidence.

Table 3 should use
`grasp/full_ambiguity_grasp_comparison.md` from the paper pack. In the full
PickCube ambiguity validation, tabletop_3 and closed-loop modes complete
`55/55` runs with `0` child failures and `pick_success_rate = 1.0000` in both
no-CLIP and with-CLIP settings. This is the central positive manipulation
evidence of the paper: open-vocabulary RGB-D target retrieval, corrected
multi-view memory, and fused grasp targets can produce simulated picks from
query-selected objects.

### 5.5 StackCube Exposes Cross-Task Limits

StackCube provides the main limitation result. Single-view `red cube` pick-only
execution succeeds, but multi-view StackCube prefers a different target source
than PickCube. PickCube refined multi-view picks should use
`memory_grasp_world_xyz`; StackCube refined multi-view picks are more reliable
when guarded to use `task_guard_selected_object_world_xyz`. Table 4 should use
`tables/stackcube_task_guard_expanded_cross_task_table.md` to show this
cross-task distinction.

The expanded StackCube validation covers seeds `0..49` and reaches
`pick_success_rate = 0.6200` for tabletop_3 and `0.5200` for closed-loop in both
no-CLIP and with-CLIP modes, with `0` child failures. `task_success_rate`
remains `0.0000`, which is expected because the executor picks and lifts cubeA
but does not stack it on cubeB. Fig. 3 or Table 5 should use
`grasp/stackcube_task_guard_expanded_failure_report.md` to show the dominant
residual classes: wrong fused grasp observation overall and third-object
absorption in closed-loop runs. This makes StackCube a cross-task pick-only
compatibility diagnostic, not a completed stacking result.

## 6. Multi-Task Simulated Grasp Evaluation

Retrieval accuracy alone is not enough for a language-queryable manipulation
pipeline. A detector can identify the right object category, and a multi-view
memory can select a plausible 3D object hypothesis, while the resulting point is
still unsuitable as a grasp target. We therefore evaluate the final selected
3D target with `sim_topdown`, an opt-in ManiSkill controller that attempts a
simple top-down grasp and lift from the selected world-coordinate target.

The strongest manipulation result is on `PickCube-v1`. After propagating
per-view grasp candidates into fused 3D memory, refined multi-view execution
uses `memory_grasp_world_xyz` as the downstream pick target while preserving the
semantic fused object center for retrieval and selection. In the full ambiguity
validation, both tabletop_3 and closed-loop multi-view modes complete `55/55`
runs with `0` child failures and `pick_success_rate = 1.0000` in no-CLIP and
with-CLIP settings. Closed-loop still improves uncertainty diagnostics:
still-needed falls from `0.4182` to `0.0909` without CLIP and from `0.3818` to
`0.0545` with CLIP, while simulated pick success remains perfect.

We also test whether the same query-to-grasp chain transfers to `StackCube-v1`
with the query `red cube`. This experiment is deliberately framed as pick-only
compatibility rather than stacking-task completion. The single-view chain
reaches `pick_success_rate = 1.0000` for seeds `0..19`, but multi-view
execution exposes a task-dependent target-source preference. On `PickCube-v1`,
refined multi-view picking benefits from the fused memory grasp point. On
`StackCube-v1`, the semantic fused selected-object center is more reliable. The
accepted task-aware guard therefore uses `task_guard_selected_object_world_xyz`
only for `StackCube-v1`, leaving `PickCube-v1` refined picks on
`memory_grasp_world_xyz`.

The expanded StackCube validation covers seeds `0..49` for tabletop_3 and
closed-loop modes, in no-CLIP and with-CLIP settings. All four StackCube modes
complete `50/50` runs with `0` child failures and `grasp_attempted_rate =
1.0000`. Tabletop_3 reaches `pick_success_rate = 0.6200` in both no-CLIP and
with-CLIP modes. Closed-loop reaches `pick_success_rate = 0.5200` in both modes.
The PickCube regression remains `3/3` successful and continues to use
`memory_grasp_world_xyz`, confirming that the StackCube guard does not weaken
the main PickCube result.

The remaining StackCube failures clarify the limitation of the current system.
The expanded failure report classifies `86` failed grasps across the four
StackCube modes. The dominant aggregate class is
`wrong_fused_grasp_observation` (`44/86` failures), indicating that the selected
object can still contain grasp evidence that is spatially inconsistent with the
effective cubeA pick target. Static tabletop_3 failures are mostly wrong fused
grasp observations plus memory fragmentation or low support. Closed-loop adds a
different failure mode: `third_object_absorption` accounts for `21` closed-loop
failures. Thus, re-observation can reduce uncertainty triggers while still
associating the extra view with an object that is not the final graspable cubeA
target.

Taken together, these experiments show that the pipeline has moved beyond
perception-only retrieval: language-selected RGB-D targets can drive a real
simulated controller and produce stable downstream grasp metrics. The evidence
is strongest for `PickCube-v1`, where full-query multi-view and closed-loop
simulated pick success reaches `1.0000`. `StackCube-v1` provides a stricter
cross-task diagnostic: it validates pick-only transfer, exposes task-dependent
grasp target preferences, and motivates future work on task-general fused grasp
target aggregation and closed-loop association.

## 7. Limitations

The current system is simulated and should not be presented as a real-robot
deployment. It does not include real camera motion, real gripper control,
hardware safety, or sim-to-real transfer. The web demo and training code are
also out of scope for the current version.

The low-level controller is intentionally simple. It is useful as a stable
diagnostic baseline because it separates graspable target quality from
perception-only retrieval, but it is not a general manipulation controller.
For `StackCube-v1`, `task_success_rate = 0.0000` is expected: the controller
can pick/lift cubeA but does not place it on cubeB.

The language interface is also conservative. Query parsing handles simple
targets, attributes, synonyms, and limited relation fields. Current results do
not show CLIP reranking as a main improvement source because detector candidate
sets are often too small for reranking to matter. Future versions should test
richer language, more clutter, and stronger proposal multiplicity before
claiming broad language understanding.

Finally, closed-loop re-observation is a diagnostic virtual-view path rather
than learned active perception. It can reduce uncertainty and improve some
selection diagnostics, but StackCube shows that lower uncertainty is not always
the same as better grasp execution. The next technical step is task-general
association between re-observed evidence, fused memory, and grasp target
quality.

## 8. Conclusion

Query-to-Grasp provides a reproducible baseline for language-queryable RGB-D
target retrieval and simulated grasp target evaluation in ManiSkill. The system
connects open-vocabulary proposals, RGB-D lifting, multi-view memory,
confidence-aware target selection, re-observation diagnostics, and opt-in
simulated pick execution. The strongest result is full-query `PickCube-v1`
multi-view and closed-loop simulated pick success of `1.0000`. The broader
StackCube validation shows pick-only transfer while exposing target-source and
closed-loop association limitations. These findings make Query-to-Grasp a
useful platform for studying the gap between semantic target retrieval and
graspable 3D action targets.

## Appendix: Reproducibility Artifacts

The recommended paper artifact pack is generated with:

```powershell
$env:PYTHONPATH=(Get-Location).Path
python scripts/build_paper_figure_pack.py --output-dir outputs/paper_figure_pack_latest --skip-missing
```

Important source artifacts include:

- `docs/architecture_query_to_grasp.md`
- `docs/paper_draft_outline.md`
- `docs/paper_milestone_log.md`
- `docs/paper_multitask_sim_grasp_section.md`
- `docs/paper_related_work_citation_plan.md`
- `outputs/h200_60071_paper_baseline/outputs/paper_ablation_table.md`
- `outputs/h200_60071_absorber_aware_full_ambiguity_seed01234/reports/reobserve_policy_report.md`
- `outputs/h200_60071_multiview_memory_grasp_point_full_ambiguity_seed01234/reports/full_ambiguity_grasp_comparison.md`
- `outputs/h200_60071_stackcube_task_guard_expanded_seed0_49/reports/cross_task_pick_table.md`
- `outputs/h200_60071_stackcube_task_guard_expanded_seed0_49/reports/stackcube_guard_failure_report.md`

Stable smoke commands remain documented in the repository task spec and README.
For real HF and ManiSkill validation, H200 runs should use the helper scripts in
`scripts/invoke_h200_command.ps1` and `scripts/sync_h200_files.ps1` because the
remote workspace is a synced tree, not a git checkout.
