# IROS/ICRA Full-Paper Upgrade Roadmap

This roadmap is the main planning entry point for upgrading Query-to-Grasp from
a lightweight pipeline demo into an H200-scale diagnostic paper.

## Positioning Upgrade

The paper should be positioned as:

> Query-to-Grasp: a target-source diagnostic benchmark for open-vocabulary
> RGB-D manipulation.

The central question is whether an open-vocabulary RGB-D observation can produce
a 3D target source that remains geometrically consistent, cross-view stable, and
executable by a simulated manipulation controller.

## Core Thesis

1. Open-vocabulary 2D detection is not the endpoint for robot manipulation.
2. RGB-D target-source quality determines whether language grounding becomes an
   executable action target.
3. Multi-view memory is useful only when camera-frame conventions and 3D lifting
   are geometrically consistent.
4. PickCube demonstrates executability when fused memory grasp targets are
   precise.
5. StackCube exposes the retrieval-to-execution gap.
6. Oracle and non-oracle target-source ladders decompose whether failures come
   from pick targets, place targets, association, or scripted execution.

## Current Evidence

### PickCube Executability

PickCube is the strongest positive execution result. The refined/fused grasp
target reaches 55/55 = 1.000 simulated pick success in the validated target
point ablation, while the semantic center reaches 53/55 = 0.964.

### StackCube Pick-Only and Oracle Upper Bounds

StackCube remains a diagnostic task, not a full non-oracle stacking claim.
Existing results separate:

- pick-only compatibility;
- oracle pick upper bound;
- fully oracle pick-place upper bound;
- query-pick plus oracle-place bridge.

### Non-Oracle Predicted Placement

The main upgrade result is now frozen at 500 seeds for explicit reference-object
placement using `place_target_source=predicted_place_object` and the place query
`green cube`.

| target mode | view mode | runs | pick success | place attempted | task success |
| --- | --- | ---: | ---: | ---: | ---: |
| refined | single-view | 500 | 363/500 = 0.726 | 410/500 = 0.820 | 276/500 = 0.552 |
| refined | tabletop multi-view | 500 | 279/500 = 0.558 | 476/500 = 0.952 | 236/500 = 0.472 |
| refined | closed-loop | 500 | 267/500 = 0.534 | 466/500 = 0.932 | 223/500 = 0.446 |
| semantic | single-view | 500 | 365/500 = 0.730 | 442/500 = 0.884 | 275/500 = 0.550 |
| semantic | tabletop multi-view | 500 | 279/500 = 0.558 | 476/500 = 0.952 | 236/500 = 0.472 |
| semantic | closed-loop | 500 | 268/500 = 0.536 | 468/500 = 0.936 | 223/500 = 0.446 |

This result is useful but not perfect. It supports a non-oracle reference-object
placement bridge, while still showing that multi-view and closed-loop recall do
not automatically improve physical execution.

### Noisy Oracle Sensitivity

The paper-revision sensitivity analysis shows the centimeter scale of the
execution bottleneck.

| perturbation | key result |
| --- | --- |
| PickCube pick noise 2 cm | pick success drops to 31/50 = 0.620 |
| PickCube pick noise 5 cm | pick success drops to 6/50 = 0.120 |
| StackCube pick noise 2 cm | pick success drops to 22/50 = 0.440 |
| StackCube pick noise 5 cm | pick success drops to 3/50 = 0.060 |
| StackCube place noise 2 cm | task success drops to 12/50 = 0.240 |
| StackCube place noise 5 cm | task success drops to 1/50 = 0.020 |

This gives the paper a stronger causal diagnostic: downstream control is highly
sensitive to target-source errors at the 2 cm scale.

### CLIP Reranking Ablation

With CLIP enabled:

- PickCube: 50/50 = 1.000 pick success, top-1 changed 0/50.
- StackCube: 44/50 = 0.880 pick success, top-1 changed 0/50.

Candidate pools are small (mean detections 1.040 and 1.480), so CLIP has no
ranking room in the current scenes. The claim should be narrow: reranking is not
the observed bottleneck under these candidate-pool sizes.

### Task-Diversity Target-Source Formation

Additional ManiSkill target-source formation runs broaden the diagnostic beyond
PickCube/StackCube:

| env | runs | target-source formation |
| --- | ---: | ---: |
| PushCube-v1 single-view | 200 | 200/200 = 1.000 |
| PushCube-v1 tabletop | 200 | 200/200 = 1.000 |
| PushCube-v1 closed-loop | 200 | 200/200 = 1.000 |
| LiftPegUpright-v1 | 200 | 200/200 = 1.000 |
| PegInsertionSide-v1 | 200 | 200/200 = 1.000 |
| StackPyramid-v1 | 200 | 194/200 = 0.970 |

These rows should be described as target-source formation diversity, not as
manipulation-success results.

## Critical Reviewer Risks

- Task diversity remains weaker than a full general manipulation benchmark.
- The controller is scripted and top-down; the paper should frame it as a
  diagnostic isolation tool, not a general grasping solution.
- Oracle rows must remain privileged baselines and upper bounds.
- The predicted-place bridge uses an explicit reference-object query
  (`green cube`), not general relation-heavy language understanding.
- CLIP reranking is not shown to help in the current scenes.
- Re-observation claims must be narrow: this rule-based policy can reduce
  uncertainty or increase target availability, but it does not necessarily
  improve manipulation success.

## Engineering Priorities

1. Done: implement and validate non-oracle StackCube predicted placement.
2. Done: expand predicted-place validation to 500 seeds for refined and semantic
   target modes.
3. Done: add noisy oracle pick/place sensitivity analysis.
4. Done: add CLIP reranking and PickCube target-point ablations.
5. Done: add target-source formation diversity for additional ManiSkill tasks.
6. Next: integrate the frozen tables into `paper/main.tex`.
7. Next: update the paper figures and captions around the target-source ladder.
8. Next: tighten citations and final claim boundaries before arXiv/workshop or
   main-track submission.

## Target-Source Ladder For Paper Table

The main results table should be organized by target source:

| row | pick source | place source | env | runs | pick | place/task | claim |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | semantic center | none | PickCube | 55 | 0.964 | n/a | target-point baseline |
| 2 | refined grasp point | none | PickCube | 55 | 1.000 | n/a | executable refined target |
| 3 | memory grasp point | none | PickCube | 55 | 1.000 | n/a | fused memory executability |
| 4 | task-guard target | none | StackCube | 50 | 0.620 tabletop / 0.520 closed-loop | 0.000 | pick-only compatibility |
| 5 | oracle cubeA | oracle cubeB | StackCube | 50 | 0.940 | 0.880 | privileged upper bound |
| 6 | query cubeA | oracle cubeB | StackCube | 50 | 0.880 | 0.720 single | query-pick bridge |
| 7 | query cubeA | predicted `green cube` | StackCube | 500 | 0.726 single | 0.552 single | main non-oracle bridge |
| 8 | query cubeA | predicted broad `cube` | StackCube | 200 | 0.535 single | 0.360 single | reference-query ablation |
| 9 | noisy oracle targets | noisy oracle targets | StackCube/PickCube | 50 per noise | varies | varies | sensitivity analysis |

The exact table can be compressed in LaTeX, but this ordering should be
preserved: it reads as a target-source ladder rather than an implementation log.

## Claim Boundaries

The paper can claim:

- a reproducible diagnostic framework for open-vocabulary RGB-D target-source
  formation;
- executable simulated pick evidence on PickCube;
- target-source, reference-query, and centimeter-scale sensitivity analyses;
- non-oracle explicit reference-object placement on StackCube;
- task-source formation diversity across additional ManiSkill tasks.

The paper must not claim:

- real-robot execution;
- learned grasping, learned control, or learned active perception;
- robust relation-heavy language grounding;
- full non-oracle StackCube stacking completion;
- a universal conclusion that active perception is unhelpful.

## Next 3 Milestones

1. Integrate the frozen paper-revision summary into `paper/main.tex`.
2. Regenerate the paper pack and submission audit with the new result tables.
3. Compile and inspect the PDF for table density, claim boundaries, and whether
   the target-source diagnostic narrative is clear enough for IROS/ICRA review.
