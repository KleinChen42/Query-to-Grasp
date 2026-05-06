# IROS/ICRA Full-Paper Upgrade Roadmap

## Positioning Upgrade

Query-to-Grasp should now be positioned as an H200-scale target-source diagnostic benchmark for open-vocabulary RGB-D manipulation, not as a Colab-friendly inference demo.

The central paper question is no longer whether an open-vocabulary detector can find a named object in 2D. The stronger question is whether the resulting RGB-D target source is geometrically consistent, cross-view stable, and executable by a simulated manipulation controller.

Working title:

> Query-to-Grasp: a target-source diagnostic benchmark for open-vocabulary RGB-D manipulation

## Core Thesis

1. Open-vocabulary 2D detection is not the endpoint for robot manipulation.
2. RGB-D target-source quality determines whether a selected object becomes an executable action target.
3. Multi-view memory is meaningful only when camera geometry and 3D lifting are cross-view consistent.
4. PickCube shows that fused memory grasp targets can bridge semantic retrieval and executable simulated picking.
5. StackCube exposes the remaining retrieval-to-execution gap.
6. Oracle and non-oracle target-source ladders decompose whether failures come from perception targets, association, placement targets, or low-level scripted execution.

## Current Evidence

- PickCube executable target evidence: fused memory grasp targets support validated simulated pick execution, including full-query multi-view and closed-loop settings.
- StackCube pick-only evidence: task-aware target selection improves pick-only compatibility, but does not claim stacking completion.
- Oracle pick-place upper bound: privileged cubeA/cubeB poses show that the scripted pick-place executor can complete StackCube under oracle target sources.
- Query-pick plus oracle-place bridge: query-derived cubeA pick targets can drive partial StackCube task completion when cubeB placement remains privileged.
- Query-pick plus predicted-place bridge: explicit reference-object placement is now validated as a non-oracle diagnostic bridge, using `place_target_source=predicted_place_object` with an explicit `green cube` place query.

## Frozen H200 Predicted-Place Results (500 seeds, result freeze 2026-05-06)

### Refined predicted-place (`green cube` explicit query, no-CLIP)

| mode | seeds | pick success | place attempted | place success | task success |
| --- | ---: | ---: | ---: | ---: | ---: |
| single-view | 500 | 363/500 = 0.726 | 410/500 = 0.820 | 276/500 = 0.552 | 276/500 = 0.552 |
| tabletop multi-view | 500 | 279/500 = 0.558 | 476/500 = 0.952 | 236/500 = 0.472 | 236/500 = 0.472 |
| closed-loop | 500 | 267/500 = 0.534 | 466/500 = 0.932 | 223/500 = 0.446 | 223/500 = 0.446 |

All 500-seed results merged from `seed0_199` (200 runs) + `seed200_499` (300 runs) with 0 run failures. `place_success_rate == task_success_rate` in all modes, consistent with sim_pick_place StackCube semantics.

### Semantic-center baseline (200 seeds, no-CLIP)

| mode | runs | pick success | place attempted | place success | task success |
| --- | ---: | ---: | ---: | ---: | ---: |
| single-view | 200 | 0.745 | 0.900 | 0.565 | 0.565 |
| tabletop multi-view | 200 | 0.520 | 0.960 | 0.435 | 0.435 |
| closed-loop | 200 | 0.510 | 0.930 | 0.420 | 0.420 |

### With-CLIP ablation (200 seeds)

| mode | no-CLIP task success | with-CLIP task success | delta |
| --- | ---: | ---: | --- |
| single-view | 0.585 | 0.590 | +0.005 (negligible) |
| tabletop multi-view | 0.435 | 0.455 | +0.020 (marginal) |
| closed-loop | 0.420 | 0.440 | +0.020 (marginal) |

### Broad vs explicit place-query specificity ablation (200 seeds, no-CLIP)

| mode | `green cube` (explicit) task | `cube` (broad) task | delta |
| --- | ---: | ---: | ---: |
| single-view | 0.585 | 0.360 | -0.225 |
| tabletop multi-view | 0.435 | 0.305 | -0.130 |
| closed-loop | 0.420 | 0.290 | -0.130 |

This specificity gap is a strong paper finding: explicit reference-object queries (`green cube`) significantly outperform broad queries (`cube`) for predicted-place target formation.

### Task-diversity target-source formation (H200, 200 seeds)

| env | detection rate | 3D target rate | notes |
| --- | ---: | ---: | --- |
| PushCube-v1 | 1.000 | 1.000 | Target-source formation successful; sim-topdown pick runs as diagnostic |
| LiftPegUpright-v1 | 1.000 | 1.000 | Placeholder pick only; sim_topdown not compatible |
| PegInsertionSide-v1 | 1.000 | 1.000 | Placeholder pick only; sim_topdown not compatible |
| StackPyramid-v1 | 0.970 | 0.970 | Placeholder pick only; 6/200 runs had no detection |

These results validate that the query-to-3D-target pipeline generalizes beyond PickCube/StackCube for target-source formation, even though the current diagnostic controller is not compatible with non-cube manipulation tasks.

## Critical Reviewer Risks

- Task diversity is still narrow. PickCube and StackCube are useful diagnostics, but a full IROS/ICRA paper needs broader ManiSkill task coverage or a clearly bounded diagnostic-benchmark framing.
- Oracle placement must not dominate the final story. It is an upper bound and failure-source probe, not a deployable perception result.
- Baselines are mostly internal. The final paper should present them as a target-source ladder and, where feasible, add at least lightweight external or task-diversity comparisons.
- The scripted top-down controller is deliberately diagnostic, but it limits claims about general grasping and non-top-down manipulation.
- Re-observation claims must remain narrow: this rule-based policy can reduce semantic or selection uncertainty, but it does not necessarily improve manipulation success.

## Engineering Priorities

1. ~~Record and commit the non-oracle StackCube placement predictor separately from paper-writing changes.~~ Done (f3ebd65).
2. ~~Run a ManiSkill candidate task inventory before launching more large validations.~~ Done: PushCube/LiftPeg/PegInsertion/StackPyramid tested.
3. ~~Smoke-test 3-5 seeds per candidate task for RGB-D extraction, HF detection, 3D lifting, target-source formation, and whether a scripted diagnostic action is meaningful.~~ Done.
4. ~~Upgrade the main results table into a target-source ladder rather than a chronology of experiments.~~ In progress: frozen data ready.
5. Keep native-720p continuous execution videos as presentation evidence; only recapture 1-2 representative predicted-place clips if needed.

## Target-Source Ladder (Paper Table I, frozen)

| row | pick target source | place target source | env | seeds | pick | place | task | claim |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | oracle_object_pose | — | PickCube | 50 | 1.000 | — | 0.040 | Oracle pick upper-bound |
| 2 | memory_grasp_world_xyz | — | PickCube | 55 | 1.000 | — | 0.145 | Fused memory pick |
| 3 | oracle_cubeA_pose | — | StackCube | 50 | 0.940 | — | 0.000 | Oracle pick upper-bound |
| 4 | task_guard_selected_xyz | — | StackCube | 50 | 0.620 (tab) / 0.520 (cl) | — | 0.000 | Query pick-only |
| 5 | oracle_cubeA | oracle_cubeB | StackCube | 50 | 0.940 | 0.880 | 0.880 | Oracle pick-place |
| 6 | query `red cube` | oracle_cubeB | StackCube | 50 | 0.880 | 0.720 | 0.720 | Query-pick + oracle-place |
| 7 | query `red cube` | semantic-center `green cube` | StackCube | 200 | 0.745 | 0.565 | 0.565 | Semantic predicted-place baseline |
| 8 | query `red cube` | refined `green cube` | StackCube | **500** | **0.726** | **0.552** | **0.552** | Refined predicted-place (main result) |
| 9 | query `red cube` | refined `cube` (broad) | StackCube | 200 | 0.535 | 0.360 | 0.360 | Broad query ablation |

## Paper Claim Boundaries

The paper can claim:

- A diagnostic system for converting open-vocabulary RGB-D detections into 3D target sources.
- Evidence that target-source quality matters for downstream simulated manipulation.
- Multi-view memory benefits only after cross-view geometric consistency is corrected.
- PickCube simulated picking can be executed with fused memory grasp targets.
- StackCube exposes target-source, association, and placement-target gaps.
- Explicit reference-object predicted placement is a non-oracle diagnostic bridge, not a full language-conditioned stacking solution.
- Place-query specificity significantly affects predicted-place success.

The paper must not claim:

- real-robot execution,
- learned grasping or learned control,
- learned active perception,
- robust relation-heavy language grounding,
- full non-oracle StackCube stacking completion,
- a general conclusion that active perception is unhelpful.

## Milestones

1. ~~Commit and record the non-oracle StackCube placement predictor.~~ Done.
2. ~~Run a ManiSkill multi-task feasibility scan.~~ Done (PushCube, LiftPeg, PegInsertion, StackPyramid).
3. ~~Refresh the paper table into a target-source ladder.~~ Done (frozen 2026-05-06).
4. ~~Pull seed 200-499 data and merge 500-seed statistics.~~ Done (verified 2026-05-06).
5. Upgrade `architecture_query_to_grasp.md` to reflect current sim-pick/sim-pick-place/predicted-place architecture. **In progress.**
6. Update `paper/main.tex` with 500-seed frozen results and target-source ladder.
7. Final LaTeX tightening and citation verification.
