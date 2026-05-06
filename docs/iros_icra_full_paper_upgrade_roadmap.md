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

Latest predicted-place H200 diagnostic result, pending paper-table integration:

| mode | runs | run failures | pick success | place attempted | task success |
| --- | ---: | ---: | ---: | ---: | ---: |
| single-view no-CLIP | 50 | 0 | 39/50 = 0.78 | 41/50 = 0.82 | 29/50 = 0.58 |
| single-view with-CLIP | 50 | 0 | 39/50 = 0.78 | 41/50 = 0.82 | 29/50 = 0.58 |
| tabletop multi-view no-CLIP | 50 | 0 | 30/50 = 0.60 | 49/50 = 0.98 | 24/50 = 0.48 |
| closed-loop no-CLIP | 50 | 0 | 26/50 = 0.52 | 45/50 = 0.90 | 22/50 = 0.44 |

This result should be described carefully. The system is not performing free-form relation parsing. The stronger and more defensible phrasing is: explicit reference-object place query (`green cube`) enables a non-oracle StackCube placement-target bridge.

## Critical Reviewer Risks

- Task diversity is still narrow. PickCube and StackCube are useful diagnostics, but a full IROS/ICRA paper needs broader ManiSkill task coverage or a clearly bounded diagnostic-benchmark framing.
- Oracle placement must not dominate the final story. It is an upper bound and failure-source probe, not a deployable perception result.
- Baselines are mostly internal. The final paper should present them as a target-source ladder and, where feasible, add at least lightweight external or task-diversity comparisons.
- The scripted top-down controller is deliberately diagnostic, but it limits claims about general grasping and non-top-down manipulation.
- Re-observation claims must remain narrow: this rule-based policy can reduce semantic or selection uncertainty, but it does not necessarily improve manipulation success.

## Engineering Priorities

1. Record and commit the non-oracle StackCube placement predictor separately from paper-writing changes.
2. Run a ManiSkill candidate task inventory before launching more large validations.
3. Smoke-test 3-5 seeds per candidate task for RGB-D extraction, HF detection, 3D lifting, target-source formation, and whether a scripted diagnostic action is meaningful.
4. Upgrade the main results table into a target-source ladder rather than a chronology of experiments.
5. Keep native-720p continuous execution videos as presentation evidence; only recapture 1-2 representative predicted-place clips if needed.

Candidate tasks for the next feasibility scan:

- `PickSingleYCB-v1`
- `PickClutterYCB-v1`, if available
- `PushCube-v1`
- `PegInsertionSide-v1`
- Other ManiSkill table manipulation tasks with RGB-D observations and object-like target structure

## Target-Source Ladder For Final Tables

The main paper table should be organized by target source, not by implementation history:

- semantic center
- single-view refined grasp point
- fused memory grasp point
- StackCube task-guard semantic target
- predicted place object
- oracle pick pose
- oracle place pose
- query-pick plus oracle-place
- query-pick plus predicted-place

The table should explicitly separate pick source and place source. It should report pick success, place success where applicable, task success where applicable, seed counts, and claim boundary.

## Paper Claim Boundaries

The paper can claim:

- A diagnostic system for converting open-vocabulary RGB-D detections into 3D target sources.
- Evidence that target-source quality matters for downstream simulated manipulation.
- Multi-view memory benefits only after cross-view geometric consistency is corrected.
- PickCube simulated picking can be executed with fused memory grasp targets.
- StackCube exposes target-source, association, and placement-target gaps.
- Explicit reference-object predicted placement is a non-oracle diagnostic bridge, not a full language-conditioned stacking solution.

The paper must not claim:

- real-robot execution,
- learned grasping or learned control,
- learned active perception,
- robust relation-heavy language grounding,
- full non-oracle StackCube stacking completion,
- a general conclusion that active perception is unhelpful.

## Next 3 Milestones

1. Commit and record the non-oracle StackCube placement predictor.
   - Use the H200 50-seed predicted-place result as a diagnostic checkpoint.
   - Mark `green cube` as an explicit reference-object place query.

2. Run a ManiSkill multi-task feasibility scan.
   - Inventory available candidate tasks on H200.
   - Run 3-5 seed smoke tests only.
   - Select 2-3 non-cube or clutter-relevant tasks for a broader diagnostic benchmark.

3. Refresh the paper table into a target-source ladder.
   - Include PickCube, StackCube pick-only, oracle pick, oracle pick-place, query-pick plus oracle-place, and query-pick plus predicted-place.
   - Preserve strict claim boundaries and separate presentation videos from quantitative evidence.
