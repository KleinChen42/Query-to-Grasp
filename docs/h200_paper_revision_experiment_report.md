# H200 Paper-Revision Experiment Report

This report freezes the May 2026 paper-revision experiments for the
IROS/ICRA-style Query-to-Grasp manuscript. It replaces earlier informal notes
and uses the pulled benchmark rows as the source of truth.

## Scope

- Platform: 8x NVIDIA H200 remote validation host.
- Runtime window: 2026-05-06 to 2026-05-07.
- Total simulation runs summarized here: 1710.
- Run failures: 0 across the paper-revision experiments.
- Generated table source:
  `outputs/paper_revision_results_summary_latest/paper_revision_results_summary.md`.

These experiments are diagnostic analyses. They do not introduce a learned
controller, learned grasp policy, learned active-perception policy, or
real-robot result.

## Code Interfaces Added

The paper-revision runs used additive benchmark options:

- `--oracle-pick-noise-std`: injects seed-deterministic Gaussian noise into the
  pick target after target selection and before execution.
- `--oracle-place-noise-std`: injects seed-deterministic Gaussian noise into the
  privileged StackCube place target when `--place-target-source oracle_cubeB_pose`.
- `--start-seed` and `--num-seeds`: convenience arguments for long contiguous
  seed ranges.

The noise paths record the pre-noise target, applied noise vector, final target,
and noise standard deviation in run metadata. Pick and place noise use separate
seeded RNG streams.

## Result Tables

### Noisy Oracle Pick Sensitivity

| env | pick noise | runs | pick success |
| --- | ---: | ---: | ---: |
| PickCube-v1 | 1 cm | 50 | 46/50 = 0.920 |
| PickCube-v1 | 2 cm | 50 | 31/50 = 0.620 |
| PickCube-v1 | 5 cm | 50 | 6/50 = 0.120 |
| StackCube-v1 | 1 cm | 50 | 35/50 = 0.700 |
| StackCube-v1 | 2 cm | 50 | 22/50 = 0.440 |
| StackCube-v1 | 5 cm | 50 | 3/50 = 0.060 |

The sensitivity curve is steep: a 2 cm target perturbation already cuts pick
success sharply, and 5 cm perturbations make the scripted top-down execution
nearly unusable.

### Noisy Oracle Pick-Place

| pick noise | runs | pick success | place success | task success |
| ---: | ---: | ---: | ---: | ---: |
| 1 cm | 50 | 35/50 = 0.700 | 24/50 = 0.480 | 24/50 = 0.480 |
| 2 cm | 50 | 22/50 = 0.440 | 9/50 = 0.180 | 9/50 = 0.180 |
| 5 cm | 50 | 3/50 = 0.060 | 1/50 = 0.020 | 1/50 = 0.020 |

Pick target error propagates through the full StackCube pick-place chain. This
supports the paper framing that target-source precision is an execution
bottleneck rather than a cosmetic perception metric.

### Noisy Oracle Place Sensitivity

| place noise | runs | pick success | place success | task success |
| ---: | ---: | ---: | ---: | ---: |
| 1 cm | 50 | 44/50 = 0.880 | 26/50 = 0.520 | 26/50 = 0.520 |
| 2 cm | 50 | 44/50 = 0.880 | 12/50 = 0.240 | 12/50 = 0.240 |
| 5 cm | 50 | 44/50 = 0.880 | 1/50 = 0.020 | 1/50 = 0.020 |

The pick phase remains stable because only the place target is perturbed. Place
success falls from 0.520 at 1 cm to 0.020 at 5 cm, showing that destination
target precision is an independent bottleneck for stacking-style tasks.

### PickCube Target Point Ablation

| target point | runs | pick success |
| --- | ---: | ---: |
| semantic center (`world_xyz`) | 55 | 53/55 = 0.964 |
| refined grasp point (`grasp_world_xyz`) | 55 | 55/55 = 1.000 |

The refined grasp point closes the small residual PickCube gap relative to a
semantic center target.

### CLIP Reranking Ablation

| setup | runs | pick success | top-1 changed | mean detections |
| --- | ---: | ---: | ---: | ---: |
| PickCube-v1 with CLIP | 50 | 50/50 = 1.000 | 0/50 = 0.000 | 1.040 |
| StackCube-v1 with CLIP | 50 | 44/50 = 0.880 | 0/50 = 0.000 | 1.480 |

In the current low-ambiguity scenes, CLIP has no ranking room: detector
candidate pools are small and the top-1 candidate never changes. The correct
paper claim is that CLIP reranking is not the bottleneck under these candidate
pool sizes.

### StackCube Predicted-Place 500-Seed Ladder

| target mode | view mode | runs | pick success | place attempted | task success |
| --- | --- | ---: | ---: | ---: | ---: |
| refined | single-view | 500 | 363/500 = 0.726 | 410/500 = 0.820 | 276/500 = 0.552 |
| refined | tabletop multi-view | 500 | 279/500 = 0.558 | 476/500 = 0.952 | 236/500 = 0.472 |
| refined | closed-loop | 500 | 267/500 = 0.534 | 466/500 = 0.932 | 223/500 = 0.446 |
| semantic | single-view | 500 | 365/500 = 0.730 | 442/500 = 0.884 | 275/500 = 0.550 |
| semantic | tabletop multi-view | 500 | 279/500 = 0.558 | 476/500 = 0.952 | 236/500 = 0.472 |
| semantic | closed-loop | 500 | 268/500 = 0.536 | 468/500 = 0.936 | 223/500 = 0.446 |

The non-oracle StackCube bridge is now a 500-seed result, not a small smoke
test. It should be described as explicit reference-object placement using the
query `green cube`, not as robust relation-heavy language grounding.

### Reference Query Specificity

| place query | view mode | runs | pick success | place attempted | task success |
| --- | --- | ---: | ---: | ---: | ---: |
| `green cube` refined | single-view | 500 | 363/500 = 0.726 | 410/500 = 0.820 | 276/500 = 0.552 |
| `cube` broad | single-view | 200 | 107/200 = 0.535 | 127/200 = 0.635 | 72/200 = 0.360 |
| `green cube` refined | tabletop multi-view | 500 | 279/500 = 0.558 | 476/500 = 0.952 | 236/500 = 0.472 |
| `cube` broad | tabletop multi-view | 200 | 94/200 = 0.470 | 184/200 = 0.920 | 61/200 = 0.305 |
| `green cube` refined | closed-loop | 500 | 267/500 = 0.534 | 466/500 = 0.932 | 223/500 = 0.446 |
| `cube` broad | closed-loop | 200 | 98/200 = 0.490 | 193/200 = 0.965 | 58/200 = 0.290 |

Explicit reference-object queries substantially outperform broad reference
queries. This is a key limitation and a useful diagnostic finding.

### Task-Diversity Target-Source Formation

| env | view mode | runs | target-source formation |
| --- | --- | ---: | ---: |
| PushCube-v1 | single-view | 200 | 200/200 = 1.000 |
| PushCube-v1 | tabletop multi-view | 200 | 200/200 = 1.000 |
| PushCube-v1 | closed-loop | 200 | 200/200 = 1.000 |
| LiftPegUpright-v1 | single-view | 200 | 200/200 = 1.000 |
| PegInsertionSide-v1 | single-view | 200 | 200/200 = 1.000 |
| StackPyramid-v1 | single-view | 200 | 194/200 = 0.970 |

These rows broaden the paper beyond PickCube and StackCube for target-source
formation. They are not manipulation-success claims because the current
scripted controller is not task-compatible with these non-cube manipulation
families.

## Main Paper Takeaways

1. Target-source precision matters at centimeter scale. Noisy oracle runs show
   that 2 cm errors can sharply reduce execution success.
2. CLIP reranking is not the limiting factor in the current low-candidate
   scenes; candidate multiplicity is too small for reranking to matter.
3. Multi-view memory increases recall and target availability, but the
   execution results show that aggregation can reduce target precision.
4. StackCube predicted-place results now provide a non-oracle reference-object
   bridge, but not a full non-oracle stacking claim.
5. The paper should be framed as a reproducible target-source diagnostic
   framework for simulated manipulation, not as a general robot deployment
   system.

## Paper Integration To-Do

1. Update the LaTeX main table into a target-source ladder.
2. Add a compact noisy-oracle sensitivity table or figure.
3. Add a CLIP reranking paragraph that explains the small candidate-pool result.
4. Add a target-point ablation paragraph for PickCube.
5. Reframe multi-view and re-observation as diagnostic recall/precision probes,
   not as universally beneficial active perception.
