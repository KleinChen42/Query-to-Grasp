# Experiment A+B Results Report

Date: 2026-05-07  
Platform: 8x NVIDIA H200  
Scope: 1,500 simulation runs across 15 configurations, 100 seeds each.

## Experiment A: External RGB-D Crop-Grasp Baseline

Experiment A tests whether a standard crop-based RGB-D grasp target can replace
the Query-to-Grasp refined/memory target source. The result is a clean
target-source precision ladder.

### PickCube-v1

| target source | N | pick success |
| --- | ---: | ---: |
| `box_center_depth` | 100 | 0.010 |
| `crop_median` | 100 | 0.950 |
| `crop_top_surface` | 100 | 0.990 |
| `oracle_object_pose` | 100 | 1.000 |

### StackCube-v1 Pick-Only

| target source | N | pick success |
| --- | ---: | ---: |
| `box_center_depth` | 100 | 0.080 |
| `crop_median` | 100 | 0.820 |
| `crop_top_surface` | 100 | 0.840 |
| `oracle_object_pose` | 100 | 0.960 |

### StackCube-v1 Pick-Place

| pick target source | place target source | N | pick success | task success |
| --- | --- | ---: | ---: | ---: |
| `box_center_depth` | `predicted_place_object` | 100 | 0.080 | 0.000 |
| `crop_top_surface` | `predicted_place_object` | 100 | 0.760 | 0.570 |

Interpretation: `box_center_depth` answers the obvious reviewer baseline and
fails badly. Crop-level aggregation is far stronger, and the top-surface crop
heuristic gives a strong non-oracle pick-place bridge on StackCube.

## Experiment B: Non-Cube Diagnostic

The feasible non-cube executable diagnostic in the installed ManiSkill setup was
`LiftPegUpright-v1`. This is not presented as a successful non-cube manipulation
result; it is a controller-target compatibility diagnostic.

| env | query | target source | N | pick success |
| --- | --- | --- | ---: | ---: |
| `LiftPegUpright-v1` | `peg` | `box_center_depth` | 100 | 0.000 |
| `LiftPegUpright-v1` | `peg` | `crop_median` | 100 | 0.000 |
| `LiftPegUpright-v1` | `peg` | `crop_top_surface` | 100 | 0.000 |
| `LiftPegUpright-v1` | `object` | `crop_top_surface` | 100 | 0.000 |
| `LiftPegUpright-v1` | `peg` | `oracle_object_pose` | 100 | 0.000 |

Interpretation: even a privileged oracle target does not make the scripted
top-down executor succeed on a horizontal peg. The result supports a limitation:
target-source quality is necessary but not sufficient when object geometry and
controller approach are mismatched.

## Recommended Paper Use

- Main text: add a compact external crop baseline table.
- Discussion: mention LiftPeg as a non-cube controller-target coupling probe.
- Claim boundary: do not claim general non-cube manipulation or learned
  grasping from these experiments.
