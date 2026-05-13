# External Crop-Grasp Baseline and Non-Cube Diagnostic Results

This note freezes the Experiment A/B results used for the IROS/ICRA paper
revision. The experiments are diagnostic: they do not introduce a learned
policy, a new grasp network, or real-robot execution. They test whether simple
RGB-D target sources and a non-cube gate expose failure modes that are hidden
by cube-only evaluations.

## Experiment A: External RGB-D Crop-Grasp Baseline

Experiment A asks why Query-to-Grasp does not simply back-project the selected
GroundingDINO bounding-box center and use that point as a grasp target. Three
deterministic crop-based target sources were compared with a privileged oracle
over 200 seeds.

| target source | definition | diagnostic role |
| --- | --- | --- |
| `box_center_depth` | selected 2D box center pixel lifted to 3D, with a small patch-median fallback for invalid depth | simplest reviewer baseline |
| `crop_median` | median of valid RGB-D points inside the selected crop after workspace filtering | standard crop aggregation baseline |
| `crop_top_surface` | workspace-filtered crop points, elevated support extraction, XY component clustering, and local support-height target | strongest crop heuristic |
| `oracle_object_pose` | privileged simulator actor pose | upper bound, not deployable perception |

### PickCube-v1, 200 Seeds

| target source | N | pick success | Wilson 95% CI | interpretation |
| --- | ---: | ---: | --- | --- |
| `box_center_depth` | 200 | 0.010 | [0.003, 0.036] | single-pixel depth is too noisy |
| `crop_median` | 200 | 0.935 | [0.892, 0.962] | robust crop aggregation is already strong |
| `crop_top_surface` | 200 | 0.990 | [0.964, 0.997] | geometric filtering nearly reaches oracle |
| `oracle_object_pose` | 200 | 1.000 | [0.981, 1.000] | privileged ceiling |

The gap from `box_center_depth` to `crop_top_surface` is 98 percentage points.
This directly supports the paper's thesis that 2D grounding is insufficient
unless the selected RGB-D target source is geometrically precise.

### StackCube-v1 Pick-Only, 200 Seeds

| target source | N | pick success | Wilson 95% CI | interpretation |
| --- | ---: | ---: | --- | --- |
| `box_center_depth` | 200 | 0.085 | [0.054, 0.132] | near-random in the harder cube scene |
| `crop_median` | 200 | 0.795 | [0.734, 0.845] | strong non-oracle crop baseline |
| `crop_top_surface` | 200 | 0.800 | [0.739, 0.850] | comparable to crop median |
| `oracle_object_pose` | 200 | 0.945 | [0.904, 0.969] | privileged pick upper bound |

StackCube degrades all non-oracle target sources relative to PickCube, but the
crop-based sources remain far stronger than box-center depth. This suggests the
important comparison is not semantic detection versus no detection, but target
source precision after detection.

### StackCube-v1 Pick-Place, 200 Seeds

| pick target source | place target source | N | pick success | task success | Wilson 95% CI (task) | interpretation |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `box_center_depth` | `predicted_place_object` | 200 | 0.085 | 0.000 | [0.000, 0.019] | poor pick target prevents completion |
| `crop_median` | `predicted_place_object` | 200 | 0.745 | 0.565 | [0.496, 0.632] | crop aggregation supports a non-oracle bridge |
| `crop_top_surface` | `predicted_place_object` | 200 | 0.745 | 0.585 | [0.516, 0.651] | strongest 200-seed crop bridge |

The crop-based pick-place rows use query-derived pick targets and predicted
reference-object placement. They are useful non-oracle bridges, but remain
diagnostic evidence rather than claims of robust relation-heavy stacking.

## Experiment B: Non-Cube Feasibility Gate

Experiment B tests whether target-source quality is sufficient when the
environment or object geometry changes. The gate used 50 seeds per row and
separates runtime compatibility failures from valid manipulation diagnostics.

| env | query | target source | N | failed runs | target available | pick success | interpretation |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `PickSingleYCB-v1` | `object` | `oracle_object_pose` | 50 | 50 | 0.000 | 0.000 | runtime/asset compatibility failure, not a manipulation result |
| `PickSingleYCB-v1` | `object` | `crop_top_surface` | 50 | 50 | 0.000 | 0.000 | runtime/asset compatibility failure |
| `PickClutterYCB-v1` | `object` | `oracle_object_pose` | 50 | 50 | 0.000 | 0.000 | runtime/asset compatibility failure |
| `PickClutterYCB-v1` | `object` | `crop_top_surface` | 50 | 50 | 0.000 | 0.000 | runtime/asset compatibility failure |
| `PickSingleEGAD-v1` | `object` | `oracle_object_pose` | 50 | 50 | 0.000 | 0.000 | runtime/asset compatibility failure |
| `PickSingleEGAD-v1` | `object` | `crop_top_surface` | 50 | 50 | 0.000 | 0.000 | runtime/asset compatibility failure |
| `LiftPegUpright-v1` | `peg` | `oracle_object_pose` | 50 | 0 | 1.000 | 0.000 | valid executor-geometry mismatch diagnostic |
| `LiftPegUpright-v1` | `peg` | `crop_top_surface` | 50 | 0 | 1.000 | 0.000 | target exists, but top-down executor fails |

The LiftPeg oracle failure is the important diagnostic result. It shows that
target-source quality is necessary but not sufficient: the scripted top-down
executor must also be compatible with object geometry. The YCB/EGAD failures
should not be written as negative manipulation evidence because the runs failed
at compatibility/runtime setup.

## Generated Paper Artifacts

The frozen 200-seed summary and paper-facing figure are generated by:

```bash
python scripts/generate_external_crop_200seed_summary.py \
  --output-dir outputs/paper_revision_results_summary_latest \
  --figure-dir paper/figures
```

Generated files:

```text
outputs/paper_revision_results_summary_latest/external_crop_200seed_summary.md
outputs/paper_revision_results_summary_latest/external_crop_200seed_summary.csv
outputs/paper_revision_results_summary_latest/external_crop_200seed_summary.json
paper/figures/external_crop_200seed_results.pdf
paper/figures/external_crop_200seed_results.png
```

## Paper Integration

Recommended manuscript handling:

- Update the external crop baseline table to 200 seeds.
- Include the compact bar figure comparing PickCube pick, StackCube pick-only,
  and StackCube pick-place task success.
- Treat PickCube pick-only raw environment success as a raw ManiSkill flag, not
  the primary metric.
- Write LiftPeg as a controller-target compatibility diagnostic.
- Do not claim general non-cube manipulation, learned grasping, or real-robot
  execution from Experiment B.
