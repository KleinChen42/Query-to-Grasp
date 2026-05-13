# Experiment A+B Comprehensive Report

This comprehensive report records the implementation and execution context for
the external RGB-D crop-grasp baseline and non-cube executable diagnostic. The
paper-facing clean summary is in
`docs/external_grasp_and_noncube_results.md`.

## Motivation

The IROS/ICRA revision needed two additional diagnostics:

1. An external-style RGB-D crop-grasp baseline answering why the system does not
   simply use a GroundingDINO box center plus depth.
2. A non-cube execution probe showing whether the retrieval-to-target interface
   and scripted executor remain meaningful outside cube geometry.

The goal was not to build a new general robot policy, introduce AnyGrasp, train
a controller, or claim real-robot execution.

## Implemented Target Modes

| mode | implementation | role |
| --- | --- | --- |
| `box_center_depth` | lift selected box center depth, with center-patch fallback | simplest crop baseline |
| `crop_median` | median of valid crop RGB-D points after workspace filtering | standard aggregation baseline |
| `crop_top_surface` | elevated crop support, XY largest component, local support height | deterministic crop-grasp heuristic |
| `oracle_object_pose` | privileged ManiSkill actor pose | diagnostic upper bound |

Code paths were added in parallel with existing semantic/refined/memory targets,
so existing paper results are not redefined.

## Modified Files

| file | change |
| --- | --- |
| `src/perception/mask_projector.py` | stores `box_center_world_xyz` in candidate metadata |
| `src/manipulation/oracle_targets.py` | adds generic `find_oracle_pick_xyz` actor lookup |
| `scripts/run_single_view_pick.py` | adds target-source routing for the new modes |
| `scripts/run_single_view_pick_benchmark.py` | forwards the new `--grasp-target-mode` choices |

## Execution Summary

The final valid launch used the benchmark default depth scale. An earlier launch
with `--depth-scale 1.0` was invalid because depth values were treated as meters
instead of millimeters; those outputs were discarded.

Final H200 run:

- Platform: 8x NVIDIA H200
- Runs: 15 configurations x 100 seeds = 1,500 runs
- Environments: `PickCube-v1`, `StackCube-v1`, `LiftPegUpright-v1`
- Executors: `sim_topdown`, `sim_pick_place`
- Approximate duration: 5 hours, including smoke, parallel PickCube/StackCube
  validation, and parallel LiftPeg/StackCube pick-place validation.

### Execution Phases

| phase | scope | purpose |
| --- | --- | --- |
| Smoke | one seed per PickCube target mode | verify depth scale and target routing before long runs |
| Phase 2 | PickCube and StackCube pick-only, 8 parallel jobs | validate crop baselines and oracle pick upper bounds |
| Phase 3 | LiftPeg diagnostics and StackCube pick-place, 7 parallel jobs | validate non-cube compatibility and non-oracle pick-place bridge |

### Depth-Scale Bug and Resolution

An initial launch passed `--depth-scale 1.0`, which treated millimeter depth
values as meters. Depth-derived world coordinates became roughly 1000x too
large, for example about 643 m instead of about 0.643 m. All incomplete outputs
from that invalid launch were discarded. The valid run removed the explicit
depth-scale override and used the benchmark default `1000.0` conversion from
millimeters to meters.

### Test Status

The experiment implementation was validated with the full local test suite
after the code changes:

```text
241 passed, 0 failed
```

## Results

See `docs/external_grasp_and_noncube_results.md` for clean paper-facing tables.
The key results are:

- `box_center_depth` reaches only 0.010 PickCube pick success and 0.080
  StackCube pick success.
- `crop_top_surface` reaches 0.990 PickCube pick success, 0.840 StackCube
  pick-only success, and 0.570 StackCube pick-place task success with predicted
  reference placement.
- `LiftPegUpright-v1` reaches 0.000 pick success for all target sources,
  including `oracle_object_pose`, showing controller-target geometry mismatch.

## Paper Interpretation

Experiment A strengthens the target-source precision thesis. Experiment B should
be written as a limitation and diagnostic result: good target sources are not
enough if the executor is incompatible with the object's geometry.

## Frozen Output Directories

```text
outputs/h200_exp_a_pickcube_box_center_depth_seed0_99/
outputs/h200_exp_a_pickcube_crop_median_seed0_99/
outputs/h200_exp_a_pickcube_crop_top_surface_seed0_99/
outputs/h200_exp_a_pickcube_oracle_object_pose_seed0_99/
outputs/h200_exp_a_stackcube_pickonly_box_center_depth_seed0_99/
outputs/h200_exp_a_stackcube_pickonly_crop_median_seed0_99/
outputs/h200_exp_a_stackcube_pickonly_crop_top_surface_seed0_99/
outputs/h200_exp_a_stackcube_pickonly_oracle_object_pose_seed0_99/
outputs/h200_exp_a_stackcube_pickplace_box_center_depth_seed0_99/
outputs/h200_exp_a_stackcube_pickplace_crop_top_surface_seed0_99/
outputs/h200_exp_b_liftpeg_box_center_depth_seed0_99/
outputs/h200_exp_b_liftpeg_crop_median_seed0_99/
outputs/h200_exp_b_liftpeg_crop_top_surface_seed0_99/
outputs/h200_exp_b_liftpeg_object_query_crop_top_surface_seed0_99/
outputs/h200_exp_b_liftpeg_oracle_object_pose_seed0_99/
```
