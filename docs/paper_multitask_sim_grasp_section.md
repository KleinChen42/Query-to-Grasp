# Multi-Task Simulated Grasp Evaluation

Retrieval accuracy alone is not enough for a language-queryable manipulation
pipeline. A detector can identify the right object category, and a multi-view
memory can select a plausible 3D object hypothesis, while the resulting point is
still unsuitable as a grasp target. We therefore evaluate the final selected
3D target with an opt-in ManiSkill controller, `sim_topdown`, that attempts a
simple top-down grasp and lift from the selected world-coordinate target. This
section reports simulated pick success separately from ManiSkill task success:
`pick_success` measures whether the object is grasped/lifted according to the
task's grasp signal, while `task_success` preserves the environment's full task
completion condition. This distinction is important for `StackCube-v1`, where a
valid pick of cubeA is not the same as stacking cubeA onto cubeB.

The strongest manipulation result is on `PickCube-v1`. After propagating
per-view grasp candidates into fused 3D memory, refined multi-view execution
uses `memory_grasp_world_xyz` as the downstream pick target while preserving the
semantic fused object center for retrieval and selection. In the full ambiguity
validation, both tabletop_3 and closed-loop multi-view modes complete `55/55`
runs with `0` child failures and `pick_success_rate = 1.0000` in no-CLIP and
with-CLIP settings. Closed-loop re-observation still has diagnostic value: it
reduces still-needed uncertainty from `0.4182` to `0.0909` without CLIP and
from `0.3818` to `0.0545` with CLIP, while maintaining perfect simulated pick
success. These results are reported in
`outputs/h200_60071_multiview_memory_grasp_point_full_ambiguity_seed01234`.

We also test whether the same query-to-grasp chain transfers to a second task,
`StackCube-v1`, using the query `red cube`. This experiment is deliberately
framed as pick-only compatibility rather than stacking-task completion. The
single-view chain reaches `pick_success_rate = 1.0000` for seeds `0..19`, but
multi-view execution exposes a task-dependent target-source preference. On
`PickCube-v1`, refined multi-view picking benefits from the fused memory grasp
point. On `StackCube-v1`, the semantic fused selected-object center is the more
reliable pick target. The accepted task-aware guard therefore uses
`task_guard_selected_object_world_xyz` only for `StackCube-v1`, leaving
`PickCube-v1` refined picks on `memory_grasp_world_xyz`.

The expanded StackCube validation covers seeds `0..49` for tabletop_3 and
closed-loop modes, in no-CLIP and with-CLIP settings. All four StackCube modes
complete `50/50` runs with `0` child failures and `grasp_attempted_rate =
1.0000`. Tabletop_3 reaches `pick_success_rate = 0.6200` in both no-CLIP and
with-CLIP modes. Closed-loop reaches `pick_success_rate = 0.5200` in both modes.
The PickCube regression remains `3/3` successful and continues to use
`memory_grasp_world_xyz`, confirming that the StackCube guard does not weaken
the main PickCube result. The expanded StackCube artifacts are in
`outputs/h200_60071_stackcube_task_guard_expanded_seed0_49`.

The query-driven placement bridge then asks whether these query-derived cubeA
targets can support the full StackCube task when the destination cubeB pose is
privileged. With `sim_pick_place`, single-view query targets reach
`pick_success_rate = 0.8800`, `place_success_rate = 0.7200`, and
`task_success_rate = 0.7200` in both no-CLIP and with-CLIP modes over 50 seeds.
Multi-view tabletop_3 reaches `0.6200` pick success and `0.5200` task success,
while closed-loop reaches `0.5200` pick success and `0.4800` task success. This
is a partial task-success bridge, not a full language-conditioned stacking
claim: cubeA is query-derived, but cubeB remains an oracle placement target.
The bridge artifacts are in
`outputs/h200_60071_query_stackcube_place_bridge_seed0_49`.

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
target. This explains why closed-loop improves diagnostic confidence but does
not improve StackCube pick success. The failure taxonomy is reported in
`outputs/h200_60071_stackcube_task_guard_expanded_seed0_49/reports/stackcube_guard_failure_report.md`.

Taken together, these experiments show that the pipeline has moved beyond
perception-only retrieval: language-selected RGB-D targets can drive a real
simulated controller and produce stable downstream grasp metrics. The evidence
is strongest for `PickCube-v1`, where full-query multi-view and closed-loop
simulated pick success reaches `1.0000`. `StackCube-v1` provides a stricter
cross-task diagnostic: it validates pick-only transfer, shows that query-derived
cubeA targets can support oracle-place task success, exposes task-dependent
grasp target preferences, and motivates future work on non-oracle placement
target grounding and closed-loop association. We do not claim real-robot
execution, robust general manipulation, or fully non-oracle StackCube stacking
completion in the current version.
