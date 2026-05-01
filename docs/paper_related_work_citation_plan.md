# Related Work Citation Plan

This file is a planning artifact for the conference manuscript. The core
BibTeX entries currently used by `paper/main.tex` have been cleaned in
`paper/references.bib` using primary proceedings/arXiv/OpenReview sources;
additional optional references below still need exact metadata verification
before they are added.

## Verified Core Bibliography

The current LaTeX draft cites the following verified core entries:

- `radford2021learning`: CLIP, ICML 2021 / PMLR 139.
- `liu2023groundingdino`: Grounding DINO, ECCV 2024.
- `shridhar2022cliport`: CLIPort, CoRL 2021 / PMLR 164.
- `shridhar2022peract`: Perceiver-Actor, CoRL 2022 / PMLR 205.
- `ahn2022saycan`: SayCan, CoRL 2022 / PMLR 205.
- `jiang2023vima`: VIMA, ICML 2023 / PMLR 202.
- `brohan2022rt1`: RT-1, RSS 2023.
- `brohan2023rt2`: RT-2, arXiv 2023.
- `gu2023maniskill2`: ManiSkill2, ICLR 2023.

## Open-Vocabulary Grounding

Purpose in paper:

- Position HF GroundingDINO as an off-the-shelf open-vocabulary 2D proposal
  layer.
- Position CLIP as an optional reranking and confidence module, not as the
  primary measured improvement in the current results.
- Explain that Query-to-Grasp studies the 2D-to-3D and action-target gap after
  language-conditioned 2D proposals are available.

Candidate references to verify:

- CLIP-style image-text pretraining.
- GroundingDINO or Grounding DINO open-set grounding.
- GLIP-style grounded language-image pretraining.
- OWL-ViT / OWL-style open-vocabulary detection if comparing detector families.
- Segment Anything only if the final paper discusses mask prompting or
  segmentation extensions.

Search terms:

- `CLIP contrastive language image pretraining paper`
- `Grounding DINO open set object detection grounding paper`
- `GLIP grounded language image pretraining paper`
- `OWL-ViT open vocabulary object detection paper`

Optional additions still needing verification:

- GLIP-style grounded language-image pretraining.
- OWL-ViT / OWL-style open-vocabulary detection.
- Segment Anything only if the final paper discusses mask prompting or
  segmentation extensions.

## Language-Conditioned Manipulation

Purpose in paper:

- Contrast Query-to-Grasp with systems that map language directly to robot
  actions, learned policies, or long-horizon plans.
- Emphasize that this project contributes an inspectable retrieval-to-3D-target
  and simulated-pick diagnostic layer, not a new learned policy.

Candidate references to verify:

- CLIPort-style language-conditioned pick-and-place.
- PerAct-style language-conditioned manipulation with voxel or action-value
  representations.
- SayCan-style language-model planning grounded in affordances.
- VIMA-style multimodal prompted manipulation.
- RT-1 / RT-2 style robot policy learning, only as broad context rather than a
  direct baseline.

Search terms:

- `language conditioned robotic manipulation CLIPort`
- `PerAct language conditioned manipulation`
- `SayCan language model affordance robot manipulation`
- `VIMA multimodal prompts robot manipulation`
- `RT-1 RT-2 robotic manipulation language vision action`

Optional additions still needing verification:

- Whether to cite additional robot-learning policy papers beyond CLIPort,
  PerAct, SayCan, VIMA, RT-1, and RT-2.
- Exact claims about policy learning and real-robot validation for any optional
  additions.

## RGB-D and Multi-View Object Memory

Purpose in paper:

- Situate the object-memory module among RGB-D fusion, multi-view perception,
  object-centric mapping, and open-vocabulary 3D scene understanding.
- Make clear that Query-to-Grasp uses a lightweight task-specific memory rather
  than a full SLAM or neural 3D reconstruction system.

Candidate references to verify:

- RGB-D fusion and point-cloud based manipulation perception.
- Object-centric 3D maps or semantic object memory for robotics.
- Open-vocabulary 3D scene understanding and 3D language grounding systems.
- Any ManiSkill or simulation benchmark references used for RGB-D manipulation
  evaluation.

Search terms:

- `RGB-D object memory robot manipulation`
- `multi-view object-centric mapping manipulation`
- `open vocabulary 3D scene understanding language grounding`
- `semantic object map RGB-D robot`

Verification still needed:

- The closest prior work for persistent object hypotheses with label votes and
  view support.
- Whether to compare against full 3D reconstruction systems or keep the
  contrast at the object-memory level.

## Active Perception and Re-Observation Diagnostics

Purpose in paper:

- Position the closed-loop extra-view path as a diagnostic re-observation
  baseline.
- Avoid overclaiming learned active view planning, real camera motion, or
  physical exploration.
- Explain why uncertainty reduction must be evaluated separately from downstream
  pick success.

Candidate references to verify:

- Next-best-view and active perception for manipulation.
- Uncertainty-aware perception or view selection in robotic manipulation.
- Closed-loop visual servoing or re-observation baselines if used for contrast.
- Diagnostic benchmark papers that separate perception confidence from task
  execution.

Search terms:

- `active perception next best view robotic manipulation`
- `uncertainty aware view selection robot manipulation`
- `closed-loop perception re-observation manipulation`
- `next best view object manipulation RGB-D`

Verification still needed:

- Which active perception references evaluate downstream manipulation success.
- Whether any cited method uses virtual views in simulation, to compare fairly
  with Query-to-Grasp.

## Simulation and Grasp Evaluation

Purpose in paper:

- Justify ManiSkill as the simulation environment and clarify the metric split
  between `pick_success` and environment `task_success`.
- Situate `sim_topdown` as a simple diagnostic controller rather than a grasp
  synthesis contribution.

Candidate references to verify:

- ManiSkill benchmark/platform papers and task documentation.
- Grasp success metrics in simulated manipulation benchmarks.
- Simple scripted controller baselines for pick tasks, if a close comparison is
  needed.

Search terms:

- `ManiSkill benchmark paper`
- `simulated robotic manipulation benchmark grasp success`
- `scripted controller baseline robotic manipulation simulation`

Verification still needed:

- Correct ManiSkill version citation for the tasks used here.
- Whether venue formatting should cite simulator docs, benchmark paper, or both.

## Project-Specific Claim Boundaries

Claims already supported by repository artifacts:

- Correcting the OpenCV-to-OpenGL camera convention sharply reduces cross-view
  memory spread.
- Fused memory grasp points enable `PickCube-v1` multi-view and closed-loop
  simulated pick success of `1.0000` on the current full ambiguity validation.
- `StackCube-v1` is a pick-only compatibility diagnostic: tabletop reaches
  `0.6200`, closed-loop reaches `0.5200`, and task success remains `0.0000`.

Claims that require more evidence before submission:

- General manipulation beyond pick/lift.
- Real-robot execution.
- Learned active perception or learned grasping.
- Robust relation-heavy language grounding.
- Full non-oracle StackCube stacking completion.
