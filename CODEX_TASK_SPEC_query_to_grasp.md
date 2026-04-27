# CODEX TASK SPEC — Query-to-Grasp via Confidence-Aware 3D Semantic Fusion

## 1. Role
You are an autonomous coding agent building a research-grade but fast-to-ship prototype for a conference-style demo and paper.

Your goal is to implement a **language-queryable 3D semantic target retrieval system for grasp preparation** in simulation, with strong AI involvement and a clean path to experiments, diagnostic reports, and a paper draft.

Current scope reset, 2026-04-27:
- The stable paper v1 is a perception/retrieval/re-observation system, not a full grasp-control paper.
- `SafePlaceholderPickExecutor` is expected in the current mainline and must not be reported as real grasp success.
- The minimal simulated grasp execution baseline now exists as an opt-in `sim_topdown` path.
- The next quality-upgrade phase is grasp-point/control diagnosis: explain why exact/oracle targets grasp while broad compact queries often lift to non-grasp-confirmed target centers.

You must optimize for:
1. **Fast end-to-end demo delivery**
2. **Minimal training / mostly inference-based AI modules**
3. **Clear modular architecture**
4. **Reproducibility in Colab Pro and local Linux**
5. **Research value**, not just a toy script

---

## 2. Project title
**Query-to-Grasp via Confidence-Aware 3D Semantic Fusion**

Optional long title:
**Language-Queryable 3D Semantic Retrieval and Confidence-Aware Target Grasping in Cluttered Scenes**

---

## 3. Problem definition
Given a natural-language query such as:
- `red cube`
- `blue mug`
- `banana`
- `small green block near the cup`

The current paper-v1 system should:
1. Parse the query into structured semantic constraints
2. Observe a ManiSkill scene using RGB + depth (+ segmentation if available)
3. Propose candidate 2D object regions using an open-vocabulary detector
4. Re-rank candidates with an image-text similarity model
5. Lift candidate detections into 3D using depth / point cloud
6. Fuse multi-view evidence into a persistent 3D object memory
7. If confidence is low, trigger one or more extra viewpoints
8. Select the best target in 3D
9. Validate the selected target with a safe placeholder pick executor
10. Return visualizations, logs, benchmark rows, reports, and paper artifacts

The next grasp-upgrade system should additionally:
1. Convert the selected 3D target into a minimal scripted or planner-assisted simulated grasp attempt
2. Report real ManiSkill grasp-attempt and pick-success metrics separately from retrieval metrics
3. Compare single-view, multi-view, and closed-loop re-observation on downstream grasp success

---

## 4. Research hypothesis
A **confidence-aware multi-view 3D semantic fusion pipeline** will improve language-conditioned **3D target retrieval and grasp-preparation diagnostics** in cluttered scenes compared with a single-view or non-fusion baseline.

Follow-up hypothesis after adding minimal simulated grasp control:

> Better language-conditioned 3D target retrieval and re-observation should improve downstream simulated grasp-attempt success, even with a simple non-learned grasp executor.

---

## 5. Scope constraints (important)
### Must do
- Single robot arm only
- Single-tabletop manipulation tasks only
- Cluttered scenes with a limited number of objects
- Natural language object queries
- Multi-view 3D fusion
- Confidence-based re-observation policy
- Evaluation scripts and ablations
- Paper-ready reports and artifact packs

### Next-stage must do for a stronger grasp paper
- Minimal simulated scripted or planner-assisted grasp attempt
- Downstream `grasp_attempted`, `pick_success`, and query-to-grasp metrics
- Ablations showing whether retrieval/re-observation improves grasp outcomes

### Must NOT do in v1
- Real robot deployment
- Real low-level robot-control claim
- Treat placeholder pick success as real grasp success
- Web demo unless the paper/debug artifacts are already frozen
- Large-scale model training
- Full relation-heavy language reasoning
- Multi-step task chains
- Multi-robot setup
- Complex motion planning research
- Novel grasp synthesis training

### Principle
**Prefer a strong, clean, fully working baseline over a large but incomplete system.**

---

## 6. Required technology stack
Use these as the primary stack unless there is a very strong implementation reason to swap:

- **Simulation / robot environment:** ManiSkill
- **3D processing / point clouds / RGBD:** Open3D
- **Open-vocabulary proposal model:** GroundingDINO
- **Image-text reranking:** CLIP or OpenCLIP
- **Optional web demo:** Gradio Blocks + Model3D
- **Language query parser:** lightweight LLM prompt wrapper with deterministic fallback rules
- **Core language / infra:** Python 3.10+
- **Deep learning runtime:** PyTorch

If a dependency is changed, explain why and preserve the same behavior and interfaces.

---

## 7. System design requirements
Implement the following modules.

### 7.1 Query parser
File: `src/perception/query_parser.py`

Input:
- free-form natural language query

Output structured dict:
```python
{
  "raw_query": "pick the small red cube near the mug",
  "target_name": "cube",
  "attributes": ["small", "red"],
  "relations": [{"type": "near", "object": "mug"}],
  "synonyms": ["cube", "block"],
  "normalized_prompt": "small red cube"
}
```

Requirements:
- Provide both:
  - `parse_query_llm()`
  - `parse_query_rules()`
- Use rules as fallback when LLM parsing is unavailable
- Keep implementation deterministic where possible

---

### 7.2 Scene observation interface
File: `src/env/maniskill_env.py`

Responsibilities:
- Create and manage ManiSkill env
- Reset scenes
- Return RGB, depth, segmentation, intrinsics, extrinsics, robot state
- Support fixed and multi-view observation capture
- Save debug frames to disk

Required interface:
```python
class ManiSkillScene:
    def reset(self, seed: int | None = None) -> dict: ...
    def get_observation(self, camera_name: str | None = None) -> dict: ...
    def get_multiview_observations(self, view_ids: list[str]) -> list[dict]: ...
    def execute_pick(self, target_xyz: np.ndarray) -> dict: ...
```

---

### 7.3 Open-vocabulary proposal generation
File: `src/perception/grounding_dino.py`

Responsibilities:
- Accept an image and text prompt
- Return candidate bounding boxes with confidence scores
- Support thresholding and top-k filtering
- Save visualization overlays

Required output schema:
```python
@dataclass
class DetectionCandidate:
    box_xyxy: np.ndarray
    det_score: float
    phrase: str
    image_crop_path: str | None = None
```

Function:
```python
def detect_candidates(image: np.ndarray, text_prompt: str, box_threshold: float, text_threshold: float, top_k: int) -> list[DetectionCandidate]:
    ...
```

---

### 7.4 CLIP reranking
File: `src/perception/clip_rerank.py`

Responsibilities:
- Crop candidate image regions
- Score each crop against the normalized query
- Optionally score against attribute-expanded prompts
- Return reranked candidates

Required output:
```python
@dataclass
class RankedCandidate:
    box_xyxy: np.ndarray
    det_score: float
    clip_score: float
    fused_2d_score: float
    phrase: str
```
```

Fuse detection and CLIP scores using configurable weights.

---

### 7.5 2D-to-3D lifting
File: `src/perception/mask_projector.py`

Responsibilities:
- Convert candidate region to 3D points using depth and camera intrinsics
- Estimate target center in camera/world frame
- Optionally use segmentation masks when available

Required output:
```python
@dataclass
class Candidate3D:
    world_xyz: np.ndarray
    camera_xyz: np.ndarray
    num_points: int
    depth_valid_ratio: float
    point_cloud_path: str | None = None
```
```

---

### 7.6 Multi-view 3D object memory
File: `src/memory/object_memory_3d.py`

Responsibilities:
- Maintain persistent object hypotheses in world coordinates
- Merge observations that correspond to the same physical object
- Track semantic labels, scores, views, and geometry stats

Each memory object should track at least:
```python
{
  "object_id": "obj_0003",
  "world_xyz": [x, y, z],
  "label_votes": {"red cube": 2.1, "cube": 0.8},
  "det_scores": [...],
  "clip_scores": [...],
  "view_ids": [...],
  "num_observations": 3,
  "geometry_confidence": 0.74,
  "semantic_confidence": 0.81,
  "overall_confidence": 0.79,
  "point_cloud_path": "..."
}
```

---

### 7.7 Confidence-aware fusion
File: `src/memory/fusion.py`

Implement a configurable fusion score.

Suggested default:
```python
S = alpha * det_score + beta * clip_score + gamma * view_score + delta * consistency_score + epsilon * geometry_score
```

Requirements:
- All weights must live in config
- Include ablation-friendly switches
- Log all score terms for debugging
- Include confidence calibration comments in code

---

### 7.8 Re-observation policy
File: `src/policy/reobserve_policy.py`

Purpose:
Trigger extra views if the system is uncertain.

Initial implementation should be rule-based, not learned.

Trigger conditions may include:
- top-1 and top-2 overall confidence gap too small
- too few valid 3D points
- high spatial variance across views
- low semantic confidence
- strong occlusion or partial crop heuristics

Required output:
```python
@dataclass
class ReobserveDecision:
    should_reobserve: bool
    reason: str
    suggested_view_ids: list[str]
```

---

### 7.9 Target selector
File: `src/policy/target_selector.py`

Responsibilities:
- Choose the final target object from memory
- Return full trace for why it was selected

Required behavior:
- deterministic ordering
- ties broken by confidence, then number of views, then geometry stability

---

### 7.10 Grasp execution
File: `src/manipulation/pick_executor.py`

Current paper-v1 requirements:
- validate selected target coordinates
- return structured placeholder pick output
- keep `pick_success=False` explicit and expected
- do not claim robot-control success from the placeholder executor

Next-stage simulated grasp baseline:
- simple scripted pick pipeline is acceptable
- move above target
- descend
- close gripper
- lift
- return `grasp_attempted`, `pick_success`, and trajectory summary

No need to do grasp-learning research in v1.

---

### 7.11 Web demo
File: `src/demo/app.py`

The demo must support:
- query text input
- scene selection / seed selection
- run single-view mode
- run multi-view mode
- show RGB image with 2D proposals
- show selected 3D target via `.ply` or equivalent
- show final result video / gif / image sequence
- show debug text explaining confidence and re-observation decisions

The UI must be organized, minimal, and demo-friendly.

---

## 8. Project directory structure
Use this exact structure unless there is a strong reason to extend it:

```text
query_to_grasp/
  README.md
  requirements.txt
  pyproject.toml
  configs/
    base.yaml
    demo.yaml
    eval.yaml
  data/
    logs/
    videos/
    pointclouds/
    debug_frames/
  src/
    env/
      maniskill_env.py
      camera_utils.py
    perception/
      query_parser.py
      grounding_dino.py
      clip_rerank.py
      mask_projector.py
    memory/
      object_memory_3d.py
      fusion.py
    policy/
      reobserve_policy.py
      target_selector.py
    manipulation/
      pick_executor.py
    demo/
      app.py
      vis.py
    eval/
      metrics.py
      run_benchmark.py
    utils/
      io.py
      logging_utils.py
      config.py
      seeds.py
  notebooks/
    01_env_debug.ipynb
    02_single_view.ipynb
    03_multiview_fusion.ipynb
  tests/
    test_query_parser.py
    test_fusion.py
    test_memory.py
```

---

## 9. Development phases
### Phase 1 — Environment and I/O
Deliverables:
- ManiSkill environment setup
- RGB/depth/segmentation acquisition
- Save sample observations
- Open3D point-cloud export and visualization

### Phase 2 — Single-view semantic retrieval
Deliverables:
- query parser
- GroundingDINO proposal pipeline
- CLIP reranking
- 2D overlay visualization
- 2D-to-3D target center estimation

### Phase 3 — End-to-end single-view pick
Deliverables:
- target selection
- safe placeholder pick executor
- retrieval and target-validation logging
- per-run JSON artifacts

### Phase 4 — Multi-view fusion
Deliverables:
- object memory
- fusion scoring
- view aggregation
- persistent 3D object hypotheses

### Phase 5 — Re-observation
Deliverables:
- uncertainty rules
- additional view capture
- improved target selection after re-observe

### Phase 6 — Demo and evaluation
Deliverables:
- benchmark runner
- ablation configs
- CSV / JSON result dumps
- plots and tables
- paper reports and figure pack

### Phase 7 - Minimal simulated grasp execution baseline
Deliverables:
- scripted or planner-assisted ManiSkill grasp attempt from selected 3D target
- `grasp_attempted`, `pick_success`, and trajectory summary fields
- query-to-grasp benchmark mode that keeps retrieval and grasp metrics separate
- ablation showing whether multi-view/re-observation improves grasp outcomes

---

## 10. Required evaluation metrics
Implement these in `src/eval/metrics.py`.

### Core metrics
- `query_grounding_success`
- `target_localization_error_3d`
- `target_retrieval_success`
- `mean_num_views_used`
- `mean_runtime_seconds`

### Next-stage grasp metrics
- `grasp_attempted_rate`
- `grasp_success_rate`
- `end_to_end_query_to_grasp_success`

### Useful diagnostics
- detector top-1 accuracy
- reranker improvement over detector-only
- memory merge precision / error count
- re-observation trigger rate
- confidence calibration plots if feasible

---

## 11. Required experiments
At minimum implement these 4 comparisons:

1. **Single-view vs Multi-view**
2. **Detector-only vs Detector + CLIP reranking**
3. **No fusion vs Confidence-aware fusion**
4. **Direct pick vs Re-observe then pick**

Optional fifth:
5. **With segmentation-assisted lifting vs depth-only lifting**

All experiments must produce machine-readable logs and a summary table.

---

## 12. Coding rules
### General rules
- Use Python type hints everywhere
- Use dataclasses for structured outputs
- Keep functions small and composable
- No hidden global state
- Put thresholds and weights in config files
- Add logging to every critical module
- Fail loudly with actionable error messages

### Reproducibility rules
- set seeds when possible
- log environment ids and config snapshots
- save per-run output folders with timestamps

### Implementation style
- Avoid giant monolithic scripts
- Prefer modular files and tested interfaces
- Every module should have a minimal `if __name__ == "__main__":` debug entry if useful

---

## 13. Testing requirements
Write lightweight tests for:
- query parsing
- fusion score calculation
- object memory merge logic
- deterministic target selection

Use synthetic arrays / mock detections where possible.
Do not make tests depend on large model downloads unless explicitly marked integration-only.

---

## 14. Performance / resource constraints
Assume the primary development environment is:
- **Colab Pro**
- single-GPU session
- limited VRAM budget
- need for fast restart and checkpoint-free inference workflows

Therefore:
- prefer inference-first pipeline
- avoid training-heavy components in v1
- cache model loads
- allow CPU fallback for non-critical paths
- make notebooks restart-friendly

---

## 15. Configuration requirements
Use YAML configs.

`configs/base.yaml` should contain:
- environment id
- camera list / view ids
- detector model config
- thresholds
- fusion weights
- re-observation thresholds
- demo settings
- eval settings
- output directory roots

All magic numbers must be moved into config.

---

## 16. Logging and artifacts
Every run should save:
- config snapshot
- raw query
- parsed query json
- 2D proposal visualizations
- rerank scores
- 3D point cloud artifact
- memory state json
- re-observation decision json
- final execution result json
- final video or image sequence

Directory convention:
```text
outputs/<timestamp>_<short_tag>/
```

---

## 17. Deliverable checklist
The final project must include:
- runnable repo
- README with setup + quickstart
- one-click or minimal-command demo launch
- evaluation runner
- sample config files
- sample outputs
- at least one benchmark result table
- at least one demo video/gif
- clean code structure suitable for paper support material

---

## 18. README requirements
The README must include:
1. project overview
2. architecture diagram (ASCII or Mermaid acceptable)
3. install steps
4. Colab usage notes
5. quickstart commands
6. demo launch instructions
7. evaluation instructions
8. known limitations
9. future work

---

## 19. What to do first
When starting implementation, follow this exact order:

1. create repo skeleton
2. implement config loading and logging
3. implement ManiSkill env wrapper
4. dump one RGB/depth/segmentation sample
5. generate one Open3D point cloud
6. implement detector wrapper
7. implement CLIP reranker
8. implement 2D-to-3D lifting
9. run one single-view query-to-target result
10. implement scripted pick
11. add multi-view memory
12. add fusion logic
13. add re-observation policy
14. build Gradio demo
15. add evaluation scripts

Do not jump ahead to polishing the demo before the single-view pipeline works end to end.

---

## 20. Output contract for the coding agent
When you work on this project, do the following:

### Before coding
- restate the current subtask
- list files you will create or modify
- state assumptions clearly

### During coding
- implement only the current milestone unless dependencies force otherwise
- keep patches minimal and focused
- explain any architectural deviations

### After coding
- summarize what changed
- list commands to run
- list expected outputs
- list next recommended step

---

## 21. Preferred milestone-by-milestone execution format
Use this format in every coding response:

```text
Subtask
Assumptions
Files changed
Implementation notes
How to run
Expected result
Next step
```

---

## 22. Acceptance criteria for v1
The v1 system is considered successful if it can:

1. accept a query like `red cube`
2. retrieve candidate detections from one or more views
3. choose a 3D target hypothesis
4. optionally run a safe placeholder pick validator without claiming real grasp success
5. log benchmark rows, summaries, selection traces, re-observation diagnostics, and paper reports
6. run ablations comparing single-view, multi-view, CLIP/no-CLIP, and re-observation variants
7. clearly document limitations around placeholder grasp execution

The stronger grasp-paper stage is considered successful if it can:

1. convert the selected 3D target into a real simulated grasp attempt
2. execute that attempt in ManiSkill with a scripted or planner-assisted controller
3. report downstream pick success separately from retrieval success
4. show whether multi-view/re-observation improves grasp outcomes

Current status, 2026-04-27:

- Items 1-3 are implemented for the opt-in single-view `sim_topdown` path.
- H200 oracle and exact `red cube` smoke tests can grasp successfully.
- The first compact query benchmark is stable but low-success (`pick_success_rate = 0.1000`), so item 4 should wait until target-center/grasp-point diagnosis is complete.

---

## 23. Nice-to-have extensions (not required for v1)
- relation-aware language grounding
- VLM-based failure explanation
- learned re-observation policy
- learned grasp ranking
- minimal simulated scripted grasp baseline
- segment-anything refinement
- simple sim-to-real export utilities
- paper figure generation scripts

---

## 24. Important project philosophy
This project is **not** about inventing a huge new model.
It is about building a **strong embodied AI system** that combines:
- language grounding
- open-vocabulary perception
- 3D geometric reasoning
- confidence-aware decision making
- robotic execution

The best implementation is the one that is:
- clean
- fast to demo
- easy to evaluate
- convincing in a paper

---

## 25. Immediate next action
Start with:
- `src/utils/config.py`
- `src/utils/logging_utils.py`
- `src/env/maniskill_env.py`
- `notebooks/01_env_debug.ipynb`

Goal of the first milestone:
**Load a ManiSkill scene, export RGB/depth/segmentation, and save one Open3D point cloud to disk.**
