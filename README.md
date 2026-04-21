# Query-to-Grasp

Research prototype for language-queryable 3D semantic target retrieval and grasping in simulation.

Current implemented stages:

- Phase 1: ManiSkill observation export and Open3D point cloud generation.
- Phase 2A: single-view semantic retrieval baseline with query parsing, detector adapter, CLIP reranking, and 2D-to-3D lifting.
- Minimal pick stage: safe placeholder pick executor integrated with the single-view pipeline.

Lightweight smoke run:

```powershell
python scripts/run_single_view_pick.py --query "red cube" --detector-backend mock --mock-box-position center --skip-clip --output-dir outputs/colab_pick_smoke
```

## 🤖 进阶应用：结合 OpenMythos 进行具身智能闭环测试

本项目不仅可以作为独立的 3D 目标检索与抓取系统，还是评估底层大语言模型（LLM）推理能力的绝佳下游执行终端（Downstream Executor）。

如果你正在研究或训练像 OpenMythos（一个具有强大深层推理能力的 Recurrent-Depth Transformer 开源架构）这样的大模型，你可以将本项目作为它的物理仿真测试沙盒。

## 🧠 系统角色分配：“大脑”与“手眼”

在面对复杂的现实场景时，简单的直接查询（如 "red cube"）往往不够用。我们可以将两个项目结合，构建一个完整的具身智能（Embodied AI）流水线：

OpenMythos（大脑）：负责高级语意理解和逻辑推理。它接收用户模糊的、多步骤的自然语言指令，通过其深层循环架构推理出具体的执行目标。

Query-to-Grasp（手眼）：负责具体的 3D 视觉感知与动作执行。它接收来自大模型精炼后的 Query，在 ManiSkill 仿真环境中完成目标检测（Detector + CLIP）与机械臂抓取（Pick Executor）。

## ⚙️ 联动工作流示例 (Pipeline)

复杂指令输入：用户给出模糊指令，如 "桌子上洒了水，帮我找个东西清理一下。"

OpenMythos 深度推理：模型理解语境，推理出需要吸水物品（海绵），并输出规范化的精确词汇：`{"target": "blue sponge"}`。

Query-to-Grasp 抓取执行：将提取出的 "blue sponge" 传入本项目的执行管线。

测试脚本伪代码：

```bash
# 1. 调用 OpenMythos 进行逻辑推理，提取具体的抓取目标
TARGET_QUERY=$(python run_openmythos_inference.py --prompt "桌子上洒了水，帮我找个东西清理一下" | grep -oP '(?<="target": ")[^"]*')
# 假设输出为 "blue sponge"

echo "OpenMythos 决策抓取目标: $TARGET_QUERY"

# 2. 将目标传递给 Query-to-Grasp 进行 3D 检测与抓取测试
python scripts/run_single_view_pick.py \
    --query "$TARGET_QUERY" \
    --detector-backend mock \
    --mock-box-position center \
    --skip-clip \
    --output-dir outputs/openmythos_eval_run
```

## 📊 为什么使用 Query-to-Grasp 测试 OpenMythos？

将 LLM 接入机器人仿真环境是评估其真实世界泛化能力的最佳方式。通过观察 Query-to-Grasp 是否成功抓取了正确的物体，你可以直观且自动化地量化评估 OpenMythos 模型在“复杂指令遵循”、“常识推理”以及“具身决策”方面的能力表现。
