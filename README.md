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
