# Benchmark Report

- Benchmark directory: `E:\CODE\Query-to-Grasp\tmp_pytest_safe_all_post_selection_continuity\pytest-of-KleinChan\pytest-0\test_generate_benchmark_report2\with_clip`
- Timestamp: 2026-04-20T00:00:00+00:00
- Total runs: 1
- Unique queries: red cube
- Detector backend: mock
- Skip CLIP: False
- Depth scale: 1000.0

## Aggregate Metrics

| Metric | Value |
| --- | ---: |
| total_runs | 1 |
| mean_raw_num_detections | 1 |
| mean_num_detections | 1 |
| mean_num_ranked_candidates | 1 |
| mean_num_3d_points | 10 |
| fraction_with_3d_target | 1 |
| pick_success_rate | 0 |
| fraction_top1_changed_by_rerank | 0 |
| mean_runtime_seconds | 2 |
| pick_stage_counts | `{"placeholder_not_executed": 1}` |

## Per-Query Breakdown

| Query | total_runs | mean_raw_num_detections | mean_num_detections | mean_num_ranked_candidates | mean_num_3d_points | fraction_with_3d_target | pick_success_rate | fraction_top1_changed_by_rerank | mean_runtime_seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| red cube | 1 | 1 | 1 | 1 | 10 | 1 | 0 | 0 | 2 |

## Comparison

- Compare benchmark directory: `E:\CODE\Query-to-Grasp\tmp_pytest_safe_all_post_selection_continuity\pytest-of-KleinChan\pytest-0\test_generate_benchmark_report2\no_clip`

| Metric | Primary | Secondary | Delta |
| --- | ---: | ---: | ---: |
| total_runs | 1 | 1 | 0 |
| mean_raw_num_detections | 1 | 1 | 0 |
| mean_num_detections | 1 | 1 | 0 |
| mean_num_ranked_candidates | 1 | 1 | 0 |
| mean_num_3d_points | 10 | 10 | 0 |
| fraction_with_3d_target | 1 | 1 | 0 |
| pick_success_rate | 0 | 0 | 0 |
| fraction_top1_changed_by_rerank | 0 | 0 | 0 |
| mean_runtime_seconds | 2 | 1 | 1 |

## Comparison Takeaway

CLIP ablation takeaway: Primary uses CLIP, but candidate multiplicity is low (mean_raw_num_detections=1) and reranking never changes top-1. Prioritize candidate generation or ambiguity before tuning CLIP.
