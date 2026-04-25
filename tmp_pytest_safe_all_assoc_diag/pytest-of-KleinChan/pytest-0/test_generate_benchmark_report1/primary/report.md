# Benchmark Report

- Benchmark directory: `E:\CODE\Query-to-Grasp\tmp_pytest_safe_all_assoc_diag\pytest-of-KleinChan\pytest-0\test_generate_benchmark_report1\primary`
- Timestamp: 2026-04-20T00:00:00+00:00
- Total runs: 1
- Unique queries: red cube
- Detector backend: mock
- Skip CLIP: True
- Depth scale: 1000.0

## Aggregate Metrics

| Metric | Value |
| --- | ---: |
| total_runs | 1 |
| mean_raw_num_detections | 2 |
| mean_num_detections | 2 |
| mean_num_ranked_candidates | 1 |
| mean_num_3d_points | 10 |
| fraction_with_3d_target | 1 |
| pick_success_rate | 0.2500 |
| fraction_top1_changed_by_rerank | 0 |
| mean_runtime_seconds | 1.5000 |
| pick_stage_counts | `{"placeholder_not_executed": 1}` |

## Per-Query Breakdown

| Query | total_runs | mean_raw_num_detections | mean_num_detections | mean_num_ranked_candidates | mean_num_3d_points | fraction_with_3d_target | pick_success_rate | fraction_top1_changed_by_rerank | mean_runtime_seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| red cube | 1 | 2 | 2 | 1 | 10 | 1 | 0.2500 | 0 | 1.5000 |

## Comparison

- Compare benchmark directory: `E:\CODE\Query-to-Grasp\tmp_pytest_safe_all_assoc_diag\pytest-of-KleinChan\pytest-0\test_generate_benchmark_report1\secondary`

| Metric | Primary | Secondary | Delta |
| --- | ---: | ---: | ---: |
| total_runs | 1 | 1 | 0 |
| mean_raw_num_detections | 2 | 1 | 1 |
| mean_num_detections | 2 | 1 | 1 |
| mean_num_ranked_candidates | 1 | 1 | 0 |
| mean_num_3d_points | 10 | 10 | 0 |
| fraction_with_3d_target | 1 | 1 | 0 |
| pick_success_rate | 0.2500 | 0 | 0.2500 |
| fraction_top1_changed_by_rerank | 0 | 0 | 0 |
| mean_runtime_seconds | 1.5000 | 1.5000 | 0 |

## Comparison Takeaway

Comparison takeaway: Primary improves mean_raw_num_detections, mean_num_detections, pick_success_rate over secondary.
