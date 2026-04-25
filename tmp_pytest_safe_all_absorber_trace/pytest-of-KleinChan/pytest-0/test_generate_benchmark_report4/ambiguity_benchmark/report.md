# Benchmark Report

- Benchmark directory: `E:\CODE\Query-to-Grasp\tmp_pytest_safe_all_absorber_trace\pytest-of-KleinChan\pytest-0\test_generate_benchmark_report4\ambiguity_benchmark`
- Timestamp: 2026-04-20T00:00:00+00:00
- Total runs: 1
- Unique queries: object
- Detector backend: mock
- Skip CLIP: True
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
| mean_runtime_seconds | 1.5000 |
| pick_stage_counts | `{"placeholder_not_executed": 1}` |

## Ambiguity Conclusion

Current ambiguity benchmark still does not provide useful reranking headroom.

## Per-Query Breakdown

| Query | total_runs | mean_raw_num_detections | mean_num_detections | mean_num_ranked_candidates | mean_num_3d_points | fraction_with_3d_target | pick_success_rate | fraction_top1_changed_by_rerank | mean_runtime_seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| object | 1 | 1 | 1 | 1 | 10 | 1 | 0 | 0 | 1.5000 |
