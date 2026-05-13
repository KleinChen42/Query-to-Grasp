#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-outputs}"

echo "===TOP_OUTPUTS_WITH_BENCHMARK_SUMMARY==="
find "$ROOT" -mindepth 1 -maxdepth 4 -name benchmark_summary.json 2>/dev/null \
  | sed 's#/benchmark_summary.json##' \
  | sort > /tmp/q2g_benchmark_dirs.txt
wc -l /tmp/q2g_benchmark_dirs.txt
tail -n 120 /tmp/q2g_benchmark_dirs.txt

echo "===PICK_JSON_ROOT_COUNTS==="
find "$ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | while read -r d; do
  c="$(find "$d" -name pick_result.json 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$c" != "0" ]]; then
    printf "%7s %s\n" "$c" "$d"
  fi
done | sort -nr | head -n 120

echo "===RAS_20260511_COUNTS==="
for d in \
  outputs/ras_revision_aggressive_20260511/ycb_aggregate_expansion \
  outputs/ras_revision_aggressive_20260511/ycb_gpu4_extra_1000_1199 \
  outputs/ras_revision_aggressive_20260511/ycb_post_main_1200_1599 \
  outputs/ras_revision_aggressive_20260511/error_correlation \
  outputs/ras_revision_aggressive_20260511/ycb_expansion_summary
do
  if [[ -d "$d" ]]; then
    printf "%s\tpick_json=%s\tsummary_json=%s\trows=%s\n" \
      "$d" \
      "$(find "$d" -name pick_result.json 2>/dev/null | wc -l | tr -d ' ')" \
      "$(find "$d" -name summary.json 2>/dev/null | wc -l | tr -d ' ')" \
      "$(find "$d" -name benchmark_rows.csv 2>/dev/null | wc -l | tr -d ' ')"
  fi
done

echo "===EXP_A_AND_NONCUBE_COUNTS==="
for d in \
  outputs/h200_60071_exp_a_200_seed_freeze_v2_20260508 \
  outputs/h200_60071_exp_a_200_seed_freeze_gpu4_7_resume_20260508 \
  outputs/h200_60071_noncube_feasibility_gpu4_7_20260508
do
  if [[ -d "$d" ]]; then
    printf "%s\tpick_json=%s\tsummary_json=%s\trows=%s\n" \
      "$d" \
      "$(find "$d" -name pick_result.json 2>/dev/null | wc -l | tr -d ' ')" \
      "$(find "$d" -name summary.json 2>/dev/null | wc -l | tr -d ' ')" \
      "$(find "$d" -name benchmark_rows.csv 2>/dev/null | wc -l | tr -d ' ')"
  fi
done

echo "===NOISY_ORACLE_COUNTS==="
find outputs -mindepth 1 -maxdepth 1 -type d -name 'h200_60071_noisy_oracle*' 2>/dev/null | sort | while read -r d; do
  printf "%s\tpick_json=%s\trows=%s\n" \
    "$d" \
    "$(find "$d" -name pick_result.json 2>/dev/null | wc -l | tr -d ' ')" \
    "$(find "$d" -name benchmark_rows.csv 2>/dev/null | wc -l | tr -d ' ')"
done
