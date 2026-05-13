#!/usr/bin/env bash
set -eu

echo "===PLACE_RELATED_BENCHMARK_DIRS==="
find outputs -mindepth 1 -maxdepth 4 -name benchmark_rows.csv 2>/dev/null \
  | grep -Ei 'predicted_place|oracle_cubeB|pickplace|place' \
  | sed 's#/benchmark_rows.csv##' \
  | sort \
  | head -n 160

echo "===PLACE_RELATED_PICK_JSON_COUNTS==="
find outputs -mindepth 1 -maxdepth 3 -type d 2>/dev/null \
  | grep -Ei 'predicted_place|pickplace|place|oracle' \
  | sort \
  | while read -r d; do
      c="$(find "$d" -name pick_result.json 2>/dev/null | wc -l | tr -d ' ')"
      if [[ "$c" != "0" ]]; then
        printf "%7s %s\n" "$c" "$d"
      fi
    done \
  | sort -nr \
  | head -n 120

echo "===SAMPLE_PLACE_METADATA==="
sample="$(find outputs -path '*predicted_place*' -name pick_result.json 2>/dev/null | head -n 1 || true)"
if [[ -n "${sample}" ]]; then
  echo "PREDICTED_SAMPLE=${sample}"
  /home/zetyun/q2g_venv/bin/python - <<PY
import json
p="${sample}"
d=json.load(open(p))
m=d.get("metadata", {})
print("target_xyz", d.get("target_xyz"))
print("place_xyz", d.get("place_xyz"))
print("place_target_xyz", m.get("place_target_xyz"))
print("place_target_source", m.get("place_target_source"))
print("place_target_metadata_keys", sorted((m.get("place_target_metadata") or {}).keys()))
PY
fi

sample="$(find outputs -path '*oracle*place*' -name pick_result.json 2>/dev/null | head -n 1 || true)"
if [[ -n "${sample}" ]]; then
  echo "ORACLE_PLACE_SAMPLE=${sample}"
  /home/zetyun/q2g_venv/bin/python - <<PY
import json
p="${sample}"
d=json.load(open(p))
m=d.get("metadata", {})
print("target_xyz", d.get("target_xyz"))
print("place_xyz", d.get("place_xyz"))
print("place_target_xyz", m.get("place_target_xyz"))
print("place_target_source", m.get("place_target_source"))
print("place_target_metadata", m.get("place_target_metadata"))
PY
fi
