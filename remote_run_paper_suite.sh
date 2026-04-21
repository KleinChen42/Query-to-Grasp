#!/usr/bin/env bash
# Run the full single-view paper-ready benchmark suite on the H200 node.
# Produces 4 benchmark dirs, 4 reports, a paper ablation table, and a
# per-query diagnostics table.

set -euo pipefail

cd "$HOME/OpenMythos_test"
# shellcheck disable=SC1091
source "$HOME/q2g_venv/bin/activate"
export PYTHONPATH="$PWD"
export HF_ENDPOINT=https://hf-mirror.com
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export TOKENIZERS_PARALLELISM=false

STAMP=$(date +%Y%m%d_%H%M%S)
SUITE_ROOT="outputs/paper_suite_${STAMP}"
mkdir -p "${SUITE_ROOT}"

HF_NO_CLIP_DIR="${SUITE_ROOT}/hf_no_clip"
HF_WITH_CLIP_DIR="${SUITE_ROOT}/hf_with_clip"
AMBIG_NO_CLIP_DIR="${SUITE_ROOT}/ambiguity_no_clip"
AMBIG_WITH_CLIP_DIR="${SUITE_ROOT}/ambiguity_with_clip"

HF_QUERIES=(
  "red cube"
  "blue cube"
  "green cube"
  "cube"
)
NUM_SEEDS=2

banner() {
  echo
  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] $*"
  echo "=============================================================="
}

banner "1/7  HF + no-CLIP benchmark  ->  ${HF_NO_CLIP_DIR}"
python scripts/run_single_view_pick_benchmark.py \
  --queries "${HF_QUERIES[@]}" \
  --num-runs "${NUM_SEEDS}" \
  --detector-backend hf \
  --skip-clip \
  --output-dir "${HF_NO_CLIP_DIR}"

banner "2/7  HF + CLIP benchmark  ->  ${HF_WITH_CLIP_DIR}"
python scripts/run_single_view_pick_benchmark.py \
  --queries "${HF_QUERIES[@]}" \
  --num-runs "${NUM_SEEDS}" \
  --detector-backend hf \
  --use-clip \
  --output-dir "${HF_WITH_CLIP_DIR}"

banner "3/7  Ambiguity + no-CLIP benchmark  ->  ${AMBIG_NO_CLIP_DIR}"
python scripts/run_ambiguity_benchmark.py \
  --num-runs "${NUM_SEEDS}" \
  --detector-backend hf \
  --skip-clip \
  --output-dir "${AMBIG_NO_CLIP_DIR}"

banner "4/7  Ambiguity + CLIP benchmark  ->  ${AMBIG_WITH_CLIP_DIR}"
python scripts/run_ambiguity_benchmark.py \
  --num-runs "${NUM_SEEDS}" \
  --detector-backend hf \
  --use-clip \
  --output-dir "${AMBIG_WITH_CLIP_DIR}"

banner "5/7  Markdown reports"
for d in "${HF_NO_CLIP_DIR}" "${HF_WITH_CLIP_DIR}" "${AMBIG_NO_CLIP_DIR}" "${AMBIG_WITH_CLIP_DIR}"; do
  python scripts/generate_benchmark_report.py --benchmark-dir "$d"
done

banner "6/7  Paper ablation table"
python scripts/generate_paper_ablation_table.py \
  --benchmark "HF-no-CLIP=${HF_NO_CLIP_DIR}" \
  --benchmark "HF-with-CLIP=${HF_WITH_CLIP_DIR}" \
  --benchmark "Ambig-no-CLIP=${AMBIG_NO_CLIP_DIR}" \
  --benchmark "Ambig-with-CLIP=${AMBIG_WITH_CLIP_DIR}" \
  --output-md "${SUITE_ROOT}/paper_ablation_table.md" \
  --output-csv "${SUITE_ROOT}/paper_ablation_table.csv"

banner "7/7  Per-query diagnostics table"
python scripts/generate_per_query_diagnostics_table.py \
  --benchmark "HF-no-CLIP=${HF_NO_CLIP_DIR}" \
  --benchmark "HF-with-CLIP=${HF_WITH_CLIP_DIR}" \
  --benchmark "Ambig-no-CLIP=${AMBIG_NO_CLIP_DIR}" \
  --benchmark "Ambig-with-CLIP=${AMBIG_WITH_CLIP_DIR}" \
  --output-md "${SUITE_ROOT}/per_query_diagnostics_table.md" \
  --output-csv "${SUITE_ROOT}/per_query_diagnostics_table.csv"

banner "ALL DONE  suite_root=${SUITE_ROOT}"
echo "${SUITE_ROOT}" > outputs/paper_suite_latest.txt
