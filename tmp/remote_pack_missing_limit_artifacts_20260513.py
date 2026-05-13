#!/usr/bin/env python3
from __future__ import annotations
import tarfile
from pathlib import Path

ROOT = Path('/home/zetyun/OpenMythos_test')
OUT_TGZ = ROOT / 'outputs' / 'remote_missing_limit_artifacts_20260513.tgz'
roots = [
    ROOT / 'outputs' / 'h200_60071_maniskill_task_expanded_targetsource_seed0_49',
    ROOT / 'outputs' / 'h200_60071_exp_a_200_seed_freeze_20260507' / 'noncube_candidate_scan',
]
roots += sorted((ROOT / 'outputs').glob('h200_exp_b_liftpeg_*'))
patterns = [
    'benchmark_summary.json',
    'benchmark_rows.csv',
    'benchmark_rows.json',
    'status.tsv',
    '*.log',
    '*.md',
    '*.csv',
    '*.json',
]
files = []
for root in roots:
    if not root.exists():
        continue
    for pat in patterns:
        for p in root.rglob(pat):
            if '/runs/' in str(p):
                continue
            if p.is_file() and p.stat().st_size < 10_000_000:
                files.append(p)
files = sorted(set(files))
with tarfile.open(OUT_TGZ, 'w:gz') as tar:
    for p in files:
        tar.add(p, arcname=str(p.relative_to(ROOT)))
print({'archive': str(OUT_TGZ.relative_to(ROOT)), 'files': len(files), 'bytes': OUT_TGZ.stat().st_size})
