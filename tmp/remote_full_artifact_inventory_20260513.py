#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/zetyun/OpenMythos_test')
OUT = ROOT / 'outputs'
JSON_OUT = OUT / 'remote_full_artifact_inventory_20260513.json'
MD_OUT = OUT / 'remote_full_artifact_inventory_20260513.md'


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        return {'_load_error': str(exc)}


def read_rows(path: Path):
    if not path.exists():
        return [], []
    try:
        if path.suffix.lower() == '.csv':
            with path.open('r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                return rows, list(reader.fieldnames or [])
        if path.suffix.lower() == '.json':
            data = load_json(path)
            rows = data if isinstance(data, list) else data.get('rows', []) if isinstance(data, dict) else []
            fields = sorted({k for row in rows if isinstance(row, dict) for k in row.keys()})
            return rows, fields
    except Exception as exc:
        return [{'_load_error': str(exc)}], []
    return [], []


def as_bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {'1', 'true', 'yes', 'y', 'success', 'ok'}


def num(v):
    try:
        if v in (None, ''):
            return None
        return float(v)
    except Exception:
        return None


def summarize_rows(rows):
    out = {'row_count': len(rows)}
    if not rows:
        return out
    seeds = []
    for r in rows:
        if isinstance(r, dict):
            val = r.get('seed')
            try:
                seeds.append(int(float(val)))
            except Exception:
                pass
    if seeds:
        out['seed_min'] = min(seeds)
        out['seed_max'] = max(seeds)
        out['seed_unique'] = len(set(seeds))
    bool_fields = ['success', 'pick_success', 'place_success', 'task_success', 'raw_env_success', 'grasp_attempted', 'target_available', 'detected', 'top1_changed_by_clip']
    for field in bool_fields:
        vals = [as_bool(r.get(field)) for r in rows if isinstance(r, dict) and field in r]
        if vals:
            out[field + '_count'] = int(sum(vals))
            out[field + '_rate'] = round(sum(vals) / len(vals), 6)
            out[field + '_denom'] = len(vals)
    for field in ['failure_type', 'env_id', 'grasp_target_mode', 'target_source', 'place_target_source', 'query']:
        vals = [str(r.get(field)) for r in rows if isinstance(r, dict) and r.get(field) not in (None, '')]
        if vals:
            out[field + '_top'] = Counter(vals).most_common(8)
    for field in ['target_error_m', 'target_error_to_oracle_cm', 'num_detections', 'valid_depth_points', 'elevated_points']:
        vals = [num(r.get(field)) for r in rows if isinstance(r, dict) and field in r]
        vals = [v for v in vals if v is not None]
        if vals:
            vals_sorted = sorted(vals)
            out[field + '_mean'] = round(sum(vals) / len(vals), 6)
            out[field + '_median'] = round(vals_sorted[len(vals_sorted)//2], 6)
            out[field + '_count'] = len(vals)
    return out


def rel(p: Path):
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)

summaries = []
for summary_path in OUT.rglob('benchmark_summary.json'):
    d = summary_path.parent
    summary = load_json(summary_path)
    rows_path = d / 'benchmark_rows.csv'
    if not rows_path.exists():
        rows_path = d / 'benchmark_rows.json'
    rows, fields = read_rows(rows_path)
    run_dir = d / 'runs'
    item = {
        'dir': rel(d),
        'summary_path': rel(summary_path),
        'rows_path': rel(rows_path) if rows_path.exists() else None,
        'rows_fields': fields[:80],
        'mtime': datetime.fromtimestamp(summary_path.stat().st_mtime).isoformat(timespec='seconds'),
        'summary': summary,
        'rows_summary': summarize_rows(rows),
        'run_summary_json_count': sum(1 for _ in run_dir.rglob('summary.json')) if run_dir.exists() else 0,
        'pick_result_json_count': sum(1 for _ in run_dir.rglob('pick_result.json')) if run_dir.exists() else 0,
        'top_candidate_3d_json_count': sum(1 for _ in run_dir.rglob('top_candidate_3d.json')) if run_dir.exists() else 0,
    }
    summaries.append(item)

status_files = []
for st in OUT.rglob('status.tsv'):
    try:
        lines = st.read_text(encoding='utf-8', errors='replace').splitlines()
    except Exception as exc:
        lines = [f'ERROR reading: {exc}']
    status_files.append({
        'path': rel(st),
        'mtime': datetime.fromtimestamp(st.stat().st_mtime).isoformat(timespec='seconds'),
        'line_count': len(lines),
        'tail': lines[-12:],
    })

report_files = []
for pat in ['*.md', '*.csv', '*.json', '*.tex']:
    for p in OUT.glob(pat):
        if p.name.startswith('remote_full_artifact_inventory_'):
            continue
        report_files.append({
            'path': rel(p),
            'size': p.stat().st_size,
            'mtime': datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec='seconds'),
        })

by_top = defaultdict(int)
for item in summaries:
    parts = Path(item['dir']).parts
    top = parts[1] if len(parts) > 1 and parts[0] == 'outputs' else parts[0]
    by_top[top] += 1

payload = {
    'created_at': datetime.now().isoformat(timespec='seconds'),
    'root': str(ROOT),
    'benchmark_summary_count': len(summaries),
    'status_file_count': len(status_files),
    'top_level_counts': dict(sorted(by_top.items(), key=lambda kv: (-kv[1], kv[0]))),
    'benchmarks': sorted(summaries, key=lambda x: x['dir']),
    'status_files': sorted(status_files, key=lambda x: x['path']),
    'top_level_report_files': sorted(report_files, key=lambda x: x['path']),
}
JSON_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

lines = []
lines.append('# Remote Full Artifact Inventory 2026-05-13')
lines.append('')
lines.append(f'- Created at: {payload["created_at"]}')
lines.append(f'- benchmark_summary.json files: {len(summaries)}')
lines.append(f'- status.tsv files: {len(status_files)}')
lines.append('')
lines.append('## Top-Level Benchmark Counts')
for k, v in payload['top_level_counts'].items():
    lines.append(f'- {k}: {v}')
lines.append('')
lines.append('## Benchmark Summaries')
for item in sorted(summaries, key=lambda x: x['dir']):
    s = item['summary'] if isinstance(item['summary'], dict) else {}
    r = item['rows_summary']
    env = s.get('env_id') or s.get('environment') or ''
    mode = s.get('grasp_target_mode') or s.get('target_source') or ''
    place = s.get('place_target_source') or ''
    total = s.get('total_runs') or r.get('row_count') or ''
    failed = s.get('failed_runs', '')
    pick = r.get('pick_success_rate', s.get('pick_success_rate', ''))
    task = r.get('task_success_rate', s.get('task_success_rate', ''))
    seeds = ''
    if 'seed_min' in r:
        seeds = f" seeds={r.get('seed_min')}-{r.get('seed_max')} unique={r.get('seed_unique')}"
    lines.append(f"- {item['dir']} | env={env} source={mode} place={place} n={total} failed={failed} pick={pick} task={task}{seeds}")
lines.append('')
lines.append('## Status Tails')
for st in sorted(status_files, key=lambda x: x['path']):
    lines.append(f"### {st['path']}")
    for line in st['tail']:
        lines.append('    ' + line)
    lines.append('')
MD_OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'json': rel(JSON_OUT), 'md': rel(MD_OUT), 'benchmark_summary_count': len(summaries), 'status_file_count': len(status_files)}, ensure_ascii=False))
