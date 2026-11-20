"""Snapshot all W&B runs for serials 10000 through 10015, inclusive."""
import argparse
import collections
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import wandb
from wandb.apis.public.runs import _create_runs_query

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
RAW = DATA / 'full_grid_10000_10015'
SERIALS = list(range(10000, 10016))


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def main():
    global DATA, RAW
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    DATA = parser.parse_args().data_dir.resolve()
    RAW = DATA / 'full_grid_10000_10015'
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / 'runs.jsonl'
    if target.exists():
        raise FileExistsError(f'Preserve existing snapshot: {target}')
    filters = {'config.serial': {'$in': SERIALS + [str(s) for s in SERIALS]}}
    runs = wandb.Api(timeout=120).runs('nycu_pcs/TiT', filters=filters, per_page=500,
                                    order='+created_at')
    runs.QUERY = _create_runs_query(lazy=False)
    total = len(runs)
    print(f'W&B runs matching serials 10000-10015: {total}', flush=True)
    rows = []
    partial = RAW / 'runs.partial.jsonl'
    with partial.open('w', encoding='utf-8') as handle:
        for r in runs:
            record = clean(dict(id=r.id, name=r.name, state=r.state, url=r.url,
                                created_at=r.created_at, config=dict(r.config), summary=dict(r.summary)))
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            c, s = record['config'], record['summary']
            row = dict(c)
            row.update(s)
            row.update(run_id=r.id, run_name=r.name, state=r.state, run_url=r.url, created_at=r.created_at)
            rows.append({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()})
            if len(rows) % 500 == 0:
                print(f'Downloaded {len(rows)}/{total}', flush=True)
    assert len(rows) == total and len({r['run_id'] for r in rows}) == total
    partial.replace(target)
    frame = pd.DataFrame(rows)
    assert set(pd.to_numeric(frame.serial).astype(int)) <= set(SERIALS)
    frame = frame.sort_values(['serial', 'dataset_name', 'model_name', 'ckpt_path', 'seed', 'run_id'])
    frame.to_csv(DATA / 'backbones_quant_10000_10015.csv', index=False)
    finished = frame[frame.state == 'finished'].copy()
    hqq = finished.hqq.astype(str).str.lower().isin(['true', '1'])
    baseline = finished[~hqq & pd.to_numeric(finished.top1, errors='coerce').notna()].copy()
    baseline.to_csv(RAW / 'baseline_runs.csv', index=False)
    coverage = frame.groupby(['serial', 'dataset_name', 'state'], dropna=False).agg(
        runs=('run_id', 'size')).reset_index()
    coverage.to_csv(RAW / 'download_counts.csv', index=False)
    manifest = dict(project='nycu_pcs/TiT', downloaded_at=datetime.now(timezone.utc).isoformat(),
                    requested_serials=SERIALS, filters=filters, total_runs=len(frame),
                    state_counts=dict(collections.Counter(frame.state)),
                    finished_hqq_runs=int(hqq.sum()), baseline_raw_runs=len(baseline),
                    scope='Full config and summary for all matched run states; no run history or remote arrays',
                    nonfinite_json_policy='Nonfinite floating values are represented as null',
                    files={str(p.relative_to(DATA)).replace('\\', '/'): dict(bytes=p.stat().st_size,
                           sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in
                           [target, DATA / 'backbones_quant_10000_10015.csv', RAW / 'baseline_runs.csv', RAW / 'download_counts.csv']})
    (RAW / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('files', 'filters')}, indent=2))
    print(coverage.to_string(index=False))
    print('Baseline precision:', baseline.groupby(['fp16'], dropna=False).size().to_dict())
    print('Baseline models:', baseline.model_name.value_counts().to_dict())
    print('Observed weight bits:', sorted(pd.to_numeric(finished.loc[hqq, 'hqq_nbits']).unique()))
    print('Observed weight group sizes:', sorted(pd.to_numeric(finished.loc[hqq, 'hqq_group_size']).unique()))


if __name__ == '__main__':
    main()
