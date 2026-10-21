"""Read-only W&B snapshot, retaining full configs and summaries for W6A4."""
import argparse
import collections
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import wandb
from wandb.apis.public.runs import _create_runs_query

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'outputs' / 'serial_20007' / 'data' / 'raw'
PROJECT = 'nycu_pcs/TiT'


def fetch(api, filters):
    runs = api.runs(PROJECT, filters=filters, per_page=500)
    runs.QUERY = _create_runs_query(lazy=False)
    rows = []
    for r in runs:
        rows.append(dict(id=r.id, name=r.name, state=r.state, url=r.url,
                         created_at=r.created_at, config=dict(r.config),
                         summary=dict(r.summary)))
    return rows


def save(name, rows):
    path = RAW / (name + '.json')
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    flat = []
    for r in rows:
        row = {k: v for k, v in r.items() if k not in ('config', 'summary')}
        for prefix in ('config', 'summary'):
            row.update({prefix + '.' + k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                        for k, v in r[prefix].items()})
        flat.append(row)
    pd.DataFrame(flat).to_csv(RAW / (name + '.csv'), index=False)


def main():
    global RAW
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=RAW)
    RAW = parser.parse_args().raw_dir.resolve()
    RAW.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=90)
    target_filter = {'config.serial': 20007}
    target = fetch(api, target_filter)
    if not target:
        raise RuntimeError('No serial 20007 runs returned')
    save('serial_20007_runs', target)
    paths = sorted({r['config']['ckpt_path'] for r in target})
    # Fetch only unquantized runs with exact checkpoint paths from this serial.
    baseline_filter = {'$and': [{'config.ckpt_path': {'$in': paths}},
                               {'$or': [{'config.hqq': False}, {'config.hqq': {'$exists': False}}]}]}
    baselines = fetch(api, baseline_filter)
    save('baseline_candidates', baselines)
    manifest = dict(project=PROJECT, serial=20007, downloaded_at=datetime.now(timezone.utc).isoformat(),
                    target_filter=target_filter, baseline_filter=baseline_filter,
                    target_count=len(target), baseline_candidate_count=len(baselines),
                    target_states=dict(collections.Counter(r['state'] for r in target)),
                    scope='W&B config and summary snapshot; no server feature arrays or checkpoints',
                    files={p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                           for p in sorted(RAW.iterdir()) if p.suffix in ('.csv', '.json') and p.name != 'source_manifest.json'})
    (RAW / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('files', 'baseline_filter')}, indent=2))
    for field in ('dataset_name', 'model_name', 'hqq_nbits', 'hqq_act_bits', 'hqq_group_size',
                  'hqq_act_group_size', 'hqq_scope', 'seed', 'fp16'):
        print(field, dict(collections.Counter(str(r['config'].get(field)) for r in target)))
    print('baseline_serials', dict(collections.Counter(str(r['config'].get('serial')) for r in baselines)))
    print('baseline_summary_keys', sorted({k for r in baselines for k in r['summary']}))


if __name__ == '__main__':
    main()
