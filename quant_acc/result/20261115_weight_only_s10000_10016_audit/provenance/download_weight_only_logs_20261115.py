import collections
import concurrent.futures
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
import requests

root = Path(__file__).resolve().parents[1]
source = root / 'legacy_hqq/data/wandb_weight_only_20261115_125000_675669/runs.jsonl'
rows = [json.loads(s) for s in source.open(encoding='utf-8')]
chosen = [r for r in rows if r['config'].get('hqq') is True
          and r['config'].get('hqq_nbits') == 3 and r['config'].get('hqq_group_size') == 128]
target = root / 'legacy_hqq/data' / ('wandb_weight_only_w3logs_' + datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y%m%d_%H%M%S_%f'))
target.mkdir(parents=True, exist_ok=False)
print('LOG_DIRECTORY', target, 'RUNS', len(chosen), flush=True)
auth = ('api', os.environ['WANDB_API_KEY'])
urls = {}
for start in range(0, len(chosen), 20):
    batch = chosen[start:start + 20]
    parts = [f'r{i}: run(name: {json.dumps(r["id"])}) {{ files(names: ["output.log"], first: 10) {{ edges {{ node {{ name url }} }} }} }}'
             for i, r in enumerate(batch)]
    query = 'query { project(name: "TiT", entityName: "nycu_pcs") {' + ' '.join(parts) + '} }'
    response = requests.post('https://api.wandb.ai/graphql', auth=auth, json={'query': query}, timeout=120)
    response.raise_for_status()
    data = response.json()
    if data.get('errors'): raise RuntimeError(str(data['errors']))
    project = data['data']['project']
    for i, r in enumerate(batch):
        files = project[f'r{i}']['files']['edges']
        if files: urls[r['id']] = files[0]['node']['url']

def download(r):
    url = urls[r['id']]
    # Send the existing API credential only to W&B's authenticated API host.
    response = requests.get(url, auth=auth if urlparse(url).hostname == 'api.wandb.ai' else None, timeout=120)
    response.raise_for_status()
    dest = target / r['id'] / 'output.log'
    dest.parent.mkdir()
    dest.write_bytes(response.content)
    text = response.content.decode('utf-8', errors='replace')
    c = r['config']
    return dict(run_id=r['id'], run_url=r['url'], serial=c['serial'], dataset=c['dataset_name'],
                model=c['model_name'], mode=re.search(r'/(ft|fz|cal)/', c['ckpt_path']).group(1),
                relative_path=dest.relative_to(target).as_posix(), size_bytes=len(response.content),
                sha256=hashlib.sha256(response.content).hexdigest(),
                reported_linear_counts=re.findall(r'Found\s+(\d+)\s+Linear layers', text),
                excluded_lines=[s for s in text.splitlines() if '[HQQ] Excluded layers:' in s],
                cal_classifier_quantization_lines=[s.strip() for s in text.splitlines()
                    if ('dfsm.fc' in s or 'dfsm.cam_head' in s) and ('  - ' in s or 'HQQ' in s)])

records = []
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for index, record in enumerate(pool.map(download, chosen), 1):
            records.append(record)
            if index % 20 == 0: print('DOWNLOADED_LOGS', index, flush=True)
finally:
    receipt = dict(input_id=target.name, status='complete' if len(records) == len(chosen) else 'failed',
                   acquired_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
                   source_snapshot_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   selection='all existing W3/G128 weight-only runs in serials 10000-10015',
                   expected=len(chosen), downloaded=len(records), files=records)
    (target / 'source_manifest.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print('COUNT_SUMMARY', collections.Counter((r['model'], r['mode'], tuple(r['reported_linear_counts'])) for r in records), flush=True)
