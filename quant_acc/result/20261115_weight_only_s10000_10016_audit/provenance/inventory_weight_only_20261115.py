import collections
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import requests

root = Path(__file__).resolve().parents[1]
target = root / 'legacy_hqq/data' / ('wandb_weight_only_files_' + datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y%m%d_%H%M%S_%f'))
target.mkdir(parents=True, exist_ok=False)
session = requests.Session()
session.auth = ('api', os.environ['WANDB_API_KEY'])
filters = {'config.serial': {'$in': list(range(10000, 10017))}}
query = '''query Inventory($cursor: String, $filters: JSONString) {
 project(name: "TiT", entityName: "nycu_pcs") {
  runs(filters: $filters, after: $cursor, first: 250) {
   edges { node {
    name fileCount
    files(first: 100) { edges { node { name sizeBytes md5 } } pageInfo { hasNextPage } }
    outputArtifacts(first: 100) { edges { node { id digest versionIndex artifactSequence { name } } } pageInfo { hasNextPage } }
   } }
   pageInfo { hasNextPage endCursor }
  }
 }
}'''
receipt = dict(status='running', project='nycu_pcs/TiT', query=filters,
               acquired_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(), input_id=target.name)
print('INVENTORY_DIRECTORY', target, flush=True)
count, cursor, file_counts, art_counts, truncated = 0, None, collections.Counter(), collections.Counter(), []
try:
    with (target / 'run_files_and_artifacts.jsonl').open('x', encoding='utf-8') as stream:
        while True:
            response = session.post('https://api.wandb.ai/graphql', json={'query': query,
                 'variables': {'cursor': cursor, 'filters': json.dumps(filters)}}, timeout=120)
            response.raise_for_status()
            data = response.json()
            if data.get('errors'): raise RuntimeError(str(data['errors']))
            page = data['data']['project']['runs']
            for edge in page['edges']:
                node = edge['node']
                files = [e['node'] for e in node['files']['edges']]
                artifacts = [e['node'] for e in node['outputArtifacts']['edges']]
                if node['files']['pageInfo']['hasNextPage'] or node['outputArtifacts']['pageInfo']['hasNextPage']:
                    truncated.append(node['name'])
                row = dict(run_id=node['name'], file_count=node['fileCount'], files=files, logged_artifacts=artifacts)
                stream.write(json.dumps(row) + '\n')
                file_counts.update(f['name'] for f in files)
                art_counts.update(a['artifactSequence']['name'] for a in artifacts)
                count += 1
            stream.flush()
            print('INVENTORIED', count, flush=True)
            if not page['pageInfo']['hasNextPage']: break
            cursor = page['pageInfo']['endCursor']
    receipt.update(status='complete', runs=count, files_by_name=dict(file_counts),
                   logged_artifacts_by_collection=dict(art_counts), truncated_runs=truncated)
except BaseException as error:
    receipt.update(status='failed', error=type(error).__name__)
    raise
finally:
    p = target / 'run_files_and_artifacts.jsonl'
    receipt['files'] = {p.name: dict(size_bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())} if p.exists() else {}
    (target / 'source_manifest.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in receipt.items() if k != 'files'}, indent=2), flush=True)
