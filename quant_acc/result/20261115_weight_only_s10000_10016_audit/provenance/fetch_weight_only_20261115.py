import collections
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

root = Path(__file__).resolve().parents[1]
stamp = datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y%m%d_%H%M%S_%f')
target = root / 'legacy_hqq' / 'data' / ('wandb_weight_only_' + stamp)
target.mkdir(parents=True, exist_ok=False)
query = {'$or': [{'config.serial': {'$in': list(range(10000, 10017))}},
                 {'config.serial': {'$in': [str(i) for i in range(10000, 10017)]}}]}
receipt = {'input_id': target.name, 'project': 'nycu_pcs/TiT', 'query': query,
           'acquired_at': datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
           'status': 'running', 'files': {}}
manifest = target / 'source_manifest.json'
manifest.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print('INPUT_DIRECTORY', target, flush=True)
session = requests.Session()
session.auth = ('api', os.environ['WANDB_API_KEY'])
gql = '''query Runs($cursor: String, $filters: JSONString) {
 project(name: "TiT", entityName: "nycu_pcs") {
  runCount(filters: $filters)
  runs(filters: $filters, after: $cursor, first: 500, order: "+createdAt") {
   edges { node { name displayName state config summaryMetrics createdAt group jobType tags commit } }
   pageInfo { endCursor hasNextPage }
  }
 }
}'''
receipt['transport'] = 'authenticated read-only HTTPS GraphQL; SDK service unavailable in sandbox'
counts = collections.Counter()
try:
    cursor, index = None, 0
    with (target / 'runs.jsonl').open('x', encoding='utf-8') as stream:
        while True:
            response = session.post('https://api.wandb.ai/graphql', json={
                'query': gql, 'variables': {'cursor': cursor, 'filters': json.dumps(query)}}, timeout=120)
            response.raise_for_status()
            result = response.json()
            if result.get('errors'):
                raise RuntimeError(str(result['errors']))
            project = result['data']['project']
            if project is None:
                raise RuntimeError('Project unavailable')
            if cursor is None:
                receipt['remote_run_count'] = project['runCount']
                print('REMOTE_MATCHES', project['runCount'], flush=True)
            page = project['runs']
            for edge in page['edges']:
                node = edge['node']
                config = json.loads(node['config']) if isinstance(node['config'], str) else node['config']
                config = {k: v.get('value') if isinstance(v, dict) and 'value' in v else v
                          for k, v in config.items()}
                summary = json.loads(node['summaryMetrics']) if isinstance(node['summaryMetrics'], str) else node['summaryMetrics']
                row = {'id': node['name'], 'name': node['displayName'],
                       'url': 'https://wandb.ai/nycu_pcs/TiT/runs/' + node['name'],
                       'state': node['state'], 'created_at': node['createdAt'],
                       'config': config, 'summary': summary, 'job_type': node['jobType'],
                       'group': node['group'], 'tags': node['tags'], 'commit': node['commit']}
                stream.write(json.dumps(row, allow_nan=False, default=str) + '\n')
                counts[(str(config.get('serial')), row['state'])] += 1
                index += 1
            stream.flush()
            print('DOWNLOADED', index, flush=True)
            if not page['pageInfo']['hasNextPage']:
                break
            cursor = page['pageInfo']['endCursor']
    source = target / 'runs.jsonl'
    receipt.update(status='complete', run_count=sum(counts.values()),
                   files={'runs.jsonl': {'size_bytes': source.stat().st_size,
                                        'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}},
                   counts=[{'serial': k[0], 'state': k[1], 'count': v}
                           for k, v in sorted(counts.items())])
except BaseException as error:
    receipt.update(status='failed', error=type(error).__name__)
    raise
finally:
    receipt['finished_at'] = datetime.now(ZoneInfo('Asia/Taipei')).isoformat()
    manifest.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({'status': receipt['status'], 'run_count': receipt.get('run_count'),
                  'counts': receipt.get('counts')}, indent=2), flush=True)
