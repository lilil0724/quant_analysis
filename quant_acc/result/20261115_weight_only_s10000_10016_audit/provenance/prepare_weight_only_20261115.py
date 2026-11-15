import collections
import hashlib
import itertools
import json
import math
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'legacy_hqq/data/wandb_weight_only_20261115_125000_675669/runs.jsonl'
RUN = '20261115_weight_only_s10000_10016_audit'
OUT = ROOT / 'legacy_hqq/result' / RUN
OUT.mkdir(parents=True, exist_ok=False)
for name in ('audit', 'prepared', 'provenance', 'logs'):
    (OUT / name).mkdir()
rows = [json.loads(line) for line in SOURCE.open(encoding='utf-8')]
datasets = 'aircraft cars cub dogs flowers food inat17 moe nabirds pets soyageing soygene soyglobal soylocal vegfru cotton'.split()
models = ['vit_b16', 'swin_base_patch4_window7_224_in22k', 'beitv2_base_patch16_224']
bits, groups = [1, 1.58, 2, 3, 4, 5, 6, 8], [8, 32, 128, 512]

def canonical(path):
    return '/work/kyle0724/data/' + path[11:] if isinstance(path, str) and path.startswith('../../data/') else path

def context(row):
    c = row['config']
    mode = re.search(r'/(ft|fz|cal)/', c['ckpt_path']).group(1)
    return c['dataset_name'], c['model_name'].removesuffix('_in22k') if c['model_name'].startswith('beit') else c['model_name'], mode

by_context = collections.defaultdict(list)
for r in rows:
    by_context[context(r)].append(r)
audits, coverage, missing, selected, baseline_table = [], [], [], [], []
eval_keys = ('seed', 'fp16', 'image_size', 'classifier', 'test_resize_size', 'test_multiple',
             'test_square_resize_center_crop', 'test_resize_directly', 'selector', 'shuffle_test',
             'df_test', 'folder_test', 'num_images_test', 'cal_ap_only', 'train_trainval', 'custom_mean_std')
for key in itertools.product(datasets, models, ('ft', 'fz', 'cal')):
    records = by_context[key]
    baselines = [r for r in records if r['config'].get('hqq') is False and r['state'] == 'finished']
    if len(baselines) > 1:
        raise ValueError(f'Ambiguous baseline: {key}')
    baseline = baselines[0] if baselines else None
    if baseline:
        selected.append(baseline)
        baseline_table.append(dict(dataset=key[0], model=key[1], mode=key[2], run_id=baseline['id'],
                                   serial=baseline['config']['serial'], top1=baseline['summary'].get('top1'),
                                   checkpoint_sha256=baseline['config'].get('checkpoint_sha256'),
                                   classifier=baseline['config'].get('classifier')))
    seen, accepted = set(), 0
    for r in records:
        c = r['config']
        if c.get('hqq') is not True:
            continue
        condition = (float(c['hqq_nbits']), int(c['hqq_group_size']))
        if condition in seen:
            raise ValueError(f'Ambiguous HQQ condition: {key}, {condition}')
        seen.add(condition)
        reasons = []
        if r['state'] != 'finished': reasons.append('not_finished')
        if c.get('hqq_act_bits') is not None or c.get('hqq_scope') not in (None, 'linear'):
            reasons.append('not_weight_only_linear')
        if c.get('debugging') is not False: reasons.append('debugging_not_false')
        if c.get('hqq_compute_dtype') != 'float16' or c.get('fp16') is not True:
            reasons.append('unexpected_precision')
        metric = r['summary'].get('hqq_top1')
        if not isinstance(metric, (float, int)) or not math.isfinite(metric) or not 0 <= metric <= 100:
            reasons.append('invalid_hqq_top1')
        if baseline is None:
            reasons.append('missing_baseline')
        else:
            b = baseline['config']
            reasons.extend('evaluation_mismatch:' + field for field in eval_keys if c.get(field) != b.get(field))
            reasons.extend('path_mismatch:' + field for field in ('ckpt_path', 'dataset_root_path')
                           if canonical(c.get(field)) != canonical(b.get(field)))
            bt = baseline['summary'].get('top1')
            if not isinstance(bt, (float, int)) or not math.isfinite(bt) or not 0 < bt <= 100:
                reasons.append('invalid_baseline_top1')
        audits.append(dict(dataset=key[0], model=key[1], mode=key[2], bits=condition[0], group_size=condition[1],
                           run_id=r['id'], run_url=r['url'], baseline_run_id=baseline['id'] if baseline else '',
                           accepted=not reasons, reasons=';'.join(reasons),
                           checkpoint_bytes_verified=bool(baseline and c.get('checkpoint_sha256') and
                               c.get('checkpoint_sha256') == baseline['config'].get('checkpoint_sha256')),
                           old_cal_classifier_exclusion_verified=False if key[2] == 'cal' else None,
                           hqq_exclude=c.get('hqq_exclude'),
                           batch_size_matches=bool(baseline and c.get('batch_size') == baseline['config'].get('batch_size'))))
        if not reasons:
            selected.append(r)
            accepted += 1
    for bit, group in itertools.product(bits, groups):
        if (bit, group) not in seen:
            missing.append(dict(dataset=key[0], model=key[1], mode=key[2], bits=bit, group_size=group,
                                reason='not_present_in_current_WandB_query'))
    coverage.append(dict(dataset=key[0], model=key[1], mode=key[2], baseline_available=bool(baseline),
                         observed_conditions=len(seen), accepted_conditions=accepted, expected_conditions=32,
                         complete=bool(baseline and accepted == 32),
                         corrected_cal_available=any(int(r['config']['serial']) == 10016 for r in records)))

frame_rows = []
for r in selected:
    c = {**r['config'], **r['summary'], 'run_id': r['id'], 'run_url': r['url'],
         'created_at': r['created_at'], 'state': r['state'], 'host': None}
    c['ckpt_path'] = canonical(c['ckpt_path'])
    frame_rows.append({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in c.items()})
pd.DataFrame(frame_rows).to_csv(OUT / 'prepared/curated_historical_runs.csv', index=False)
for name, table in [('pair_audit', audits), ('context_coverage', coverage), ('missing_conditions', missing),
                    ('baseline_audit', baseline_table)]:
    pd.DataFrame(table).to_csv(OUT / 'audit' / (name + '.csv'), index=False)
receipt = dict(run_id=RUN, status='running', input_source=str(SOURCE),
               input_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               started_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
               command='python -B -u ' + str(Path(__file__)),
               code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               scope='Historical 10000-10015 only; corrected CAL 10016 not returned by live query',
               selected_baselines=len(baseline_table), selected_hqq=sum(a['accepted'] for a in audits),
               rejected_hqq=sum(not a['accepted'] for a in audits), missing_conditions=len(missing),
               complete_contexts=sum(c['complete'] for c in coverage), expected_contexts=144,
               corrected_cal_runs_found=sum(int(r['config']['serial']) == 10016 for r in rows),
               pending='Do not substitute old CAL for the unavailable corrected serial 10016', stages={})
(OUT / 'run.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in receipt.items() if k not in ('stages',)}, indent=2))
print('REJECTION_REASONS', collections.Counter(a['reasons'] for a in audits if not a['accepted']))
