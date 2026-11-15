"""Prepare the corrected BEiTv2 cohort without mixing evaluation readouts."""
import argparse
import hashlib
import itertools
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DATASETS = ('aircraft', 'cars', 'cotton', 'cub', 'dogs', 'flowers', 'food', 'inat17',
            'moe', 'nabirds', 'pets', 'soyageing', 'soygene', 'soyglobal', 'soylocal', 'vegfru')
MODELS = ('vit_b16', 'swin_base_patch4_window7_224_in22k', 'beitv2_base_patch16_224')
MODES = ('ft', 'fz', 'cal')
AFFECTED = {'dogs', 'flowers', 'moe', 'pets', 'vegfru'}
BITS = (1, 1.58, 2, 3, 4, 5, 6, 8)
GROUPS = (8, 32, 128, 512)
EVAL_FIELDS = ('dataset_name', 'seed', 'fp16', 'image_size', 'classifier', 'test_resize_size',
               'test_multiple', 'test_square_resize_center_crop', 'test_resize_directly',
               'cal_single_crop', 'selector', 'shuffle_test', 'df_test', 'folder_test',
               'num_images_test', 'cal_ap_only', 'train_trainval', 'custom_mean_std')
REQUIRED = {'dataset_name', 'seed', 'fp16', 'image_size', 'classifier', 'test_resize_size',
            'test_multiple', 'test_square_resize_center_crop', 'num_images_test'}


def canonical(path):
    if isinstance(path, str) and path.startswith('../../data/'):
        return '/work/kyle0724/data/' + path[len('../../data/'):]
    return path


def select_context(historical, current, affected):
    merged = {r['id']: r for r in ([] if affected else historical)}
    merged.update({r['id']: r for r in current})
    selected = []
    for run in merged.values():
        metric = 'hqq_top1' if run['config'].get('hqq') else 'top1'
        value = run['summary'].get(metric)
        if run['state'] == 'finished' and isinstance(value, (int, float)) and math.isfinite(value):
            selected.append(run)
    return selected


def context(run):
    c = run['config']
    path = canonical(c['ckpt_path'])
    match = re.search(r'/(?:ckpt/)?(cal|ft|fz)(?:_ckpts)?/', path)
    if not match:
        raise ValueError(f'Unknown checkpoint mode: {path}')
    model = c['model_name']
    if model == 'beitv2_base_patch16_224_in22k':
        model = 'beitv2_base_patch16_224'
    return c['dataset_name'], model, match[1]


def differences(left, right):
    mismatch, unrecorded = [], []
    for field in EVAL_FIELDS:
        if field not in left and field not in right and field not in REQUIRED:
            unrecorded.append(field)
        elif field not in left or field not in right or left[field] != right[field]:
            mismatch.append(field)
    for field in ('ckpt_path', 'dataset_root_path'):
        if field not in left or field not in right or canonical(left[field]) != canonical(right[field]):
            mismatch.append(field)
    if left.get('checkpoint_sha256') and right.get('checkpoint_sha256'):
        if left['checkpoint_sha256'] != right['checkpoint_sha256']:
            mismatch.append('checkpoint_sha256')
    return mismatch, unrecorded


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-jsonl', type=Path, default=root / 'data/baseline_recovery/source/historical_runs.jsonl')
    parser.add_argument('--current-jsonl', type=Path, default=root / 'data/baseline_recovery/snapshots/20261021/full_grid_10000_10015/runs.jsonl')
    parser.add_argument('--output-dir', type=Path, default=root / 'data/baseline_recovery/selected_20261021')
    parser.add_argument('--rerun-after', default='2026-10-20T00:00:00Z')
    args = parser.parse_args(argv)
    historical, current = load_jsonl(args.historical_jsonl), load_jsonl(args.current_jsonl)
    old_manifest = args.historical_jsonl.with_name('historical_manifest.json')
    expected = json.loads(old_manifest.read_text(encoding='utf-8'))['files']['full_grid_10000_10015/runs.jsonl']['sha256']
    if sha(args.historical_jsonl) != expected:
        raise ValueError('Historical snapshot hash mismatch')
    current_manifest = args.current_jsonl.with_name('source_manifest.json')
    expected = json.loads(current_manifest.read_text(encoding='utf-8'))['files']['full_grid_10000_10015/runs.jsonl']['sha256']
    if sha(args.current_jsonl) != expected:
        raise ValueError('Current snapshot hash mismatch')
    old_ids, live_ids = {r['id'] for r in historical}, {r['id'] for r in current}
    old_groups, new_groups = {}, {}
    for r in historical:
        old_groups.setdefault(context(r), []).append(r)
    for r in current:
        new_groups.setdefault(context(r), []).append(r)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    selected, audits, missing, contexts, baseline_rows = [], [], [], [], []
    for key in itertools.product(DATASETS, MODELS, MODES):
        dataset, model, mode = key
        affected = dataset in AFFECTED and model.startswith('beitv2')
        recent = new_groups.get(key, [])
        if affected:
            recent = [r for r in recent if r['id'] not in old_ids and r['created_at'] >= args.rerun_after]
        records = select_context(old_groups.get(key, []), recent, affected)
        baseline = [r for r in records if r['config'].get('hqq') is False]
        quantized = [r for r in records if r['config'].get('hqq') is True]
        if len(baseline) > 1:
            raise ValueError(f'Ambiguous baseline: {key}, {[r["id"] for r in baseline]}')
        base = baseline[0] if baseline else None
        expected_classifier = 'pool' if mode == 'cal' else 'cls'
        if base and (base['summary']['top1'] <= 0 or base['config'].get('fp16') is not True):
            raise ValueError(f'Invalid unquantized AMP baseline: {key}')
        if affected and base and base['config'].get('classifier') != expected_classifier:
            raise ValueError(f'Corrected baseline has wrong classifier: {key}')
        if base:
            selected.append(base)
            baseline_rows.append(dict(dataset=dataset, model=model, mode=mode,
                run_id=base['id'], serial=base['config']['serial'], top1=base['summary']['top1'],
                classifier=base['config'].get('classifier'), precision='unquantized_AMP',
                checkpoint_bytes_verified=False,
                provenance='corrected_rerun' if affected else ('current' if base['id'] in live_ids else 'historical_snapshot')))
        seen = set()
        accepted = 0
        for run in quantized:
            c = run['config']
            condition = (float(c['hqq_nbits']), int(c['hqq_group_size']))
            if condition in seen:
                raise ValueError(f'Duplicate HQQ condition: {key}, {condition}')
            seen.add(condition)
            mismatch, unrecorded = differences(c, base['config']) if base else (['missing_baseline'], [])
            if c.get('hqq_scope') not in (None, 'linear') or c.get('hqq_act_bits') is not None:
                mismatch.append('not_weight_only_linear')
            if c.get('hqq_exclude') != 'head' or c.get('hqq_compute_dtype') != 'float16':
                mismatch.append('quantization_settings')
            if condition not in set(itertools.product(BITS, GROUPS)):
                mismatch.append('unexpected_condition')
            valid = not mismatch
            audits.append(dict(dataset=dataset, model=model, mode=mode, nbits=condition[0], group_size=condition[1],
                quant_run_id=run['id'], baseline_run_id=base['id'] if base else '',
                accepted=valid, mismatches=';'.join(mismatch), shared_unrecorded_fields=';'.join(unrecorded),
                batch_size_matches=c.get('batch_size') == base['config'].get('batch_size') if base else False,
                restored_from_historical_snapshot=run['id'] not in live_ids,
                corrected_rerun=affected))
            if valid:
                selected.append(run)
                accepted += 1
        for bits, group in itertools.product(BITS, GROUPS):
            if (bits, group) not in seen:
                missing.append(dict(dataset=dataset, model=model, mode=mode, nbits=bits, group_size=group,
                                    reason='corrected_rerun_not_finished' if affected else 'missing_condition'))
        contexts.append(dict(dataset=dataset, model=model, mode=mode, baseline_available=base is not None,
            accepted_conditions=accepted, expected_conditions=32, complete=accepted == 32 and base is not None,
            corrected_rerun=affected))
    rows = []
    for run in selected:
        row = dict(run['config'])
        row.update(run['summary'])
        row.update(run_id=run['id'], run_url=run['url'], created_at=run['created_at'], state=run['state'],
                   host=None, ckpt_path=canonical(run['config']['ckpt_path']))
        rows.append({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'curated_16_datasets.csv', index=False)
    frame[frame.dataset_name.ne('cotton')].to_csv(output / 'curated_15_datasets.csv', index=False)
    (output / 'selected_runs.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in selected), encoding='utf-8')
    pd.DataFrame(audits).to_csv(output / 'pair_audit.csv', index=False)
    pd.DataFrame(missing, columns=['dataset', 'model', 'mode', 'nbits', 'group_size', 'reason']).to_csv(
        output / 'missing_conditions.csv', index=False)
    pd.DataFrame(contexts).to_csv(output / 'context_coverage.csv', index=False)
    pd.DataFrame(baseline_rows).to_csv(output / 'baselines.csv', index=False)
    result = dict(generated_at=datetime.now(timezone.utc).isoformat(),
        status='complete' if all(c['complete'] for c in contexts) else 'partial_pending_server_rerun',
        selected_runs=len(rows), selected_hqq_runs=int(frame.hqq.sum()), selected_baselines=int((~frame.hqq).sum()),
        rejected_pairs=sum(not r['accepted'] for r in audits), complete_contexts=sum(c['complete'] for c in contexts),
        expected_contexts=144, missing_conditions=len(missing),
        affected_policy='Replace FP and all HQQ conditions for all 15 affected BEiTv2 contexts; never fall back to old runs',
        historical_policy='Retain verified archived runs for unaffected contexts, even when removed from current W&B',
        precision='unquantized_AMP; fp32_top1 is only a legacy schema name',
        path_alias='../../data/ -> /work/kyle0724/data/; no basename matching',
        checkpoint_bytes_identity='unverified unless both records explicitly contain a matching checkpoint_sha256',
        inputs={str(p):sha(p) for p in (args.historical_jsonl, old_manifest, args.current_jsonl, current_manifest)},
        artifacts={p.name:sha(p) for p in output.iterdir() if p.suffix in ('.csv', '.jsonl')})
    (output / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','artifacts')}, indent=2))
    print(pd.DataFrame(contexts).query('not complete').to_string(index=False))


if __name__ == '__main__':
    main()
