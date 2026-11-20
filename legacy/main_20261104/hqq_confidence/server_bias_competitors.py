"""Run bias/competitor diagnostics on nano4 TGDA paired W3/G128 exports.

Input: SERIAL/DATASET/MODEL/MODE/{manifest.json,samples.json,fp/,w3_g128/}.
Only logits and sample/prediction metadata are read; large layer features are unused.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
import platform
import re
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from . import bias_competitor_core as core

DATASETS = ('aircraft', 'cars', 'cub', 'dogs', 'flowers', 'food', 'inat17', 'moe',
            'nabirds', 'pets', 'soyageing', 'soygene', 'soyglobal', 'soylocal', 'vegfru', 'cotton')
MODELS = ('vit_b16', 'swin_base_patch4_window7_224_in22k', 'beitv2_base_patch16_224')
MODES = ('ft', 'fz', 'cal')
USED_FILES = ('samples.json', 'fp/logits.npy', 'fp/predictions.csv',
              'w3_g128/logits.npy', 'w3_g128/predictions.csv')
SCORE_COLUMNS = ['model_name', 'checkpoint_mode', 'scope', 'dataset_name', 'outcome', 'model',
                 'n_test', 'event_rate', 'brier', 'log_loss', 'auroc', 'auroc_status', 'n_train',
                 'n_train_datasets', 'iterations']
COMPARISON_COLUMNS = ['model_name', 'checkpoint_mode', 'scope', 'dataset_name', 'outcome', 'n_test',
                      'event_rate', *[f'{prefix}_{metric}' for metric in ('brier', 'log_loss', 'auroc')
                                     for prefix in ('margin', 'competitors', 'delta')]]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n', encoding='utf-8')


def safe_file(folder, name):
    part = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or part.is_absolute() or '..' in part.parts:
        raise ValueError(f'Unsafe source path: {name}')
    path = folder.joinpath(*part.parts)
    if folder.resolve() not in path.resolve().parents:
        raise ValueError(f'Source link escapes pair folder: {path}')
    return path


def csv_selection(value):
    values = value.split(',')
    if not values or len(set(values)) != len(values) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', v) for v in values):
        raise argparse.ArgumentTypeError('Use unique safe names separated by commas, without spaces')
    return values


def build_parser():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=repo/'../../data/w3_g128_analysis')
    parser.add_argument('--serial', required=True, type=int)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--datasets', type=csv_selection, default=list(DATASETS))
    parser.add_argument('--models', type=csv_selection, default=list(MODELS))
    parser.add_argument('--modes', type=csv_selection, default=list(MODES))
    parser.add_argument('--chunk-size', type=int, default=256)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=core.SEED)
    parser.add_argument('--preflight-only', action='store_true', help='Validate the complete selection without writing outputs')
    parser.add_argument('--save-sample-results', action='store_true', help='Retain derived features/labels/predictions NPZ per pair')
    return parser


def roots(args):
    root = args.input_root.expanduser().resolve()
    if root.name == str(args.serial):
        root = root.parent
    serial_root = root/str(args.serial)
    output = (args.output_dir.expanduser() if args.output_dir else serial_root/'analysis/bias_competitors_v1').resolve()
    if output == root or output == serial_root or output in root.parents:
        raise ValueError('Output cannot contain or replace source roots')
    if root in output.parents and serial_root/'analysis' not in (output, *output.parents):
        raise ValueError('Output within the input root must be under SERIAL/analysis')
    return root, serial_root, output


def integer_column(rows, name):
    numeric = pd.to_numeric(rows[name], errors='raise').to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.all(numeric == np.floor(numeric)):
        raise ValueError(f'Noninteger sample column: {name}')
    return numeric.astype(np.int64)


def close_maps(*arrays):
    for array in arrays:
        mapping = getattr(array, '_mmap', None)
        if mapping is not None:
            mapping.close()


def validate_predictions(folder, condition, ids, target, n, c, chunk):
    z = np.load(safe_file(folder, f'{condition}/logits.npy'), mmap_mode='r', allow_pickle=False)
    try:
        if z.shape != (n, c) or z.dtype != np.float32:
            raise ValueError(f'{folder}: incompatible {condition} logits shape/dtype')
        rows = pd.read_csv(safe_file(folder, f'{condition}/predictions.csv'), dtype={'sample_id': str}, keep_default_na=False)
        required = {'row', 'sample_id', 'target', 'prediction', 'correct', 'margin', 'normalized_margin', 'true_class_margin'}
        if not required <= set(rows) or len(rows) != n:
            raise ValueError(f'{folder}: incomplete sample predictions')
        pred, correct = integer_column(rows, 'prediction'), integer_column(rows, 'correct')
        if (not np.array_equal(integer_column(rows, 'row'), np.arange(n))
                or not np.array_equal(rows.sample_id.to_numpy(), ids)
                or not np.array_equal(integer_column(rows, 'target'), target)
                or np.any(pred < 0) or np.any(pred >= c)
                or not np.array_equal(correct, (pred == target).astype(int))):
            raise ValueError(f'{folder}: invalid sample identity/prediction/target pairing')
        tied, disagreements = 0, 0
        for start in range(0, n, chunk):
            end = min(n, start+chunk)
            a = np.asarray(z[start:end], dtype=np.float64)
            if not np.isfinite(a).all() or np.any(a.std(axis=1) <= 0):
                raise ValueError(f'{folder}: nonfinite or zero-spread logits')
            ii = np.arange(end-start)
            maximum = a.max(axis=1)
            if not np.all(a[ii, pred[start:end]] == maximum):
                raise ValueError(f'{folder}: stored winner is not maximal')
            tied += int(((a == maximum[:, None]).sum(axis=1) > 1).sum())
            disagreements += int((a.argmax(axis=1) != pred[start:end]).sum())
            second = np.partition(a, -2, axis=1)[:, -2]
            margin = maximum-second
            true_value = a[ii, target[start:end]].copy()
            a[ii, target[start:end]] = -np.inf
            true_margin = true_value-a.max(axis=1)
            for key, expected in [('margin', margin), ('normalized_margin', margin/z[start:end].astype(np.float64).std(axis=1)),
                                  ('true_class_margin', true_margin)]:
                recorded = pd.to_numeric(rows[key].iloc[start:end], errors='raise').to_numpy(dtype=np.float64)
                if not np.isfinite(recorded).all() or not np.allclose(recorded, expected, rtol=1e-6, atol=1e-7):
                    raise ValueError(f'{folder}: inconsistent {condition} {key}')
        return pred, dict(tied_maxima=tied, stored_vs_numpy_argmax_disagreements=disagreements)
    finally:
        close_maps(z)


def check_pair(folder, serial, dataset, model, mode, chunk=256, keep_predictions=False):
    path = safe_file(folder, 'manifest.json')
    meta = json.loads(path.read_text(encoding='utf-8'))
    expected = dict(status='complete', serial=serial, dataset=dataset, model=model, checkpoint_mode=mode)
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError(f'{folder}: incomplete or incompatible manifest identity')
    quant = meta.get('quantization', {})
    if any(quant.get(k, 'missing') != v for k, v in
           [('weight_bits', 3), ('weight_group_size', 128), ('activation_bits', None)]):
        raise ValueError(f'{folder}: expected weight-only W3/G128')
    if (type(meta.get('n_samples')) is not int or meta['n_samples'] < core.FOLDS
            or meta.get('logit_dtype') != 'float32' or type(meta.get('amp_fp16')) is not bool
            or type(meta.get('seed')) is not int or not isinstance(meta.get('preprocessing'), dict)
            or not isinstance(meta.get('checkpoint'), str) or not meta['checkpoint']
            or not isinstance(meta.get('test_csv'), str) or not meta['test_csv']
            or quant.get('compute_dtype') not in ('float16', 'float32', 'bfloat16')
            or not isinstance(quant.get('exclude'), list) or any(not isinstance(v, str) for v in quant['exclude'])):
        raise ValueError(f'{folder}: incomplete evaluation/precision metadata')
    for key in ('checkpoint_sha256', 'test_csv_sha256', 'cfg_sha256', 'code_sha256', 'hqq_source_sha256', 'samples_sha256'):
        if not re.fullmatch(r'[0-9a-fA-F]{64}', str(meta.get(key, ''))):
            raise ValueError(f'{folder}: missing source identity {key}')
    files = meta.get('files', {})
    hashes = {'manifest.json': sha256_file(path)}
    stamps = {}
    for name in USED_FILES:
        source = safe_file(folder, name)
        if not source.is_file() or sha256_file(source) != files.get(name):
            raise ValueError(f'{folder}: missing file or checksum mismatch: {name}')
        hashes[name] = files[name]
    for name in hashes:
        stat = safe_file(folder, name).stat()
        stamps[name] = (stat.st_size, stat.st_mtime_ns)
    samples = json.loads(safe_file(folder, 'samples.json').read_text(encoding='utf-8'))
    ids, targets = samples.get('sample_ids'), samples.get('targets')
    n = meta['n_samples']
    if (not isinstance(ids, list) or not isinstance(targets, list) or len(ids) != n or len(targets) != n
            or any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != n
            or any(type(v) is not int or v < 0 for v in targets)
            or meta['samples_sha256'] != digest_json(dict(ids=ids, targets=targets))):
        raise ValueError(f'{folder}: invalid sample identity/digest')
    target = np.asarray(targets, dtype=np.int64)
    z = np.load(safe_file(folder, 'fp/logits.npy'), mmap_mode='r', allow_pickle=False)
    try:
        if z.ndim != 2 or z.shape[1] < 2 or max(target) >= z.shape[1]:
            raise ValueError(f'{folder}: invalid class dimension/targets')
        c = z.shape[1]
    finally:
        close_maps(z)
    fp_pred, fp_ties = validate_predictions(folder, 'fp', ids, target, n, c, chunk)
    q_pred, q_ties = validate_predictions(folder, 'w3_g128', ids, target, n, c, chunk)
    item = dict(folder=folder, dataset=dataset, model=model, mode=mode, meta=meta, num_classes=c,
                hashes=hashes, stamps=stamps, fp_ties=fp_ties, quant_ties=q_ties)
    if keep_predictions:
        item.update(ids=np.asarray(ids), target=target, fp_pred=fp_pred, quant_pred=q_pred)
    return item


def preflight(args):
    if args.serial <= 0 or args.chunk_size <= 0 or args.threads <= 0 or args.seed < 0:
        raise ValueError('serial/chunk-size/threads must be positive; seed must be nonnegative')
    if not set(args.modes) <= set(MODES):
        raise ValueError('modes must be ft, fz or cal')
    root, serial_root, output = roots(args)
    selections = list(itertools.product(args.datasets, args.models, args.modes))
    missing = [f'{d}/{m}/{k}' for d, m, k in selections if not (serial_root/d/m/k/'manifest.json').is_file()]
    if missing:
        raise ValueError(f'Missing selected source pairs ({len(missing)}): {", ".join(missing[:12])}')
    pairs = []
    for dataset, model, mode in selections:
        folder = serial_root/dataset/model/mode
        if serial_root not in folder.resolve().parents or folder.resolve() in (output, *output.parents):
            raise ValueError(f'Unsafe source/output link or overlap: {folder}')
        pair = check_pair(folder, args.serial, dataset, model, mode, args.chunk_size)
        pairs.append(pair)
        print(f'CHECKED {dataset}/{model}/{mode}: N={pair["meta"]["n_samples"]} C={pair["num_classes"]}', flush=True)
    for model, mode in itertools.product(args.models, args.modes):
        group = [p for p in pairs if p['model'] == model and p['mode'] == mode]
        signatures = {digest_json(dict(amp_fp16=p['meta']['amp_fp16'], quantization=p['meta']['quantization'],
                                      code=p['meta']['code_sha256'], hqq=p['meta']['hqq_source_sha256'])) for p in group}
        if len(signatures) != 1:
            raise ValueError(f'Mixed export/quantization protocol within {model}/{mode}; narrow the dataset selection')
    return pairs, output


def coverage_rows(pairs, serial):
    rows = []
    for p in pairs:
        m = p['meta']
        rows.append(dict(serial=serial, dataset_name=p['dataset'], model_name=p['model'], checkpoint_mode=p['mode'],
                         n_samples=m['n_samples'], num_classes=p['num_classes'], source=str(p['folder']),
                         checkpoint=m['checkpoint'], checkpoint_sha256=m['checkpoint_sha256'],
                         test_csv=m['test_csv'], test_csv_sha256=m['test_csv_sha256'], inference_seed=m['seed'],
                         amp_fp16=m['amp_fp16'], hqq_compute_dtype=m['quantization']['compute_dtype'],
                         pooling=m.get('pooling'), preprocessing=json.dumps(m['preprocessing'], sort_keys=True),
                         code_sha256=m['code_sha256'], fp_tied_maxima=p['fp_ties']['tied_maxima'],
                         quant_tied_maxima=p['quant_ties']['tied_maxima']))
    return rows


def analyze_pair(pair, output, args):
    folder, meta = pair['folder'], pair['meta']
    for name, expected in pair['stamps'].items():
        stat = safe_file(folder, name).stat()
        if (stat.st_size, stat.st_mtime_ns) != expected:
            raise ValueError(f'Source changed after preflight: {folder}/{name}')
    samples = json.loads((folder/'samples.json').read_text(encoding='utf-8'))
    target = np.asarray(samples['targets'], dtype=np.int64)
    fp_pred = pd.read_csv(folder/'fp/predictions.csv', usecols=['prediction']).prediction.to_numpy(dtype=np.int64)
    q_pred = pd.read_csv(folder/'w3_g128/predictions.csv', usecols=['prediction']).prediction.to_numpy(dtype=np.int64)
    fp = np.load(folder/'fp/logits.npy', mmap_mode='r', allow_pickle=False)
    quant = np.load(folder/'w3_g128/logits.npy', mmap_mode='r', allow_pickle=False)
    try:
        n, c = fp.shape
        folds = core.assign_folds(target, seed=args.seed)
        means = core.crossfit_bias(fp, quant, folds, chunk=args.chunk_size)
        rng = np.random.default_rng(args.seed+11)
        shuffled = np.stack([means[:, rng.permutation(c)] for _ in range(core.SHUFFLES)])
        margin, counts = np.empty(n), np.empty((n, 3), dtype=np.int32)
        fixed_pred, lower, upper = (np.empty(n, dtype=np.int32) for _ in range(3))
        shuffle_correct = np.zeros(core.SHUFFLES, dtype=np.int64)
        for start in range(0, n, args.chunk_size):
            end = min(n, start+args.chunk_size)
            a = fp[start:end].astype(np.float64)
            # Remove the stored winner before selecting another tied/second maximum.
            a[np.arange(end-start), fp_pred[start:end]] = -np.inf
            second = a.argmax(axis=1)
            margin[start:end], counts[start:end] = core.competitor_features(fp[start:end], fp_pred[start:end], second)
            z, f = quant[start:end].astype(np.float64), folds[start:end]
            fixed_pred[start:end] = core.predict_preserving_ties(z-means[f], q_pred[start:end])
            for shuffle in range(core.SHUFFLES):
                corrected = core.predict_preserving_ties(z-shuffled[shuffle, f], q_pred[start:end])
                shuffle_correct[shuffle] += (corrected == target[start:end]).sum()
            lower[start:end], upper[start:end] = core.winner_rank_bounds(fp[start:end], q_pred[start:end])
        fp_correct, q_correct, fixed_correct = fp_pred == target, q_pred == target, fixed_pred == target
        swap, damage = q_pred != fp_pred, fp_correct & ~q_correct
        class_n = np.bincount(target, minlength=c)
        def macro(correct):
            return np.mean(np.bincount(target, weights=correct, minlength=c)[class_n > 0]/class_n[class_n > 0])
        gain = 100*(fixed_correct.mean()-q_correct.mean())
        shuffle_gain = 100*(shuffle_correct/n-q_correct.mean())
        identity = dict(serial=args.serial, dataset_name=pair['dataset'], model_name=pair['model'], checkpoint_mode=pair['mode'])
        bias = dict(**identity, n_samples=n, num_classes=c, fp_acc_pct=100*fp_correct.mean(),
                    quant_acc_pct=100*q_correct.mean(), corrected_acc_pct=100*fixed_correct.mean(), gain_pp=gain,
                    macro_gain_pp=100*(macro(fixed_correct)-macro(q_correct)), shuffle_mean_gain_pp=shuffle_gain.mean(),
                    shuffle_min_gain_pp=shuffle_gain.min(), shuffle_max_gain_pp=shuffle_gain.max(),
                    specificity_pp=gain-shuffle_gain.mean(), improved_n=int((fixed_correct & ~q_correct).sum()),
                    harmed_n=int((~fixed_correct & q_correct).sum()), damage_n=int(damage.sum()),
                    corrected_damage_n=int((fp_correct & ~fixed_correct).sum()), rescue_n=int((~fp_correct & q_correct).sum()),
                    corrected_rescue_n=int((~fp_correct & fixed_correct).sum()), fp_correct_n=int(fp_correct.sum()),
                    quant_correct_n=int(q_correct.sum()), corrected_correct_n=int(fixed_correct.sum()), bias_rms=float(np.sqrt(np.mean(means**2))))
        ranks = dict(**identity, swaps_n=int(swap.sum()), damage_n=int(damage.sum()))
        for label, mask in [('swap', swap), ('damage', damage)]:
            beyond = mask & (lower >= 3)
            ranks[f'{label}_definitely_beyond_top2_n'] = int(beyond.sum())
            ranks[f'{label}_definitely_beyond_top2_pct'] = 100*beyond.sum()/mask.sum() if mask.any() else np.nan
            ranks[f'{label}_rank2_3_tie_ambiguous_n'] = int((mask & (lower < 3) & (upper >= 3)).sum())
        fold_rows = []
        for fold in range(core.FOLDS):
            held = folds == fold
            train_class_n = np.bincount(target[~held], minlength=c)
            fold_rows.append(dict(**identity, fold=fold, n_test=int(held.sum()), n_train=int((~held).sum()),
                                  observed_classes_missing_train=int(((class_n > 0) & (train_class_n == 0)).sum()),
                                  nominal_classes_missing_train=int((train_class_n == 0).sum()),
                                  gain_pp=100*(fixed_correct[held].mean()-q_correct[held].mean())))
        features = core.model_features(margin, counts, c)
        pair_output = output/pair['model']/pair['mode']/pair['dataset']
        pair_output.mkdir(parents=True)
        np.savez_compressed(pair_output/'bias_vectors.npz', bias=means,
                            n_train=np.array([(folds != f).sum() for f in range(core.FOLDS)]))
        if args.save_sample_results:
            np.savez_compressed(pair_output/'sample_results.npz', sample_id=np.asarray(samples['sample_ids']),
                                target=target, folds=folds, fp_pred=fp_pred, quant_pred=q_pred, corrected_pred=fixed_pred,
                                features=features, fp_correct=fp_correct, swap=swap, damage=damage,
                                quant_winner_fp_rank_lower=lower, quant_winner_fp_rank_upper=upper)
        print(f'ANALYZED {pair["dataset"]}/{pair["model"]}/{pair["mode"]}: gain={gain:+.4f} pp', flush=True)
        return bias, ranks, fold_rows, dict(features=features, fp_correct=fp_correct, labels=dict(swap=swap, damage=damage))
    finally:
        close_maps(fp, quant)


def run_predictions(data, model, mode, args, output):
    names = sorted(data)
    identity = dict(model_name=model, checkpoint_mode=mode)
    rows, fits, statuses = [], [], []
    splits = [('lodo', [n for n in names if n != held], [held]) for held in names] if len(names) >= 3 else []
    soy, nonsoy = [n for n in names if n in core.SOY], [n for n in names if n not in core.SOY]
    if len(soy) >= 2 and len(nonsoy) >= 2:
        splits += [('nonsoy_to_soy', nonsoy, soy), ('soy_to_nonsoy', soy, nonsoy)]
    else:
        statuses.append(dict(**identity, scope='family_holdout', outcome='both', dataset_name='all',
                             status='not_applicable', reason='requires_at_least_two_soy_and_two_nonsoy_datasets'))
    if not splits:
        statuses.append(dict(**identity, scope='lodo', outcome='both', dataset_name='all',
                             status='incomplete', reason='too_few_datasets: requires at least 3'))
    with threadpool_limits(limits=args.threads):
        for outcome in ('damage', 'swap'):
            for scope, train, test in splits:
                pending_rows, pending_fits = [], []
                try:
                    for expanded in (False, True):
                        result, fit = core.predict_dataset_split(data, outcome, train, test, expanded)
                        pending_rows.extend(dict(**identity, scope=scope, **r) for r in result)
                        pending_fits.append(dict(**identity, scope=scope, outcome=outcome, expanded=expanded, **fit))
                except core.IncompletePrediction as exc:
                    for name in test:
                        statuses.append(dict(**identity, scope=scope, outcome=outcome, dataset_name=name,
                                             status='incomplete', reason=str(exc)))
                    print(f'INCOMPLETE {model}/{mode} {scope} {outcome}: {exc}', flush=True)
                    continue
                rows.extend(pending_rows)
                fits.extend(pending_fits)
                for name in test:
                    statuses.append(dict(**identity, scope=scope, outcome=outcome, dataset_name=name, status='ok', reason=''))
                print(f'PREDICTED {model}/{mode} {scope} {outcome}: {",".join(test)}', flush=True)
    write_json(output/model/mode/'model_fits.json', fits)
    scores = pd.DataFrame(rows, columns=SCORE_COLUMNS)
    comparisons = []
    if len(scores):
        keys = ['model_name', 'checkpoint_mode', 'scope', 'dataset_name', 'outcome']
        a = scores[scores.model == 'margin_only'].set_index(keys)
        b = scores[scores.model == 'margin_plus_competitors'].set_index(keys)
        table = a[['n_test', 'event_rate']].copy()
        for metric in ('brier', 'log_loss', 'auroc'):
            table[f'margin_{metric}'] = a[metric]
            table[f'competitors_{metric}'] = b[metric]
            table[f'delta_{metric}'] = b[metric]-a[metric]
        comparisons = table.reset_index().to_dict('records')
    return rows, comparisons, statuses


def summarize(output, pairs, args):
    bias = pd.read_csv(output/'bias_results.csv')
    comparisons = pd.read_csv(output/'heldout_prediction_comparisons.csv')
    statuses = pd.read_csv(output/'prediction_status.csv', keep_default_na=False)
    summary = []
    lines = [f'# W3/G128 bias and competitor diagnostics: serial {args.serial}', '',
             f'Complete selected matrix: {len(args.datasets)} datasets x {len(args.models)} models x {len(args.modes)} modes = {len(pairs)} pairs.',
             'Existing test-loader logits only. Precision, pooling, preprocessing and declared checkpoint/test-CSV identities are in coverage.csv.',
             'No new inference. Bias correction is retrospective five-fold cross-fitting, not an independently calibrated accuracy benchmark.', '',
             '| Model | Mode | Bias gain (pp) | Improved datasets | Shuffle gain (pp) | LODO damage Brier change | Prediction status |',
             '|---|---|---:|---:|---:|---:|---|']
    for model, mode in itertools.product(args.models, args.modes):
        b = bias[(bias.model_name == model) & (bias.checkpoint_mode == mode)]
        base = dict(model_name=model, checkpoint_mode=mode, n_datasets=len(b), mean_gain_pp=b.gain_pp.mean(),
                    improved_datasets=int((b.gain_pp > 0).sum()), mean_shuffle_gain_pp=b.shuffle_mean_gain_pp.mean(),
                    mean_specificity_pp=b.specificity_pp.mean())
        for outcome in ('damage', 'swap'):
            p = comparisons[(comparisons.model_name == model) & (comparisons.checkpoint_mode == mode)
                            & (comparisons.scope == 'lodo') & (comparisons.outcome == outcome)]
            s = statuses[(statuses.model_name == model) & (statuses.checkpoint_mode == mode)
                         & (statuses.scope == 'lodo') & statuses.outcome.isin([outcome, 'both'])]
            complete = len(p) == len(args.datasets) and len(s) == len(args.datasets) and (s.status == 'ok').all()
            row = dict(**base, outcome=outcome, prediction_status='complete' if complete else 'incomplete',
                       heldout_datasets=len(p), mean_delta_brier=p.delta_brier.mean() if complete else np.nan,
                       mean_delta_log_loss=p.delta_log_loss.mean() if complete else np.nan,
                       mean_delta_auroc=p.delta_auroc.mean() if complete else np.nan,
                       auroc_defined_datasets=int(p.delta_auroc.notna().sum()),
                       brier_improved_datasets=int((p.delta_brier < 0).sum()))
            summary.append(row)
            if outcome == 'damage':
                value = f'{row["mean_delta_brier"]:+.6f}' if complete else 'NA'
                lines.append(f'| {model} | {mode} | {base["mean_gain_pp"]:+.4f} | {base["improved_datasets"]}/{len(b)} | {base["mean_shuffle_gain_pp"]:+.4f} | {value} | {row["prediction_status"]} |')
    pd.DataFrame(summary).to_csv(output/'setting_summary.csv', index=False)
    lines += ['', 'Positive bias gain favors correction. Negative Brier/log-loss change favors competitors; positive AUROC change favors competitors.',
              'All means weight datasets equally within each model/mode. Different modes/models reuse datasets and are not independent replications.',
              'Primary LODO means are NA if any selected dataset lacks a valid paired prediction result. Undefined AUROC is retained explicitly.', '',
              '## Scientific and validation boundaries', '',
              '- Five stratified folds; bias is estimated from paired logits on the other four folds, strength fixed at 1. Five shuffled-class controls.',
              '- Fixed competitors within 0.25/0.5/1 FP logit std; nonlinear margin basis plus log class count in both risk models.',
              '- Logistic L2 C=100, train-only standardization, each training dataset has total weight 1. No test-based tuning.',
              '- Damage prediction is conditional on FP-correct images. Swap models include known FP correctness in both comparators as an analysis control.',
              '- Soybean-family holdouts are exploratory and appear in the detailed comparison/status files when applicable.',
              '- Stored maximal tie winners are preserved; rank-2/3 ties do not count as definitely beyond top two.',
              '- Pooling/preprocessing can legitimately vary with the trained checkpoint; coverage records these differences as potential confounders.',
              '- CAL logits retain the exporter\'s crop/flip aggregation. This analysis uses logits only, not raw-image backbone layer features.',
              '- Checkpoint and test-CSV identities are exporter-declared hashes; checkpoints/images are not reopened or inference revalidated.',
              '- Checksums cover manifest, samples, both logits and prediction CSVs. Unused layer feature files are not read or rehashed.',
              '- Dataset properties, checkpoint training and seed effects remain confounded. No causal or significance claim.', '',
              '## Outputs', '',
              'coverage.csv, source_audit.csv, protocol.json, bias_results.csv, fold_audit.csv, winner_ranks.csv,',
              'heldout_prediction_scores.csv, heldout_prediction_comparisons.csv, prediction_status.csv, setting_summary.csv.',
              'Per-setting model_fits.json and per-dataset bias_vectors.npz preserve fitted diagnostics.',
              'Optional --save-sample-results retains derived features, labels, folds and predictions, excluding raw logits.', '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def run(args):
    _, _, candidate_output = roots(args)
    if not args.preflight_only and candidate_output.exists():
        raise FileExistsError(f'Preserving existing output: {candidate_output}. Choose a new --output-dir.')
    pairs, output = preflight(args)
    print(f'PREFLIGHT PASSED: {len(pairs)} complete paired exports', flush=True)
    if args.preflight_only:
        return
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'protocol.json', dict(
        started_utc=datetime.now(timezone.utc).isoformat(), serial=args.serial,
        datasets=args.datasets, models=args.models, modes=args.modes, seed=args.seed,
        folds=core.FOLDS, shuffles=core.SHUFFLES, shuffle_seed=args.seed+11, bias_strength=1,
        thresholds=core.THRESHOLDS.tolist(), margin_knots=core.KNOTS.tolist(),
        fitting=dict(C=100, max_iter=1000, tol=1e-7, solver='lbfgs', dataset_total_weight=1,
                     train_only_standardization=True),
        validation='separate model/mode LODO, exploratory soybean-family holdout',
        primary='damage Brier among FP-correct samples; bias accuracy gain in pp',
        tie_rule='prefer original quant winner if maximal, else lowest index',
        retrospective_test_crossfit=True, chunk_size=args.chunk_size, threads=args.threads,
        source_hash_scope=['manifest.json', *USED_FILES],
        script_hashes={p.name: sha256_file(p) for p in [Path(__file__), Path(core.__file__)]},
        runtime=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                     pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)))
    pd.DataFrame(coverage_rows(pairs, args.serial)).to_csv(output/'coverage.csv', index=False)
    audit = [dict(dataset_name=p['dataset'], model_name=p['model'], checkpoint_mode=p['mode'],
                  path=str(p['folder']/name), sha256=checksum) for p in pairs for name, checksum in p['hashes'].items()]
    pd.DataFrame(audit).to_csv(output/'source_audit.csv', index=False)
    bias, ranks, folds, scores, comparisons, statuses = [], [], [], [], [], []
    for model, mode in itertools.product(args.models, args.modes):
        data = {}
        for pair in pairs:
            if (pair['model'], pair['mode']) != (model, mode):
                continue
            b, r, f, derived = analyze_pair(pair, output, args)
            bias.append(b)
            ranks.append(r)
            folds.extend(f)
            data[pair['dataset']] = derived
        s, c, status = run_predictions(data, model, mode, args, output)
        scores.extend(s)
        comparisons.extend(c)
        statuses.extend(status)
        del data
    for name, rows, columns in [('bias_results', bias, None), ('winner_ranks', ranks, None), ('fold_audit', folds, None),
                                ('heldout_prediction_scores', scores, SCORE_COLUMNS),
                                ('heldout_prediction_comparisons', comparisons, COMPARISON_COLUMNS),
                                ('prediction_status', statuses, None)]:
        pd.DataFrame(rows, columns=columns).to_csv(output/f'{name}.csv', index=False)
    summarize(output, pairs, args)
    # Detect source mutation during a long run before accepting completion.
    for p in pairs:
        for name, expected in p['hashes'].items():
            if sha256_file(safe_file(p['folder'], name)) != expected:
                raise ValueError(f'Source changed during analysis: {p["folder"]}/{name}')
    write_json(output/'completion.json', dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
        source_pairs=len(pairs), all_used_source_hashes_rechecked=True,
        output_hashes={p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob('*') if p.is_file()}))
    print(f'COMPLETE: {output / "REPORT.md"}', flush=True)


def main():
    args = build_parser().parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == '__main__':
    main()
