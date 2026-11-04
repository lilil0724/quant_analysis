"""Pair the complete W6A4 cohort with compatible unquantized AMP baselines."""
import itertools
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from legacy_hqq.recover_baselines import context, differences, DATASETS, MODELS, MODES

SETTINGS = dict(hqq=True, serial=20007, hqq_nbits=6, hqq_act_bits=4,
                hqq_group_size=32, hqq_act_group_size=32, hqq_scope='extended',
                hqq_exclude='', hqq_compute_dtype='float16', hqq_quant_zero=True,
                hqq_quant_scale=True, debugging=False, fp16=True)
KEYS = ['dataset', 'model', 'mode']


def prepare_pairs(records, targets):
    expected = set(itertools.product(DATASETS, MODELS, MODES))
    baselines = {}
    for record in records:
        if record['config'].get('hqq') is False:
            key = context(record)
            if key in baselines:
                raise ValueError(f'Duplicate baseline: {key}')
            baselines[key] = record
    if set(baselines) != expected:
        raise ValueError('Baseline cohort must contain the complete 144-context matrix')
    target_keys = [context(r) for r in targets]
    if len(target_keys) != 144 or set(target_keys) != expected:
        raise ValueError('W6A4 cohort has missing or duplicate contexts')
    rows = []
    for record, key in zip(targets, target_keys):
        c, baseline = record['config'], baselines[key]
        invalid = [name for name, value in SETTINGS.items() if name not in c or c[name] != value]
        if invalid or record['state'] != 'finished':
            raise ValueError(f'Invalid W6A4 run {record["id"]}: {invalid or ["state"]}')
        b = baseline['config']
        quant, fp = record['summary']['hqq_top1'], baseline['summary']['top1']
        if (not isinstance(quant, (int, float)) or not isinstance(fp, (int, float))
                or not math.isfinite(quant) or not math.isfinite(fp) or not 0 <= quant <= 100
                or not 0 < fp <= 100 or b.get('fp16') is not True
                or baseline['state'] != 'finished' or b.get('debugging', False)):
            raise ValueError(f'Invalid metrics or baseline: {key}')
        mismatch, unrecorded = differences(c, b)
        rows.append(dict(dataset=key[0], model=key[1], mode=key[2],
            w6a4_run_id=record['id'], w6a4_run_url=record['url'], baseline_run_id=baseline['id'],
            baseline_run_url=baseline['url'], baseline_serial=b['serial'],
            w6a4_top1_pct=quant, baseline_top1_pct=fp,
            accuracy_ratio=quant/fp if not mismatch else np.nan,
            accuracy_drop_pp=fp-quant if not mismatch else np.nan,
            accepted=not mismatch, mismatch_fields=';'.join(mismatch),
            shared_unrecorded_fields=';'.join(unrecorded),
            w6a4_classifier=c.get('classifier'), baseline_classifier=b.get('classifier'),
            batch_size_matches=c.get('batch_size') == b.get('batch_size'),
            checkpoint_bytes_verified=bool(c.get('checkpoint_sha256') and c.get('checkpoint_sha256') == b.get('checkpoint_sha256')),
            baseline_precision='unquantized_AMP'))
    return pd.DataFrame(rows).sort_values(KEYS)


def verify_pairs(frame, reference):
    actual = frame.sort_values(KEYS).reset_index(drop=True)
    expected = reference.sort_values(KEYS).reset_index(drop=True)
    if len(actual) != 144 or not actual.accepted.all() or len(expected) != 144:
        raise ValueError('Expected 144 accepted pairs')
    pd.testing.assert_frame_equal(actual[KEYS + ['w6a4_run_id', 'baseline_run_id']],
                                  expected[KEYS + ['w6a4_run_id', 'baseline_run_id']])
    np.testing.assert_allclose(actual.accuracy_ratio, actual.w6a4_top1_pct/actual.baseline_top1_pct, rtol=0, atol=1e-12)
    fields = ['w6a4_top1_pct', 'baseline_top1_pct', 'accuracy_ratio', 'accuracy_drop_pp']
    np.testing.assert_allclose(actual[fields], expected[fields], rtol=0, atol=1e-12)
    return dict(contexts=144, usable_ratios=144, unpaired=0, formula_verified=True,
                reference_run_ids_match=True, reference_values_match=True, tolerance=1e-12)


def plot_heatmap(frame, output):
    wide = frame.pivot(index='dataset', columns=['model','mode'], values='accuracy_ratio').reindex(
        index=DATASETS, columns=pd.MultiIndex.from_product([MODELS, MODES]))
    wide.to_csv(output.parent / 'tables/heatmap_values.csv')
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=max(1.05, float(frame.accuracy_ratio.max())))
    cmap = plt.get_cmap('RdBu').copy()
    cmap.set_bad('#e5e7eb')
    fig, axes = plt.subplots(1, 3, figsize=(16,9), sharey=True, layout='constrained')
    for axis, model, label in zip(axes, MODELS, ['ViT-B/16','Swin-B','BEiTv2-B']):
        matrix = wide[model].to_numpy()
        im = axis.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect='auto')
        axis.set_title(label)
        axis.set_xticks(range(3), [m.upper() for m in MODES])
        axis.set_yticks(range(len(DATASETS)), DATASETS)
        for i, j in np.ndindex(matrix.shape):
            value = matrix[i,j]
            rgb = cmap(norm(value))[:3] if np.isfinite(value) else (1,1,1)
            light = .2126*rgb[0] + .7152*rgb[1] + .0722*rgb[2]
            axis.text(j, i, f'{value:.3f}' if np.isfinite(value) else 'N/A', ha='center', va='center', fontsize=8,
                      color='white' if light < .5 else '#374151')
    fig.colorbar(im, ax=axes, label='W6A4 accuracy / matched unquantized AMP accuracy', shrink=.75)
    fig.suptitle('Serial 20007: recovered baseline (W6A4, weight G32 / activation G32, extended)')
    fig.supxlabel('144/144 usable; all corrected baselines available')
    for extension in ('png','svg','pdf'):
        fig.savefig(output / f'acc_ratio_heatmap.{extension}', dpi=180)
    plt.close(fig)
