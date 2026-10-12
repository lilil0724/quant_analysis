"""Reproducible exploratory audit of current compact quantization results.

Run from the quant_analysis root with opencode_env Python. No inference,
downloads, or edits to source data. Outputs are final research deliverables.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import itertools
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import summarize_quant
from dataset_robustness.analyze import prepare_cells, analyze

SOURCE = ROOT / 'five_dataset_features/data/w3_g128_results'
DISTANCES = ROOT / 'five_dataset_features/analysis/serial_20004_20261010/class_distances/final_class_distances.csv'
OUT = ROOT / 'five_dataset_features/analysis/dataset_relationship_20261010'
SOY = {'soyageing', 'soygene', 'soyglobal', 'soylocal'}
PLANT = SOY | {'cotton'}
KEY = ['dataset', 'model', 'mode']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rho(x, y):
    x, y = np.asarray(x), np.asarray(y)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 4 or np.ptp(x[good]) == 0 or np.ptp(y[good]) == 0:
        return np.nan
    return float(spearmanr(x[good], y[good]).statistic)


def partial(x, y, controls):
    # Descriptive sensitivity, not a significance or causal analysis.
    a, b = rankdata(x), rankdata(y)
    z = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in controls])
    ar = a - z @ np.linalg.lstsq(z, a, rcond=None)[0]
    br = b - z @ np.linalg.lstsq(z, b, rcond=None)[0]
    return float(np.corrcoef(ar, br)[0, 1])


def concordance(panel):
    # Kendall W, including average tied ranks and tie correction.
    n, m = panel.shape
    ranks = np.column_stack([rankdata(panel[:, j]) for j in range(m)])
    ss = np.sum((ranks.sum(axis=1) - m * (n + 1) / 2) ** 2)
    ties = 0
    for j in range(m):
        _, counts = np.unique(panel[:, j], return_counts=True)
        ties += np.sum(counts ** 3 - counts)
    return float(12 * ss / (m * m * (n ** 3 - n) - m * ties))


def variance_parts(panel):
    grand = panel.mean()
    dataset_ss = panel.shape[1] * np.sum((panel.mean(axis=1) - grand) ** 2)
    context_ss = panel.shape[0] * np.sum((panel.mean(axis=0) - grand) ** 2)
    residual = panel - panel.mean(axis=1)[:, None] - panel.mean(axis=0)[None, :] + grand
    residual_ss = np.sum(residual ** 2)
    total = np.sum((panel - grand) ** 2)
    assert np.isclose(dataset_ss + context_ss + residual_ss, total)
    return dict(dataset_ss_fraction=dataset_ss / total,
                context_ss_fraction=context_ss / total,
                interaction_and_unmeasured_error_ss_fraction=residual_ss / total)


def predict_ols(train, test, columns, target):
    y = train[target].to_numpy()
    if not columns:
        return np.full(len(test), y.mean())
    x, t = train[columns].to_numpy(), test[columns].to_numpy()
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1
    design = np.column_stack([np.ones(len(x)), (x - mean) / std])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = np.column_stack([np.ones(len(t)), (t - mean) / std]) @ beta
    # Independent PRESS identity for the single-dataset folds is checked below.
    return prediction


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE / 'analysis_manifest.json').read_text())
    checked, omitted = [], []
    for rel, expected in manifest['outputs'].items():
        path = SOURCE / rel
        if not path.exists():
            if Path(rel).name not in ('coords_fp.npy', 'coords_w3_g128.npy'):
                raise ValueError(f'Missing compact source {rel}')
            omitted.append(rel)
            continue
        assert digest(path) == expected, rel
        checked.append(rel)
    d = pd.read_csv(SOURCE / 'all_pairs.csv')
    assert len(d) == 144 and not d.duplicated(KEY).any()
    assert set(d.serial) == {20004}
    assert (d.groupby(['model', 'mode']).size() == 16).all()
    assert d.groupby('dataset').n.nunique().eq(1).all()
    d['acc_ratio'] = d.accuracy / d.fp_accuracy
    assert np.allclose(1 - d.acc_ratio, d.relative_loss)
    assert np.allclose((d.damage_count - d.rescue_count) / d.n, d.accuracy_drop)
    assert np.allclose(d.swap_rate * d.net_damage_r / d.fp_accuracy, d.relative_loss)
    for _, row in d.iterrows():
        summary = json.loads((SOURCE / row.dataset / row.model / row['mode'] / 'summary.json').read_text())
        for field in ('n', 'fp_accuracy', 'accuracy', 'damage_count', 'rescue_count', 'swap_rate'):
            assert np.isclose(row[field], summary[field]), (tuple(row[KEY]), field)
    # Use original compact per-pair layer metrics, not prior derived distances.
    final = []
    for _, row in d.iterrows():
        layers = pd.read_csv(SOURCE / row.dataset / row.model / row['mode'] / 'layer_metrics.csv')
        rows = layers[layers.layer.eq('final')].set_index('condition')
        assert len(rows) == 2
        inter, intra = rows.loc['fp', ['d_inter', 'd_intra']]
        pairs = int(rows.loc['fp', 'n_inter_class_pairs'])
        classes = int(round((1 + np.sqrt(1 + 8 * pairs)) / 2))
        assert classes * (classes - 1) // 2 == pairs
        values = {**row[KEY].to_dict(), 'n_classes': classes}
        for condition, prefix in [('fp', 'fp'), ('w3_g128', 'quant')]:
            inter, intra = rows.loc[condition, ['d_inter', 'd_intra']]
            values[f'{prefix}_gap'] = inter - intra
            values[f'{prefix}_s_norm'] = (inter - intra) / (inter + intra)
        final.append(values)
    d = d.merge(pd.DataFrame(final), on=KEY, validate='one_to_one')
    old = pd.read_csv(DISTANCES)
    comparison = d.merge(old[KEY + ['fp_s_norm', 'w3_s_norm']], on=KEY, suffixes=('', '_prior'), validate='one_to_one')
    assert np.allclose(comparison.fp_s_norm, comparison.fp_s_norm_prior)
    assert np.allclose(comparison.quant_s_norm, comparison.w3_s_norm)
    d['conditional_damage'] = d.damage_count / (d.n * d.fp_accuracy)
    d['conditional_rescue'] = d.rescue_count / (d.n * (1 - d.fp_accuracy))
    d['h_score_retention'] = d.w3_g128_h_score_final / d.fp_h_score_final
    d['delta_s_norm'] = d.quant_s_norm - d.fp_s_norm
    d['gap_retention'] = d.quant_gap / d.fp_gap
    d['log_q10'] = np.log(d.fp_normalized_margin_q10)
    d['log_classes'] = np.log(d.n_classes)
    d['log_test_n'] = np.log(d.n)
    d['context'] = d.model + '/' + d['mode']
    d['family_block'] = d.dataset.where(~d.dataset.isin(SOY), 'soy_family')
    d.to_csv(OUT / 'paired_inputs.csv', index=False)
    cohorts = {'all_9': d,
               'without_beit_fz_8': d[~(d.model.eq('beitv2_base_patch16_224') & d['mode'].eq('fz'))]}
    correlations, stability, pair_agreement, profiles, contrasts = [], [], [], [], []
    predictors = ['fp_accuracy', 'fp_normalized_margin_q10', 'fp_s_norm',
                  'fp_gap', 'quant_s_norm', 'delta_s_norm', 'gap_retention',
                  'final_cka', 'h_score_retention', 'fp_h_score_final',
                  'fp_norm_p99_p50', 'n_classes', 'n', 'swap_rate', 'net_damage_r']
    for cohort, frame in cohorts.items():
        profile = frame.groupby('dataset', as_index=False).agg(
            contexts=('context', 'size'), fp_accuracy=('fp_accuracy', 'mean'),
            accuracy=('accuracy', 'mean'), acc_ratio=('acc_ratio', 'mean'),
            accuracy_drop=('accuracy_drop', 'mean'), swap_rate=('swap_rate', 'mean'),
            net_damage_r=('net_damage_r', 'mean'), conditional_damage=('conditional_damage', 'mean'),
            conditional_rescue=('conditional_rescue', 'mean'), n=('n', 'first'), n_classes=('n_classes', 'first'))
        profile['cohort'] = cohort
        profiles.append(profile)
        for outcome in ['acc_ratio', 'accuracy_drop']:
            panel = frame.pivot(index='dataset', columns='context', values=outcome)
            agreements = []
            for a, b in itertools.combinations(panel.columns, 2):
                value = rho(panel[a], panel[b])
                agreements.append(value)
                pair_agreement.append(dict(cohort=cohort, outcome=outcome, a=a, b=b, rho=value))
            stability.append(dict(cohort=cohort, outcome=outcome,
                                  kendall_w=concordance(panel.to_numpy()),
                                  median_pair_spearman=np.median(agreements),
                                  min_pair_spearman=np.min(agreements),
                                  max_pair_spearman=np.max(agreements), **variance_parts(panel.to_numpy())))
        for context, group in frame.groupby('context'):
            for subset, excluded in [('all', set()), ('without_soy', SOY), ('without_soy_cotton', PLANT)]:
                g = group[~group.dataset.isin(excluded)]
                for predictor in predictors:
                    for outcome in ['acc_ratio', 'accuracy_drop', 'swap_rate']:
                        controls = [g.fp_accuracy, g.log_classes, g.log_test_n]
                        correlations.append(dict(cohort=cohort, context=context, subset=subset,
                            n_datasets=len(g), predictor=predictor, outcome=outcome,
                            rho=rho(g[predictor], g[outcome]),
                            partial_fp_accuracy=(partial(g[predictor], g[outcome], [g.fp_accuracy])
                                                 if predictor != 'fp_accuracy' else np.nan),
                            partial_fp_classes_testn=(partial(g[predictor], g[outcome], controls)
                                                     if predictor not in ['fp_accuracy', 'n_classes', 'n'] else np.nan)))
            for label, names in [('soy_only', SOY), ('soy_and_cotton', PLANT)]:
                exposed, other = group[group.dataset.isin(names)], group[~group.dataset.isin(names)]
                contrasts.append(dict(cohort=cohort, context=context, selection=label,
                    n_selected=len(exposed), n_other=len(other),
                    ratio_selected=exposed.acc_ratio.mean(), ratio_other=other.acc_ratio.mean(),
                    ratio_difference=exposed.acc_ratio.mean() - other.acc_ratio.mean(),
                    drop_selected=exposed.accuracy_drop.mean(), drop_other=other.accuracy_drop.mean(),
                    swap_selected=exposed.swap_rate.mean(), swap_other=other.swap_rate.mean(),
                    r_selected=exposed.net_damage_r.mean(), r_other=other.net_damage_r.mean()))
    pd.concat(profiles).to_csv(OUT / 'dataset_profiles.csv', index=False)
    pd.DataFrame(stability).to_csv(OUT / 'rank_stability_variance.csv', index=False)
    pd.DataFrame(pair_agreement).to_csv(OUT / 'context_rank_agreement.csv', index=False)
    pd.DataFrame(correlations).to_csv(OUT / 'within_context_correlations.csv', index=False)
    pd.DataFrame(contrasts).to_csv(OUT / 'soy_cotton_contrasts.csv', index=False)
    specs = {'constant': [], 'fp_accuracy_only': ['fp_accuracy'],
        'baseline_controls': ['fp_accuracy', 'log_classes', 'log_test_n'],
        'raw_q10_only': ['fp_normalized_margin_q10'], 'log_q10_only': ['log_q10'],
        'fp_separation_only': ['fp_s_norm'], 'fp_plus_log_q10': ['fp_accuracy', 'log_q10'],
        'fp_plus_separation': ['fp_accuracy', 'fp_s_norm'],
        'fp_plus_log_q10_separation': ['fp_accuracy', 'log_q10', 'fp_s_norm']}
    predictions, scores = [], []
    for context, g in d.groupby('context'):
        g = g.sort_values('dataset').reset_index(drop=True)
        for scheme, blocks in [('LODO', g.dataset), ('leave_soy_family_out', g.family_block),
                               ('leave_soy_cotton_block_out', g.dataset.where(~g.dataset.isin(PLANT), 'soy_cotton_block'))]:
            for outcome in ['acc_ratio', 'accuracy_drop']:
                for model, columns in specs.items():
                    pred = np.empty(len(g))
                    for block in blocks.unique():
                        mask = blocks.eq(block)
                        pred[mask] = predict_ols(g[~mask], g[mask], columns, outcome)
                    if scheme == 'LODO':
                        x = g[columns].to_numpy() if columns else np.empty((len(g), 0))
                        mean = x.mean(axis=0)
                        scale = x.std(axis=0)
                        scale[scale == 0] = 1
                        z = np.column_stack([np.ones(len(g)), (x - mean) / scale])
                        hat = z @ np.linalg.pinv(z)
                        y = g[outcome].to_numpy()
                        press = y - (y - hat @ y) / (1 - np.diag(hat))
                        assert np.allclose(pred, press, atol=1e-7), (context, model, outcome)
                    for i, row in g.iterrows():
                        predictions.append(dict(context=context, scheme=scheme, outcome=outcome, model=model,
                            dataset=row.dataset, block=blocks.iloc[i], observed=row[outcome], predicted=pred[i],
                            absolute_error=abs(pred[i] - row[outcome]), n_train=int((blocks != blocks.iloc[i]).sum())))
                    scores.append(dict(context=context, scheme=scheme, outcome=outcome, model=model,
                        n_datasets=len(g), mae=np.mean(np.abs(pred - g[outcome])),
                        rmse=np.sqrt(np.mean((pred - g[outcome]) ** 2)),
                        oof_rho=rho(pred, g[outcome]) if columns else np.nan,
                        ratio_outside_0_1_count=int(np.sum((pred < 0) | (pred > 1))) if outcome == 'acc_ratio' else np.nan))
    pd.DataFrame(predictions).to_csv(OUT / 'heldout_predictions.csv', index=False)
    pd.DataFrame(scores).to_csv(OUT / 'heldout_summary.csv', index=False)
    # Reuse the legacy pairing and rank audit functions without generating XLSX.
    raw = pd.read_csv(ROOT / 'data/backbones_quant.csv')
    _, hqq, summary = summarize_quant.prepare_summary(raw, SimpleNamespace(
        serials=None, subset_datasets=None, subset_models=None))
    legacy, excluded = prepare_cells(summary)
    assert len(legacy) == 4320 and len(excluded) == 0
    summary.to_csv(OUT / 'legacy_matched_summary.csv', index=False)
    summarize_quant.configuration_quality(hqq).to_csv(OUT / 'legacy_configuration_quality.csv', index=False)
    tables = analyze(legacy)
    for name in ('within_context_agreement', 'cross_context_agreement', 'metric_agreement', 'baseline_audit'):
        tables[name].to_csv(OUT / ('legacy_' + name + '.csv'), index=False)
    legacy['soy'] = legacy.dataset_name.isin(SOY)
    # First aggregate each dataset, then equal dataset means inside each group.
    ld = legacy.groupby(['dataset_name', 'hqq_nbits', 'soy'], as_index=False).agg(
        mean_ratio=('accuracy_ratio', 'mean'), mean_drop_pp=('accuracy_drop_points', 'mean'))
    ld.groupby(['hqq_nbits', 'soy'], as_index=False).agg(n_datasets=('dataset_name', 'size'),
        mean_ratio=('mean_ratio', 'mean'), mean_drop_pp=('mean_drop_pp', 'mean')).to_csv(OUT / 'legacy_soy_bit_profiles.csv', index=False)
    # Plot only the transparent sensitivity cohort; variation is across
    # contexts, not a confidence interval. No model selected for lowest error.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = cohorts['without_beit_fz_8']
    names = frame.groupby('dataset').acc_ratio.mean().sort_values().index
    short = lambda value: value.replace('beitv2_base_patch16_224', 'BEiT').replace(
        'swin_base_patch4_window7_224_in22k', 'Swin').replace('vit_b16', 'ViT').replace('/', ' ')
    panel = frame.pivot(index='dataset', columns='context', values='acc_ratio').reindex(names)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1.15, 1]})
    im = axes[0].imshow(panel.to_numpy() * 100, aspect='auto', vmin=0, vmax=100, cmap='viridis')
    axes[0].set_yticks(range(len(panel)), panel.index)
    axes[0].set_xticks(range(len(panel.columns)), [short(x) for x in panel.columns], rotation=55, ha='right')
    axes[0].set_title('W3/g128 accuracy retention (%)\n16 datasets; BEiT FZ excluded')
    fig.colorbar(im, ax=axes[0], shrink=.8, label='100 x accuracy ratio')
    score = pd.DataFrame(scores).query("scheme == 'LODO' and outcome == 'acc_ratio'")
    score = score[score.context.isin(panel.columns)]
    selected = ['constant', 'fp_accuracy_only', 'fp_plus_log_q10', 'fp_plus_separation']
    colors = ['#aaaaaa', '#265f9a', '#dc7b23', '#4c9a6f']
    labels = ['Training mean', 'FP accuracy', 'FP accuracy + log Q10', 'FP accuracy + FP separation']
    for j, (model, color, label) in enumerate(zip(selected, colors, labels)):
        values = score[score.model.eq(model)].set_index('context').reindex(panel.columns).mae.to_numpy()
        axes[1].barh(np.arange(8) + (j - 1.5) * .18, values * 100, height=.17, color=color, label=label)
    axes[1].set_yticks(range(8), [short(x) for x in panel.columns])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Held-out MAE (accuracy-ratio percentage points)')
    axes[1].set_title('Leave one whole dataset out\nExploratory OLS; no tuning or clipping')
    axes[1].legend(loc='lower right', fontsize=8)
    axes[1].grid(axis='x', alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / 'dataset_relationship.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    config = dict(date='2026-10-10', source_root=str(SOURCE), output_root=str(OUT),
        source_hashes={str(p): digest(p) for p in [SOURCE / 'analysis_manifest.json', SOURCE / 'all_pairs.csv', DISTANCES, ROOT / 'data/backbones_quant.csv']},
        script_sha256=digest(Path(__file__)), compact_files_verified=len(checked), omitted_coordinates=len(omitted),
        n_pairs=len(d), n_datasets=d.dataset.nunique(), legacy_cells=len(legacy),
        analysis='exploratory retrospective; no significance claims, model tuning, clipping or external validation',
        holdout='OLS with training-fold-only standardization. LODO, whole soybean family, and whole soy+cotton block; dataset equal error weighting.',
        feature_definitions='Q10 pre-quantization normalized logit-margin tail; S=(inter-intra)/(inter+intra); n is TEST count, not training-set size.',
        variance='balanced additive descriptive sums of squares; residual combines interaction and unknown measurement error, no causal variance.',
        limits='raw logits/features/predictions and source checkpoints absent; compact output integrity and algebra verified, source metrics not independently recomputed')
    (OUT / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps(config, indent=2))
    print(pd.DataFrame(stability).to_string(index=False))
    print(pd.DataFrame(scores).query("outcome == 'acc_ratio'").groupby(['scheme', 'model']).mae.mean().unstack().to_string())


if __name__ == '__main__':
    main()
