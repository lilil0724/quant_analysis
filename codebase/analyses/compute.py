"""Compute every analysis table and printed number of the paper from data/ into results/."""
import numpy as np
import pandas as pd
from scipy import stats

from codebase.registry import (BITS, RESULTS, DATASETS, GRID_DATASETS, GROUP_SIZES, DATA, MODEL_KEYS, PERMUTATIONS,
                      PRIMARY, SEED, SETTINGS, SOY_COTTON, group_of)

NUMBERS = []


def put(key, value, dp):
    value = float(value)
    NUMBERS.append({'key': key, 'value': round(value, 6), 'dp': dp})


def load_grid():
    grid = pd.read_csv(DATA / 'grid.csv')
    grid['model'] = grid.model_name.map(MODEL_KEYS)
    grid['mode'] = grid.ckpt_kind
    grid['dataset'] = grid.dataset_name
    grid['bits'] = grid.hqq_nbits
    grid['gs'] = grid.hqq_group_size
    grid['retention'] = grid.hqq_top1 / grid.fp32_top1
    grid['group'] = grid.dataset.map(group_of)
    assert grid.model.notna().all() and set(grid.dataset) == set(DATASETS)
    assert not grid.duplicated(['dataset', 'model', 'mode', 'bits', 'gs']).any()
    return grid


def load_master():
    master = pd.read_csv(DATA / 'dataset_master.csv')
    master['model'] = master.model.map(MODEL_KEYS)
    master['group'] = master.dataset.map(group_of)
    assert len(master) == 144 and not master.duplicated(['dataset', 'model', 'mode']).any()
    return master


def kendall_w(matrix):
    """Kendall's W of the datasets (rows) ranked by each condition (columns), without tie correction."""
    ranks = matrix.rank(axis=0).to_numpy()
    n, k = ranks.shape
    totals = ranks.sum(axis=1)
    return 12 * ((totals - totals.mean()) ** 2).sum() / (k ** 2 * (n ** 3 - n))


def permutation_p(x, y, rng):
    """One-sided p of a Spearman correlation at most the observed one."""
    observed = stats.spearmanr(x, y).statistic
    draws = np.array([stats.spearmanr(rng.permutation(x), y).statistic for _ in range(PERMUTATIONS)])
    return (1 + (draws <= observed).sum()) / (1 + PERMUTATIONS)


def lodo_error(X, y):
    errors = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        coef = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        errors.append((X[i] @ coef - y[i]) ** 2)
    return float(np.mean(errors))


def grid_analysis(grid):
    cells = grid[grid.dataset.isin(GRID_DATASETS)]
    counts = cells.groupby(['model', 'mode', 'bits', 'gs']).size()
    assert (counts == len(GRID_DATASETS)).all() and len(counts) == 9 * 32
    groups = cells.groupby(['model', 'mode', 'bits', 'gs', 'group']).retention.mean().unstack('group')
    groups['all'] = cells.groupby(['model', 'mode', 'bits', 'gs']).retention.mean()
    groups['gap'] = groups.other - groups.sc
    groups.reset_index().to_csv(RESULTS / 'grid_groups.csv', index=False)
    by_dataset = cells[cells.gs == 128].groupby(['dataset', 'bits']).retention.mean().unstack('bits')
    by_dataset.to_csv(RESULTS / 'grid_datasets_g128.csv')

    put('cells', len(grid), 0)
    put('contexts', len(grid.drop_duplicates(['dataset', 'model', 'mode'])), 0)
    put('gridDatasets', len(GRID_DATASETS), 0)
    mean = groups.groupby(['bits', 'gs']).mean()
    put('ret_w3g128', mean.loc[(3.0, 128), 'all'], 2)
    put('retSc_w3g128', mean.loc[(3.0, 128), 'sc'], 2)
    put('retOther_w3g128', mean.loc[(3.0, 128), 'other'], 2)
    put('ret_w2g128', mean.loc[(2.0, 128), 'all'], 2)
    for bits, gs in [(3.0, 128), (2.0, 8), (2.0, 128)]:
        put(f'gap_w{int(bits)}g{gs}', mean.loc[(bits, gs), 'gap'], 2)
    put('ret_w8_min', mean.loc[8.0, 'all'].min(), 3)
    put('ret_w1_max', mean.loc[1.0, 'all'].max(), 3)
    positive = (groups.gap > 0).groupby(['bits']).sum()
    put('gapPos_w3', positive.loc[3.0], 0)
    put('gapPos_w4', positive.loc[4.0], 0)
    put('nSettingGs', len(SETTINGS) * len(GROUP_SIZES), 0)
    beit_ft = groups.loc[('beit', 'ft', 1.0)].gap
    assert (beit_ft < 0).all() and len(beit_ft) == len(GROUP_SIZES)
    put('maxSc_w3g128', by_dataset.loc[SOY_COTTON, 3.0].max(), 2)
    put('minOther_w3g128', by_dataset.drop(index=SOY_COTTON).loc[:, 3.0].min(), 2)
    assert by_dataset.loc[SOY_COTTON, 3.0].max() < by_dataset.drop(index=SOY_COTTON).loc[:, 3.0].min()

    base = grid.drop_duplicates(['dataset', 'model', 'mode'])
    acc = base.groupby(['group', 'dataset']).fp32_top1.mean().groupby('group').mean()
    put('fpAccSc', acc.loc['sc'], 1)
    put('fpAccOther', acc.loc['other'], 1)
    put('lowBaseline', (base.fp32_top1 < 15).sum(), 0)
    assert (base[base.fp32_top1 < 15].group == 'sc').all()


def difficulty_analysis(grid, master):
    primary = grid[grid.dataset.isin(GRID_DATASETS)].copy()
    primary = primary[[(b, g) in PRIMARY for b, g in zip(primary.bits, primary.gs)]]
    primary['loss'] = 1 - primary.retention
    primary['rank'] = primary.groupby(['model', 'mode', 'bits', 'gs']).loss.rank(pct=True)
    # Rounding keeps equal mean ranks tied despite summation order.
    score = primary.groupby(['model', 'mode', 'dataset'])['rank'].mean().round(10).unstack(['model', 'mode'])
    score = score[pd.MultiIndex.from_tuples(SETTINGS)]
    score.to_csv(RESULTS / 'difficulty.csv')
    agreement = score.corr(method='spearman')
    agreement.to_csv(RESULTS / 'difficulty_agreement.csv')

    rng = np.random.default_rng(SEED)
    rows = []
    properties = ['fp_normalized_margin_q10', 'fp_accuracy', 'num_classes', 'fp_separability',
                  'fp_h_score_final', 'fp_norm_p99_p50', 'final_cka']
    for model, mode in SETTINGS:
        y = score[(model, mode)]
        x = master[(master.model == model) & (master['mode'] == mode)].set_index('dataset').reindex(y.index)
        assert x.fp_normalized_margin_q10.notna().all()
        losses = primary[(primary.model == model) & (primary['mode'] == mode)].pivot_table(
            index='dataset', columns=['bits', 'gs'], values='loss').reindex(y.index)
        margin = x.fp_normalized_margin_q10
        controls = np.c_[np.ones(len(y)), x.fp_accuracy, np.log(x.num_classes)]
        row = {'model': model, 'mode': mode, 'w': kendall_w(losses),
               'rho_margin': stats.spearmanr(margin, y).statistic,
               'p_margin': permutation_p(margin.to_numpy(), y.to_numpy(), rng),
               'neg_conditions': int(sum(stats.spearmanr(margin, losses[c]).statistic < 0 for c in losses)),
               'lodo_max_rho': max(stats.spearmanr(margin.drop(d), y.drop(d)).statistic for d in y.index),
               'err_controls': lodo_error(controls, y.to_numpy()),
               'err_margin': lodo_error(np.c_[controls, margin], y.to_numpy())}
        for name in properties:
            row[f'rho_{name}'] = stats.spearmanr(x[name], y).statistic
        rows.append(row)
    gates = pd.DataFrame(rows)
    gates['passes'] = ((gates.w >= 0.6) & (gates.rho_margin <= -0.5) & (gates.p_margin < 0.05)
                       & (gates.neg_conditions >= 3) & (gates.lodo_max_rho < 0))
    gates.to_csv(RESULTS / 'difficulty_gates.csv', index=False)

    upper = agreement.to_numpy()[np.triu_indices(len(SETTINGS), 1)]
    pairs = agreement.where(np.triu(np.ones(agreement.shape, bool), 1)).stack([0, 1], future_stack=True).dropna()
    least = pairs.idxmin()
    assert {least[:2], least[2:]} == {('beit', 'cal'), ('beit', 'ft')}
    hardest = (score.rank(ascending=False, method='min') <= 5).loc[SOY_COTTON].sum()
    put('scHardestMin', hardest.min(), 0)
    fine = score.drop(index=SOY_COTTON)
    separation = [np.mean([(u > f) + 0.5 * (u == f) for u in score.loc[SOY_COTTON, s] for f in fine[s]])
                  for s in score.columns]
    put('sepMin', min(separation), 2)
    put('sepOne', sum(x == 1 for x in separation), 0)
    within = fine.corr(method='spearman').to_numpy()[np.triu_indices(len(SETTINGS), 1)]
    put('agreeOtherMedian', np.median(within), 2)
    put('scHardestAll', (hardest == len(SOY_COTTON)).sum(), 0)
    put('wMin', gates.w.min(), 2)
    put('wMax', gates.w.max(), 2)
    put('agreeMedian', np.median(upper), 2)
    put('agreeMin', upper.min(), 2)
    put('agreeMax', upper.max(), 2)
    put('gatePass', gates.passes.sum(), 0)
    put('gatePPass', (gates.p_margin < 0.05).sum(), 0)
    put('rhoMarginMedian', gates.rho_margin.median(), 2)
    put('rhoAccMedian', gates.rho_fp_accuracy.median(), 2)
    put('marginHelps', (gates.err_margin < gates.err_controls).sum(), 0)
    for name, key in [('fp_separability', 'Sep'), ('fp_h_score_final', 'Hscore'),
                      ('fp_norm_p99_p50', 'Norm'), ('final_cka', 'Cka')]:
        put(f'rho{key}Median', gates[f'rho_{name}'].median(), 2)


def w3_analysis(master):
    master = master.copy()
    for column in ['fp_accuracy', 'accuracy', 'swap_rate', 'damage_rate', 'rescue_rate']:
        master[column] = 100 * master[column]
    bias = pd.read_csv(DATA / 'bias_results.csv')
    bias['model'] = bias.model_name.map(MODEL_KEYS)
    bias = bias.rename(columns={'dataset_name': 'dataset', 'checkpoint_mode': 'mode'})
    merged = master.merge(bias[['dataset', 'model', 'mode', 'gain_pp', 'shuffle_mean_gain_pp']],
                          on=['dataset', 'model', 'mode'], validate='1:1')
    comparisons = pd.read_csv(DATA / 'competitor_comparisons.csv')
    comparisons = comparisons[(comparisons.scope == 'lodo') & (comparisons.outcome == 'damage')].copy()
    comparisons['model'] = comparisons.model_name.map(MODEL_KEYS)
    competitor = comparisons.groupby(['model', 'checkpoint_mode']).delta_brier.mean()
    summary = merged.groupby(['model', 'mode']).agg(
        fp_accuracy=('fp_accuracy', 'mean'), accuracy=('accuracy', 'mean'),
        relative_loss=('relative_loss', 'median'), final_cka=('final_cka', 'median'),
        swap_rate=('swap_rate', 'mean'), damage_rate=('damage_rate', 'mean'), rescue_rate=('rescue_rate', 'mean'),
        bias_gain=('gain_pp', 'mean'), shuffle_gain=('shuffle_mean_gain_pp', 'mean'))
    summary['relative_loss'] *= 100
    summary['delta_brier'] = [competitor.loc[s] for s in summary.index]
    summary = summary.reindex(pd.MultiIndex.from_tuples(SETTINGS, names=['model', 'mode']))
    summary.reset_index().to_csv(RESULTS / 'w3_summary.csv', index=False)

    put('testRows', master[(master.model == 'vit') & (master['mode'] == 'ft')].n.sum(), 0)
    put('beitFzCka', summary.loc[('beit', 'fz'), 'final_cka'], 2)
    put('beitFzFp', summary.loc[('beit', 'fz'), 'fp_accuracy'], 1)
    put('beitFzQ', summary.loc[('beit', 'fz'), 'accuracy'], 1)
    put('damageOverRescueMin', (summary.damage_rate / summary.rescue_rate).min(), 1)
    put('biasGainMin', summary.bias_gain.min(), 1)
    put('biasGainMax', summary.bias_gain.max(), 1)
    assert (summary.shuffle_gain < 0).all()
    put('shuffleLossMin', -summary.shuffle_gain.max(), 1)
    put('shuffleLossMax', -summary.shuffle_gain.min(), 1)
    put('competitorBetter', (summary.delta_brier < 0).sum(), 0)
    scores = pd.read_csv(DATA / 'competitor_scores.csv')
    put('aurocUndefined', (scores.auroc_status != 'ok').sum(), 0)
    put('aurocRows', len(scores), 0)

    group = merged.groupby(['model', 'mode', 'group'])[['swap_rate', 'damage_rate']].mean().unstack('group')
    put('swapHigherSc', (group[('swap_rate', 'sc')] > group[('swap_rate', 'other')]).sum(), 0)
    columns = ['fp_accuracy', 'accuracy', 'swap_rate', 'damage_rate', 'rescue_rate']
    groups = merged.groupby(['model', 'mode', 'group'])[columns].mean().groupby('group').median()
    groups.loc[['sc', 'other']].to_csv(RESULTS / 'w3_groups.csv')
    sc = merged[merged.group == 'sc'].groupby(['model', 'mode'])[columns[2:]].mean()
    put('wrongSwapScMin', 100 * (1 - (sc.damage_rate + sc.rescue_rate) / sc.swap_rate).min(), 0)
    put('w3FpSc', groups.loc['sc', 'fp_accuracy'], 1)
    for name, key in [('sc', 'Sc'), ('other', 'Other')]:
        put(f'w3Swap{key}', groups.loc[name, 'swap_rate'], 1)


def winner_analysis(master):
    """Share of swaps whose quantized winner was ranked below second by the baseline."""
    ranks = pd.read_csv(DATA / 'winner_ranks.csv')
    ranks['model'] = ranks.model_name.map(MODEL_KEYS)
    ranks = ranks.rename(columns={'dataset_name': 'dataset', 'checkpoint_mode': 'mode'})
    ranks = ranks.merge(master[['dataset', 'model', 'mode', 'group', 'num_classes', 'fp_normalized_margin_q10']],
                        on=['dataset', 'model', 'mode'], validate='1:1')
    share = ranks.swap_definitely_beyond_top2_pct
    by_dataset = ranks.groupby('dataset').swap_definitely_beyond_top2_pct.median()
    put('beyondContexts', len(ranks), 0)
    put('beyondMedian', share.median(), 0)
    put('beyondSc', ranks[ranks.group == 'sc'].swap_definitely_beyond_top2_pct.median(), 0)
    put('beyondOther', ranks[ranks.group == 'other'].swap_definitely_beyond_top2_pct.median(), 0)
    put('beyondCotton', by_dataset['cotton'], 0)
    put('beyondCub', by_dataset['cub'], 0)
    put('cottonClasses', ranks[ranks.dataset == 'cotton'].num_classes.iloc[0], 0)
    put('beyondRhoClasses', stats.spearmanr(ranks.num_classes, share).statistic, 2)
    put('beyondRhoMargin', stats.spearmanr(ranks.fp_normalized_margin_q10, share).statistic, 2)
    assert by_dataset[SOY_COTTON].min() > by_dataset.drop(index=SOY_COTTON).max()


def margin_analysis():
    gaps = pd.read_csv(DATA / 'margin_gaps.csv')
    gaps['model'] = gaps.model_name.map(MODEL_KEYS)
    support = pd.read_csv(DATA / 'matching_support.csv')
    support['model'] = support.model.map(MODEL_KEYS)
    main = gaps[gaps.bin_width == 0.05].set_index(['model', 'checkpoint_mode'])
    worst = support[support.bin_width == 0.05].groupby(['model', 'mode']).agg(
        overlap=('overlap', 'first'), ess_ratio=('matched_to_raw_ess', 'min'))
    table = pd.DataFrame({'raw': main.raw, 'matched': main.matched,
                          'reduction': 100 * main.absolute_gap_reduction_fraction})
    table = table.join(worst.rename_axis(['model', 'checkpoint_mode']))
    table = table.reindex(pd.MultiIndex.from_tuples(SETTINGS, names=['model', 'mode']))
    table.reset_index().to_csv(RESULTS / 'margin_table.csv', index=False)
    assert len(table) == 9 and table.notna().all().all()

    put('reductionMin', table.reduction.min(), 1)
    put('reductionMax', table.reduction.max(), 1)
    put('reductionMedian', table.reduction.median(), 0)
    put('matchedPositive', (table.matched > 0).sum(), 0)
    put('overlapMin', table.overlap.min(), 2)
    put('overlapMax', table.overlap.max(), 2)
    put('essRatioMin', 100 * support.matched_to_raw_ess.min(), 1)
    put('weightMax', support.max_weight_multiplier.max(), 2)
    sensitivity = gaps.groupby('bin_width').absolute_gap_reduction_fraction.agg(['min', 'max'])
    sensitivity.to_csv(RESULTS / 'margin_sensitivity.csv')
    put('reductionMinAll', 100 * gaps.absolute_gap_reduction_fraction.min(), 1)
    put('reductionMaxAll', 100 * gaps.absolute_gap_reduction_fraction.max(), 1)


def layer_analysis(master):
    layers = pd.read_csv(DATA / 'layer_metrics.csv')
    layers = layers[layers.condition == 'w3_g128'].copy()
    layers['model'] = layers.model.map(MODEL_KEYS)
    layers['group'] = layers.dataset.map(group_of)
    layers['depth'] = layers.groupby(['dataset', 'model', 'mode']).cumcount()
    layers['depth'] /= layers.groupby(['dataset', 'model', 'mode']).depth.transform('max')
    curves = layers.groupby(['model', 'mode', 'group', 'depth']).cka_vs_fp.median().reset_index()
    curves.to_csv(RESULTS / 'layer_cka.csv', index=False)
    final = layers[layers.layer == 'final'].merge(master[['dataset', 'model', 'mode', 'final_cka']],
                                                  on=['dataset', 'model', 'mode'], validate='1:1')
    assert np.allclose(final.cka_vs_fp, final.final_cka)
    mid = layers[layers.depth.between(0.45, 0.55)].groupby('group').cka_vs_fp.median()
    put('midCkaSc', mid.loc['sc'], 2)
    put('midCkaOther', mid.loc['other'], 2)


def consistency(grid, master):
    cell = grid[(grid.bits == 3.0) & (grid.gs == 128)]
    pairs = cell.merge(master, on=['dataset', 'model', 'mode'], validate='1:1')
    put('consistContexts', len(pairs), 0)
    put('consistFp', (pairs.fp32_top1 - 100 * pairs.fp_accuracy).abs().max(), 2)
    put('consistQ', (pairs.hqq_top1 - 100 * pairs.accuracy).abs().max(), 2)


def datasets_table(master):
    coverage = pd.read_csv(DATA / 'margin_coverage.csv')
    sizes = coverage.groupby('dataset_name').agg(n=('n_samples', 'first'), classes=('num_classes', 'first'),
                                                  n_unique=('n_samples', 'nunique'))
    assert (sizes.n_unique == 1).all()
    accuracy = (100 * master.groupby('dataset').fp_accuracy.mean()).rename('fp_accuracy')
    table = sizes.join(accuracy).reindex(DATASETS)
    table.to_csv(RESULTS / 'datasets.csv')
    put('nDatasets', len(DATASETS), 0)
    put('nSc', len(SOY_COTTON), 0)
    put('nOther', len(DATASETS) - len(SOY_COTTON), 0)


def tsne_example():
    coordinates = pd.read_csv(DATA / 'tsne_cub_vit_b16_ft.csv')
    coverage = pd.read_csv(DATA / 'margin_coverage.csv')
    put('tsneClasses', coordinates.target.nunique(), 0)
    put('cubClasses', coverage[coverage.dataset_name == 'cub'].num_classes.iloc[0], 0)


PAIRS = {'A': ('beit', 'cal', 'cars', 'food'), 'B': ('beit', 'ft', 'dogs', 'cars'), 'C': ('swin', 'cal', 'soygene', 'inat17')}


def accuracy_analysis(grid, master):
    """How far baseline accuracy alone predicts retention at W3/G128."""
    cell = grid[(grid.bits == 3.0) & (grid.gs == 128) & grid.dataset.isin(GRID_DATASETS)].copy()
    cell[['dataset', 'model', 'mode', 'group', 'fp32_top1', 'retention']].to_csv(RESULTS / 'accuracy_retention.csv',
                                                                                 index=False)
    put('pooledN', len(cell), 0)
    X = np.c_[np.ones(len(cell)), cell.fp32_top1]
    residual = cell.retention - X @ np.linalg.lstsq(X, cell.retention, rcond=None)[0]
    put('pooledR2', 100 * (1 - residual.var() / cell.retention.var()), 0)

    cell = cell.merge(master[['dataset', 'model', 'mode', 'fp_normalized_margin_q10']],
                      on=['dataset', 'model', 'mode'], validate='1:1')
    close, differ, largest, ordered, ordered_differ = 0, 0, 0.0, 0, 0
    for _, part in cell.groupby(['model', 'mode']):
        values = part[['fp32_top1', 'retention', 'fp_normalized_margin_q10']].to_numpy()
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if abs(values[i, 0] - values[j, 0]) <= 1.5:
                    close += 1
                    gap = abs(values[i, 1] - values[j, 1])
                    # The pair is ordered when the smaller q10 goes with the lower retention.
                    right = (values[i, 2] != values[j, 2]
                             and (values[i, 2] < values[j, 2]) == (values[i, 1] < values[j, 1]))
                    ordered += right
                    differ += gap > 0.2
                    ordered_differ += right and gap > 0.2
                    largest = max(largest, gap)
    put('pairsClose', close, 0)
    put('pairsDiffer', differ, 0)
    put('pairsMaxDiff', largest, 2)
    put('pairsQ10', ordered, 0)
    put('differQ10', ordered_differ, 0)

    primary = grid[[(b, g) in PRIMARY for b, g in zip(grid.bits, grid.gs)]]
    for tag, (model, mode, low, high) in PAIRS.items():
        part = primary[(primary.model == model) & (primary['mode'] == mode)].pivot_table(
            index='dataset', columns=['bits', 'gs'], values='retention')
        assert (part.loc[low] < part.loc[high]).all()
        row = cell[(cell.model == model) & (cell['mode'] == mode)].set_index('dataset')
        assert abs(row.fp32_top1[low] - row.fp32_top1[high]) <= 1.5
        for side, dataset in [('Low', low), ('High', high)]:
            put(f'pair{tag}{side}Fp', row.fp32_top1[dataset], 1)
            put(f'pair{tag}{side}R', row.retention[dataset], 2)
            if tag in 'AB':
                put(f'pair{tag}{side}Q', row.fp_normalized_margin_q10[dataset], 2)

    score = pd.read_csv(RESULTS / 'difficulty.csv', index_col=0, header=[0, 1])
    partial = []
    for model, mode in SETTINGS:
        y = score[(model, mode)]
        x = master[(master.model == model) & (master['mode'] == mode)].set_index('dataset').reindex(y.index)
        ranks = [stats.rankdata(v) for v in (x.fp_normalized_margin_q10, y, x.fp_accuracy)]
        resid = [r - np.polyval(np.polyfit(ranks[2], r, 1), ranks[2]) for r in ranks[:2]]
        partial.append(np.corrcoef(*resid)[0, 1])
    pd.DataFrame([dict(model=m, mode=k, partial=r) for (m, k), r in zip(SETTINGS, partial)]).to_csv(
        RESULTS / 'partial.csv', index=False)
    put('partialMedian', np.median(partial), 2)
    put('partialNeg', sum(r < 0 for r in partial), 0)


def robustness_analysis(grid):
    """Retention gap at W3/G128 under alternative retention definitions and groupings."""
    classes = pd.read_csv(DATA / 'margin_coverage.csv').groupby('dataset_name').num_classes.first()
    sizes = pd.read_csv(DATA / 'margin_coverage.csv').groupby('dataset_name').n_samples.first()
    cells = grid[grid.dataset.isin(GRID_DATASETS)].copy()
    chance = 100 / cells.dataset.map(classes)
    cells['chance_corrected'] = (cells.hqq_top1 - chance) / (cells.fp32_top1 - chance)
    low = cells.groupby(['dataset', 'model', 'mode']).fp32_top1.transform('first') < 15
    soybean = [d for d in SOY_COTTON if d != 'cotton']

    def gap(frame, column='retention', weights=None):
        frame = frame[(frame.bits == 3.0) & (frame.gs == 128)]
        other = frame[frame.group == 'other'].groupby(['model', 'mode'])[column].mean()
        sc = frame[frame.group == 'sc']
        if weights == 'units':
            sc = sc.assign(unit=np.where(sc.dataset.isin(soybean), 'soybean', sc.dataset))
            sc = sc.groupby(['model', 'mode', 'unit'])[column].mean().groupby(['model', 'mode']).mean()
        else:
            sc = sc.groupby(['model', 'mode'])[column].mean()
        return (other - sc).reindex(pd.MultiIndex.from_tuples(SETTINGS))

    leave_out = {d: gap(cells[cells.dataset != d]) for d in SOY_COTTON}
    worst = min(leave_out, key=lambda d: leave_out[d].mean())
    checks = [('Main analysis', gap(cells)),
              ('Chance-corrected retention', gap(cells, 'chance_corrected')),
              ('Baselines below 15\\% removed', gap(cells[~low])),
              ('Soybean datasets as one unit', gap(cells, weights='units')),
              ('Cotton alone', gap(cells[~cells.dataset.isin(soybean)])),
              ('Soybean datasets alone', gap(cells[cells.dataset != 'cotton'])),
              ('Leave out one dataset (worst)', leave_out[worst])]
    table = pd.DataFrame([{'check': name, 'mean': g.mean(), 'min': g.min(), 'positive': (g > 0).sum()}
                          for name, g in checks])
    table.to_csv(RESULTS / 'robustness.csv', index=False)
    assert (table.positive == len(SETTINGS)).all()
    put('looMinGap', min(g.min() for g in leave_out.values()), 2)
    put('gapChance_w3g128', checks[1][1].mean(), 2)
    put('gapNoLow_w3g128', checks[2][1].mean(), 2)
    one_bit = cells[(cells.bits == 1.0) & (cells.model == 'beit') & (cells['mode'] == 'ft')]
    one_bit = one_bit.groupby(['gs', 'group']).chance_corrected.mean().unstack('group')
    assert ((one_bit.other - one_bit.sc) < 0).all()

    cell = grid[(grid.bits == 3.0) & (grid.gs == 128)].copy()
    n = cell.dataset.map(sizes)
    a, q = cell.fp32_top1 / 100, cell.hqq_top1 / 100
    cell['se'] = cell.retention * np.sqrt((1 - q) / (n * q) + (1 - a) / (n * a))
    se = cell.groupby('dataset').se.median()
    put('seMax', se.max(), 2)
    assert se.idxmax() == 'cotton'
    put('seOtherMax', se.drop(index=SOY_COTTON).max(), 3)

    rng = np.random.default_rng(SEED)
    bounds = []
    primary = cell[cell.dataset.isin(GRID_DATASETS)]
    for model, mode in SETTINGS:
        part = primary[(primary.model == model) & (primary['mode'] == mode)]
        sc = part[part.group == 'sc'].retention.to_numpy()
        other = part[part.group == 'other'].retention.to_numpy()
        draws = [rng.choice(other, len(other)).mean() - rng.choice(sc, len(sc)).mean() for _ in range(PERMUTATIONS)]
        bounds.append(np.percentile(draws, 2.5))
    # BEiTv2 FT's bound sits at zero within Monte Carlo error, so it is printed on its own.
    put('bootPositive', sum(b > 0 for b, s in zip(bounds, SETTINGS) if s != ('beit', 'ft')), 0)
    put('bootLowBeitFt', bounds[SETTINGS.index(('beit', 'ft'))], 2)


def main():
    RESULTS.mkdir(exist_ok=True)
    grid = load_grid()
    master = load_master()
    grid_analysis(grid)
    difficulty_analysis(grid, master)
    w3_analysis(master)
    margin_analysis()
    winner_analysis(master)
    layer_analysis(master)
    consistency(grid, master)
    datasets_table(master)
    tsne_example()
    accuracy_analysis(grid, master)
    robustness_analysis(grid)
    put('nConditions', len(BITS) * len(GROUP_SIZES), 0)
    put('nBits', len(BITS), 0)
    put('nGroupSizes', len(GROUP_SIZES), 0)
    numbers = pd.DataFrame(NUMBERS)
    assert not numbers.key.duplicated().any()
    numbers.to_csv(RESULTS / 'numbers.csv', index=False)


if __name__ == '__main__':
    main()
