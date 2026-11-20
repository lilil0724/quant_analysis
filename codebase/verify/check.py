"""Re-derive every printed number and table cell from data/ without the generator's code; exit 1 on drift."""
import csv
import re
import sys
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper'
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
SC = {'soyageing', 'soygene', 'soyglobal', 'soylocal', 'cotton'}
ALL = ['aircraft', 'cars', 'cub', 'dogs', 'flowers', 'food', 'inat17', 'moe', 'nabirds', 'pets', 'vegfru',
       'soyageing', 'soygene', 'soyglobal', 'soylocal', 'cotton']
COMPLETE = ALL
SETTINGS = [(m, k) for m in ['vit', 'swin', 'beit'] for k in ['ft', 'fz', 'cal']]
MODEL = {'vit_b16': 'vit', 'swin_base_patch4_window7_224_in22k': 'swin',
         'beitv2_base_patch16_224_in22k': 'beit', 'beitv2_base_patch16_224': 'beit'}
PRIMARY = [(3.0, 8), (3.0, 128), (4.0, 8), (4.0, 128)]
FAILURES = []
# Bootstrap bounds are re-drawn with another seed, so they match within Monte Carlo error.
TOLERANCE = {'bootLowBeitFt': 0.015}


def rows(name):
    with open(DATA / name, newline='') as handle:
        return list(csv.DictReader(handle))


def text(value, dp):
    rounded = Decimal(repr(round(float(value), 6))).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP)
    out = str(rounded)
    if dp == 0 and abs(rounded) >= 10000:
        out = f'{int(rounded):,}'.replace(',', '{,}')
    return '$-$' + out[1:] if out.startswith('-') else out


def spearman(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def median(values):
    return float(np.median(list(values)))


def derive():
    v = {}
    grid = rows('grid.csv')
    for r in grid:
        r['m'] = MODEL[r['model_name']]
        r['b'] = float(r['hqq_nbits'])
        r['g'] = int(r['hqq_group_size'])
        r['r'] = float(r['hqq_top1']) / float(r['fp32_top1'])
    v['cells'] = len(grid)
    v['contexts'] = len({(r['dataset_name'], r['m'], r['ckpt_kind']) for r in grid})
    v['gridDatasets'] = len(COMPLETE)
    v['gridOther'] = len([d for d in COMPLETE if d not in SC])

    cell = {(r['dataset_name'], r['m'], r['ckpt_kind'], r['b'], r['g']): r['r'] for r in grid}
    bits = sorted({r['b'] for r in grid})
    sizes = sorted({r['g'] for r in grid})
    gaps = {}
    level = defaultdict(list)
    for m, k in SETTINGS:
        for b in bits:
            for g in sizes:
                sc = mean(cell[(d, m, k, b, g)] for d in COMPLETE if d in SC)
                other = mean(cell[(d, m, k, b, g)] for d in COMPLETE if d not in SC)
                allm = mean(cell[(d, m, k, b, g)] for d in COMPLETE)
                gaps[(m, k, b, g)] = other - sc
                level[(b, g)].append((allm, sc, other, other - sc))
    for b, g in [(3.0, 128), (2.0, 8), (2.0, 128), (4.0, 128), (4.0, 8), (3.0, 8), (5.0, 128)]:
        tag = f'w{int(b)}g{g}'
        a = np.array(level[(b, g)]).mean(axis=0)
        v[f'ret_{tag}'], v[f'retSc_{tag}'], v[f'retOther_{tag}'], v[f'gap_{tag}'] = a
    v['ret_w8_min'] = min(mean(x[0] for x in level[(8.0, g)]) for g in sizes)
    v['ret_w1_max'] = max(mean(x[0] for x in level[(1.0, g)]) for g in sizes)
    v['gapPos_w3'] = sum(gaps[k] > 0 for k in gaps if k[2] == 3.0)
    v['gapPos_w4'] = sum(gaps[k] > 0 for k in gaps if k[2] == 4.0)
    v['nSettingGs'] = sum(1 for k in gaps if k[2] == 3.0)
    by_ds = {d: mean(cell[(d, m, k, 3.0, 128)] for m, k in SETTINGS) for d in COMPLETE}
    v['maxSc_w3g128'] = max(by_ds[d] for d in SC)
    v['minOther_w3g128'] = min(by_ds[d] for d in COMPLETE if d not in SC)
    base = {(r['dataset_name'], r['m'], r['ckpt_kind']): float(r['fp32_top1']) for r in grid}
    per_ds = defaultdict(list)
    for (d, m, k), a in base.items():
        per_ds[d].append(a)
    v['fpAccSc'] = mean(mean(per_ds[d]) for d in SC)
    v['fpAccOther'] = mean(mean(per_ds[d]) for d in ALL if d not in SC)
    v['lowBaseline'] = sum(a < 15 for a in base.values())
    if not all(gaps[('beit', 'ft', 1.0, g)] < 0 for g in sizes):
        FAILURES.append('BEiTv2 FT 1-bit gap is not negative at every group size')

    master = {(r['dataset'], MODEL[r['model']], r['mode']): r for r in rows('dataset_master.csv')}
    score = {}
    losses = {}
    for m, k in SETTINGS:
        pct = defaultdict(list)
        for b, g in PRIMARY:
            loss = [1 - cell[(d, m, k, b, g)] for d in COMPLETE]
            losses[(m, k, b, g)] = loss
            for d, p in zip(COMPLETE, rankdata(loss) / len(loss)):
                pct[d].append(p)
        score[(m, k)] = [round(mean(pct[d]), 10) for d in COMPLETE]

    def kendall(m, k):
        ranks = np.array([rankdata(losses[(m, k, b, g)]) for b, g in PRIMARY])
        kk, n = ranks.shape
        s = ((ranks.sum(axis=0) - kk * (n + 1) / 2) ** 2).sum()
        return 12 * s / (kk ** 2 * (n ** 3 - n))

    def lodo(features, y):
        X = np.column_stack(features)
        err = []
        for i in range(len(y)):
            keep = [j for j in range(len(y)) if j != i]
            beta = np.linalg.pinv(X[keep]) @ y[keep]
            err.append(float(X[i] @ beta - y[i]) ** 2)
        return mean(err)

    rng = np.random.default_rng(12345)
    gates = {}
    for m, k in SETTINGS:
        y = np.array(score[(m, k)])
        get = lambda col: np.array([float(master[(d, m, k)][col]) for d in COMPLETE])
        q10, acc, classes = get('fp_normalized_margin_q10'), get('fp_accuracy'), get('num_classes')
        rho = spearman(q10, y)
        draws = np.array([spearman(rng.permutation(q10), y) for _ in range(10000)])
        controls = [np.ones(len(y)), acc, np.log(classes)]
        gate = {'w': kendall(m, k), 'rho': rho, 'p': (1 + (draws <= rho).sum()) / 10001,
                'neg': sum(spearman(q10, losses[(m, k, b, g)]) < 0 for b, g in PRIMARY),
                'lodo': max(spearman(np.delete(q10, i), np.delete(y, i)) for i in range(len(y))),
                'acc': spearman(acc, y), 'e0': lodo(controls, y), 'e1': lodo(controls + [q10], y)}
        for col in ['num_classes', 'fp_separability', 'fp_h_score_final', 'fp_norm_p99_p50', 'final_cka']:
            gate[col] = spearman(get(col), y)
        gate['fp_normalized_margin_q10'] = rho
        gate['fp_accuracy'] = gate['acc']
        gates[(m, k)] = gate
    g = list(gates.values())
    v['wMin'] = min(x['w'] for x in g)
    v['wMax'] = max(x['w'] for x in g)
    pairs = {(a, b): spearman(score[a], score[b]) for i, a in enumerate(SETTINGS) for b in SETTINGS[i + 1:]}
    v['agreeMedian'] = median(pairs.values())
    v['agreeMin'] = min(pairs.values())
    v['agreeMax'] = max(pairs.values())
    if set(min(pairs, key=pairs.get)) != {('beit', 'cal'), ('beit', 'ft')}:
        FAILURES.append('least-agreeing pair is not BEiTv2 CAL and BEiTv2 FT')
    passes = [x['w'] >= 0.6 and x['rho'] <= -0.5 and x['p'] < 0.05 and x['neg'] >= 3 and x['lodo'] < 0 for x in g]
    v['gatePass'] = sum(passes)
    v['gateRhoPass'] = sum(x['rho'] <= -0.5 for x in g)
    v['gatePPass'] = sum(x['p'] < 0.05 for x in g)
    v['rhoMarginMedian'] = median(x['rho'] for x in g)
    v['rhoAccMedian'] = median(x['acc'] for x in g)
    v['rhoMarginBeitFt'] = gates[('beit', 'ft')]['rho']
    v['marginHelps'] = sum(x['e1'] < x['e0'] for x in g)
    for col, key in [('fp_separability', 'Sep'), ('fp_h_score_final', 'Hscore'), ('fp_norm_p99_p50', 'Norm'),
                     ('final_cka', 'Cka'), ('num_classes', 'Classes')]:
        v[f'rho{key}Median'] = median(x[col] for x in g)
    hardest = []
    for m, k in SETTINGS:
        order = rankdata([-s for s in score[(m, k)]], method='min')
        hardest.append(sum(o <= 5 for d, o in zip(COMPLETE, order) if d in SC))
    v['scHardestMin'] = min(hardest)
    v['scHardestAll'] = sum(h == len(SC) for h in hardest)
    sep = []
    for m, k in SETTINGS:
        u = [s for d, s in zip(COMPLETE, score[(m, k)]) if d in SC]
        f = [s for d, s in zip(COMPLETE, score[(m, k)]) if d not in SC]
        sep.append(mean(1.0 if a > b else 0.5 if a == b else 0.0 for a in u for b in f))
    v['sepMin'], v['sepOne'] = min(sep), sum(x == 1 for x in sep)
    fine = lambda key: [s for d, s in zip(COMPLETE, score[key]) if d not in SC]
    v['agreeOtherMedian'] = median(spearman(fine(a), fine(b)) for i, a in enumerate(SETTINGS) for b in SETTINGS[i + 1:])

    # Paired W3/G128 summary.
    bias = {(r['dataset_name'], MODEL[r['model_name']], r['checkpoint_mode']): r for r in rows('bias_results.csv')}
    brier = defaultdict(list)
    for r in rows('competitor_comparisons.csv'):
        if r['scope'] == 'lodo' and r['outcome'] == 'damage':
            brier[(MODEL[r['model_name']], r['checkpoint_mode'])].append(float(r['delta_brier']))
    summary = {}
    for m, k in SETTINGS:
        ctx = [master[(d, m, k)] for d in ALL]
        f = lambda col: [float(r[col]) for r in ctx]
        summary[(m, k)] = {
            'fp': 100 * mean(f('fp_accuracy')), 'q': 100 * mean(f('accuracy')),
            'loss': 100 * median(f('relative_loss')), 'cka': median(f('final_cka')),
            'swap': 100 * mean(f('swap_rate')), 'damage': 100 * mean(f('damage_rate')),
            'rescue': 100 * mean(f('rescue_rate')),
            'gain': mean(float(bias[(d, m, k)]['gain_pp']) for d in ALL),
            'shuffle': mean(float(bias[(d, m, k)]['shuffle_mean_gain_pp']) for d in ALL),
            'brier': mean(brier[(m, k)])}
    s = list(summary.values())
    v['testRows'] = sum(int(master[(d, 'vit', 'ft')]['n']) for d in ALL)
    v['beitFzCka'] = summary[('beit', 'fz')]['cka']
    v['beitFzFp'] = summary[('beit', 'fz')]['fp']
    v['beitFzQ'] = summary[('beit', 'fz')]['q']
    v['damageOverRescueMin'] = min(x['damage'] / x['rescue'] for x in s)
    v['biasGainMin'] = min(x['gain'] for x in s)
    v['biasGainMax'] = max(x['gain'] for x in s)
    v['shuffleLossMin'] = -max(x['shuffle'] for x in s)
    v['shuffleLossMax'] = -min(x['shuffle'] for x in s)
    v['competitorBetter'] = sum(x['brier'] < 0 for x in s)
    scores = rows('competitor_scores.csv')
    v['aurocUndefined'] = sum(r['auroc_status'] != 'ok' for r in scores)
    v['aurocRows'] = len(scores)
    higher = 0
    for m, k in SETTINGS:
        sw = lambda grp: mean(float(master[(d, m, k)]['swap_rate']) for d in ALL if (d in SC) == grp)
        higher += sw(True) > sw(False)
    v['swapHigherSc'] = higher
    w3_groups = {}
    for grp, key in [(True, 'Sc'), (False, 'Other')]:
        w3_groups[key] = [100 * median(mean(float(master[(d, m, k)][col]) for d in ALL if (d in SC) == grp)
                                       for m, k in SETTINGS)
                          for col in ['fp_accuracy', 'accuracy', 'swap_rate', 'damage_rate', 'rescue_rate']]
        v[f'w3Swap{key}'] = w3_groups[key][2]
    v['w3FpSc'] = w3_groups['Sc'][0]
    v['_w3_groups'] = w3_groups
    share = []
    for m, k in SETTINGS:
        rate = lambda col: mean(float(master[(d, m, k)][col]) for d in SC)
        share.append(100 * (1 - (rate('damage_rate') + rate('rescue_rate')) / rate('swap_rate')))
    v['wrongSwapScMin'] = min(share)

    # Swaps whose new winner was below the baseline's second class.
    wr = rows('winner_ranks.csv')
    share = [float(r['swap_definitely_beyond_top2_pct']) for r in wr]
    key = lambda r: (r['dataset_name'], MODEL[r['model_name']], r['checkpoint_mode'])
    v['beyondContexts'] = len(wr)
    v['beyondMedian'] = median(share)
    v['beyondSc'] = median(float(r['swap_definitely_beyond_top2_pct']) for r in wr if r['dataset_name'] in SC)
    v['beyondOther'] = median(float(r['swap_definitely_beyond_top2_pct']) for r in wr if r['dataset_name'] not in SC)
    per = defaultdict(list)
    for r in wr:
        per[r['dataset_name']].append(float(r['swap_definitely_beyond_top2_pct']))
    v['beyondCotton'], v['beyondCub'] = median(per['cotton']), median(per['cub'])
    v['cottonClasses'] = int(master[('cotton', 'vit', 'ft')]['num_classes'])
    v['beyondRhoClasses'] = spearman([float(master[key(r)]['num_classes']) for r in wr], share)
    v['beyondRhoMargin'] = spearman([float(master[key(r)]['fp_normalized_margin_q10']) for r in wr], share)

    # Margin matching.
    gaps_rows = rows('margin_gaps.csv')
    main_gaps = {(MODEL[r['model_name']], r['checkpoint_mode']): r for r in gaps_rows if float(r['bin_width']) == 0.05}
    support = rows('matching_support.csv')
    red = [100 * float(main_gaps[x]['absolute_gap_reduction_fraction']) for x in SETTINGS]
    v['reductionMin'], v['reductionMax'] = min(red), max(red)
    v['reductionMedian'] = median(red)
    v['matchedPositive'] = sum(float(main_gaps[x]['matched']) > 0 for x in SETTINGS)
    overlap = [float(r['overlap']) for r in support if float(r['bin_width']) == 0.05]
    v['overlapMin'], v['overlapMax'] = min(overlap), max(overlap)
    v['essRatioMin'] = 100 * min(float(r['matched_to_raw_ess']) for r in support)
    v['weightMax'] = max(float(r['max_weight_multiplier']) for r in support)
    allred = [100 * float(r['absolute_gap_reduction_fraction']) for r in gaps_rows]
    v['reductionMinAll'], v['reductionMaxAll'] = min(allred), max(allred)

    # Layer CKA at half depth.
    layers = defaultdict(list)
    for r in rows('layer_metrics.csv'):
        if r['condition'] == 'w3_g128':
            layers[(r['dataset'], r['model'], r['mode'])].append(float(r['cka_vs_fp']))
    mid = defaultdict(list)
    for (d, m, k), values in layers.items():
        for i, c in enumerate(values):
            if 0.45 <= i / (len(values) - 1) <= 0.55:
                mid[d in SC].append(c)
    v['midCkaSc'], v['midCkaOther'] = median(mid[True]), median(mid[False])

    # Consistency of the two runs.
    fp_diff, q_diff = [], []
    for r in grid:
        if r['b'] == 3.0 and r['g'] == 128:
            p = master[(r['dataset_name'], r['m'], r['ckpt_kind'])]
            fp_diff.append(abs(float(r['fp32_top1']) - 100 * float(p['fp_accuracy'])))
            q_diff.append(abs(float(r['hqq_top1']) - 100 * float(p['accuracy'])))
    v['consistContexts'], v['consistFp'], v['consistQ'] = len(fp_diff), max(fp_diff), max(q_diff)

    coverage = rows('margin_coverage.csv')
    v['nDatasets'] = len({r['dataset_name'] for r in coverage})
    v['nSc'] = len(SC)
    v['nOther'] = v['nDatasets'] - len(SC)
    v['nBits'], v['nGroupSizes'] = len(bits), len(sizes)
    v['nConditions'] = len(bits) * len(sizes)
    v['tsneClasses'] = len({r['target'] for r in rows('tsne_cub_vit_b16_ft.csv')})
    v['cubClasses'] = int(next(r['num_classes'] for r in coverage if r['dataset_name'] == 'cub'))
    sizes_ds = {r['dataset_name']: (int(r['n_samples']), int(r['num_classes'])) for r in coverage}
    fp_ds = {d: 100 * mean(float(master[(d, m, k)]['fp_accuracy']) for m, k in SETTINGS) for d in ALL}

    # Baseline accuracy against retention at W3/G128.
    pts = [(d, m, k, base[(d, m, k)], cell[(d, m, k, 3.0, 128)]) for d in COMPLETE for m, k in SETTINGS]
    acc_all = np.array([p[3] for p in pts])
    ret_all = np.array([p[4] for p in pts])
    v['pooledN'] = len(pts)
    v['pooledRho'] = spearman(acc_all, ret_all)
    v['pooledR2'] = 100 * float(np.corrcoef(acc_all, ret_all)[0, 1] ** 2)
    close = differ = ordered = ordered_differ = 0
    largest = 0.0
    q10 = lambda d, m, k: float(master[(d, m, k)]['fp_normalized_margin_q10'])
    for m, k in SETTINGS:
        for i, a in enumerate(COMPLETE):
            for b in COMPLETE[i + 1:]:
                if abs(base[(a, m, k)] - base[(b, m, k)]) <= 1.5:
                    close += 1
                    ra, rb = cell[(a, m, k, 3.0, 128)], cell[(b, m, k, 3.0, 128)]
                    qa, qb = q10(a, m, k), q10(b, m, k)
                    agrees = qa != qb and (qa - qb) * (ra - rb) > 0
                    differ += abs(ra - rb) > 0.2
                    ordered += agrees
                    ordered_differ += agrees and abs(ra - rb) > 0.2
                    largest = max(largest, abs(ra - rb))
    v['pairsClose'], v['pairsDiffer'], v['pairsMaxDiff'] = close, differ, largest
    v['pairsQ10'], v['differQ10'] = ordered, ordered_differ
    v['pairALowQ'], v['pairAHighQ'] = q10('cars', 'beit', 'cal'), q10('food', 'beit', 'cal')
    v['pairBLowQ'], v['pairBHighQ'] = q10('dogs', 'beit', 'ft'), q10('cars', 'beit', 'ft')
    for tag, (m, k, low, high) in {'A': ('beit', 'cal', 'cars', 'food'), 'B': ('beit', 'ft', 'dogs', 'cars'),
                                   'C': ('swin', 'cal', 'soygene', 'inat17')}.items():
        if not all(cell[(low, m, k, b, g)] < cell[(high, m, k, b, g)] for b, g in PRIMARY):
            FAILURES.append(f'pair {tag}: order differs across primary conditions')
        for side, d in [('Low', low), ('High', high)]:
            v[f'pair{tag}{side}Fp'] = base[(d, m, k)]
            v[f'pair{tag}{side}R'] = cell[(d, m, k, 3.0, 128)]
    partial = []
    for m, k in SETTINGS:
        y = np.array(score[(m, k)])
        get = lambda col: np.array([float(master[(d, m, k)][col]) for d in COMPLETE])
        z = np.column_stack([np.ones(len(y)), rankdata(get('fp_accuracy'))])
        res = [r - z @ np.linalg.lstsq(z, r, rcond=None)[0] for r in (rankdata(get('fp_normalized_margin_q10')),
                                                                     rankdata(y))]
        partial.append(float(np.dot(res[0], res[1]) / np.sqrt(np.dot(res[0], res[0]) * np.dot(res[1], res[1]))))
        gates[(m, k)]['partial'] = partial[-1]
    v['partialMedian'] = median(partial)
    v['partialNeg'] = sum(r < 0 for r in partial)

    # Robustness of the W3/G128 gap.
    classes = {r['dataset_name']: int(r['num_classes']) for r in coverage}
    n_test = {r['dataset_name']: int(r['n_samples']) for r in coverage}
    top = {(r['dataset_name'], r['m'], r['ckpt_kind'], r['b'], r['g']): (float(r['fp32_top1']), float(r['hqq_top1']))
           for r in grid}

    def gap_of(datasets, value, units=False):
        out = []
        for m, k in SETTINGS:
            other = mean(value(d, m, k) for d in datasets if d not in SC)
            if units:
                soy = mean(value(d, m, k) for d in datasets if d in SC and d != 'cotton')
                sc = mean([soy, value('cotton', m, k)])
            else:
                sc = mean(value(d, m, k) for d in datasets if d in SC)
            out.append(other - sc)
        return out

    plain = lambda d, m, k: cell[(d, m, k, 3.0, 128)]

    def corrected(d, m, k):
        a, q = top[(d, m, k, 3.0, 128)]
        c = 100 / classes[d]
        return (q - c) / (a - c)

    def no_low(datasets):
        out = []
        for m, k in SETTINGS:
            vals = lambda grp: [plain(d, m, k) for d in datasets if (d in SC) == grp and base[(d, m, k)] >= 15]
            out.append(mean(vals(False)) - mean(vals(True)))
        return out

    leave = {d: gap_of([x for x in COMPLETE if x != d], plain) for d in SC}
    worst = min(leave, key=lambda d: mean(leave[d]))
    robust = [('Main analysis', gap_of(COMPLETE, plain)),
              ('Chance-corrected retention', gap_of(COMPLETE, corrected)),
              ('Baselines below 15\\% removed', no_low(COMPLETE)),
              ('Soybean datasets as one unit', gap_of(COMPLETE, plain, units=True)),
              ('Cotton alone', gap_of([d for d in COMPLETE if d not in SC or d == 'cotton'], plain)),
              ('Soybean datasets alone', gap_of([d for d in COMPLETE if d != 'cotton'], plain)),
              ('Leave out one dataset (worst)', leave[worst])]
    v['looMinGap'] = min(min(g) for g in leave.values())
    v['gapChance_w3g128'] = mean(robust[1][1])
    v['gapNoLow_w3g128'] = mean(robust[2][1])
    v['gapCottonMin'] = min(robust[4][1])
    se = defaultdict(list)
    for (d, m, k, b, g), (a, q) in top.items():
        if b == 3.0 and g == 128:
            a, q = a / 100, q / 100
            se[d].append(q / a * np.sqrt((1 - q) / (n_test[d] * q) + (1 - a) / (n_test[d] * a)))
    se = {d: median(x) for d, x in se.items()}
    v['seMax'] = max(se.values())
    v['seOtherMax'] = max(x for d, x in se.items() if d not in SC)
    rng_boot = np.random.default_rng(777)
    lows = {}
    for m, k in SETTINGS:
        sc = np.array([plain(d, m, k) for d in COMPLETE if d in SC])
        other = np.array([plain(d, m, k) for d in COMPLETE if d not in SC])
        draws = other[rng_boot.integers(0, len(other), (10000, len(other)))].mean(1) - \
            sc[rng_boot.integers(0, len(sc), (10000, len(sc)))].mean(1)
        lows[(m, k)] = float(np.percentile(draws, 2.5))
    v['bootPositive'] = sum(x > 0 for s, x in lows.items() if s != ('beit', 'ft'))
    v['bootLowBeitFt'] = lows[('beit', 'ft')]
    return v, gates, summary, main_gaps, support, sizes_ds, fp_ds, robust


def table_rows(name):
    body = (PAPER / 'tables' / name).read_text().split('\\midrule')[1].split('\\bottomrule')[0]
    return [[c.strip() for c in line.rstrip('\\ ').split('&')] for line in body.strip().splitlines()]


def expect(where, got, want):
    if got != want:
        FAILURES.append(f'{where}: printed {got!r}, derived {want!r}')


def main():
    v, gates, summary, main_gaps, support, sizes_ds, fp_ds, robust = derive()
    numbers = {r['key']: r for r in rows_build('numbers.csv')}
    macros = dict(re.findall(r'V@(\w+)\\endcsname\{(.*)\}', (PAPER / 'tables/macros.tex').read_text()))
    sources = [p for d in ['sec', 'sec_short', 'tables'] for p in (PAPER / d).glob('*.tex')]
    used = set()
    for path in sources:
        used |= set(re.findall(r'\\V\{(\w+)\}', path.read_text()))
    for key in sorted(used - set(v)):
        FAILURES.append(f'{key}: printed but not derivable by the checker')
    for key in sorted(set(numbers) - used):
        FAILURES.append(f'{key}: computed but never printed')
    for key in sorted(used & set(v)):
        if key in TOLERANCE:
            if abs(float(macros[key].replace('$-$', '-')) - v[key]) > TOLERANCE[key]:
                FAILURES.append(f'macro {key}: printed {macros[key]}, derived {v[key]:.4f}')
        else:
            expect(f'macro {key}', macros.get(key), text(v[key], int(numbers[key]['dp'])))
    for i, (name, gaps_) in enumerate(robust):
        expect(f'robustness row {i}', table_rows('robustness.tex')[i],
               [name, text(mean(gaps_), 2), text(min(gaps_), 2), str(sum(x > 0 for x in gaps_))])

    for i, (m, k) in enumerate(SETTINGS):
        x = summary[(m, k)]
        want = [text(x['fp'], 1), text(x['q'], 1), text(x['loss'], 1), text(x['swap'], 1),
                text(x['damage'], 1), text(x['rescue'], 1)]
        expect(f'w3_summary row {i}', table_rows('w3_summary.tex')[i][2:], want)
        want = [text(x['cka'], 2), text(x['gain'], 2), text(x['shuffle'], 2), text(1000 * x['brier'], 2)]
        expect(f'w3_diagnostics row {i}', table_rows('w3_diagnostics.tex')[i][2:], want)
        gte = gates[(m, k)]
        p = '$<$0.01' if gte['p'] < 0.01 else text(gte['p'], 2)
        want = [text(gte['w'], 2), text(gte['rho'], 2), p, text(gte['acc'], 2), text(1000 * gte['e0'], 1),
                text(1000 * gte['e1'], 1)]
        expect(f'gates row {i}', table_rows('gates.tex')[i][2:], want)
        expect(f'gates_short row {i}', table_rows('gates_short.tex')[i][2:], [want[0], want[1], want[3], text(gte['partial'], 2)])
        cols = ['fp_normalized_margin_q10', 'fp_accuracy', 'num_classes', 'fp_separability', 'fp_h_score_final',
                'fp_norm_p99_p50', 'final_cka']
        expect(f'properties row {i}', table_rows('properties.tex')[i][2:], [text(gte[c], 2) for c in cols])
        g = main_gaps[(m, k)]
        sup = [r for r in support if MODEL[r['model']] == m and r['mode'] == k and float(r['bin_width']) == 0.05]
        want = [text(g['raw'], 3), text(g['matched'], 3), text(100 * float(g['absolute_gap_reduction_fraction']), 1),
                text(sup[0]['overlap'], 2), text(100 * min(float(r['matched_to_raw_ess']) for r in sup), 1)]
        expect(f'margin row {i}', table_rows('margin.tex')[i][2:], want)
        expect(f'margin_short row {i}', table_rows('margin_short.tex')[i][2:], want[:3])
    for i, key in enumerate(['Sc', 'Other']):
        expect(f'w3_groups row {i}', table_rows('w3_groups.tex')[i][1:], [text(x, 1) for x in v['_w3_groups'][key]])
    for row in table_rows('datasets.tex'):
        d = row[0].lower()
        n, c = sizes_ds[d]
        comma = lambda x: f'{x:,}'.replace(',', '{,}')
        expect(f'datasets {d}', row[2:], [comma(n), comma(c), text(fp_ds[d], 1)])
        expect(f'datasets {d} group', row[1], 'UFG' if d in SC else 'FG')

    tbd = sum(len(re.findall(r'\\TBD\{', p.read_text())) for p in sources)
    synthetic = sum(p.read_text().count('_placeholder.pdf') for p in sources)
    print(f'checked {len(used)} macros and the cells of 10 tables; '
          f'placeholders not checked: {tbd} TBD notes, synthetic figures: {synthetic}')
    for failure in FAILURES:
        print('FAIL', failure)
    sys.exit(1 if FAILURES else 0)


def rows_build(name):
    with open(RESULTS / name, newline='') as handle:
        return list(csv.DictReader(handle))


if __name__ == '__main__':
    main()
