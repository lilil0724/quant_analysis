"""Test whether overconfident dataset-model pairs are robust per prediction swap."""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if __package__:
    from .common import EXPECTED_CONDITIONS, PREPARED_DIR, write_run_config
else:
    from common import EXPECTED_CONDITIONS, PREPARED_DIR, write_run_config


DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results', 'overconfidence_hypothesis')
PREDICTORS = {
    'oece': 'Overconfidence ECE',
    'signed_confidence_gap': 'Signed confidence gap',
    'ece_uncalibrated': 'ECE',
    'wrong_sample_confidence': 'Wrong-sample confidence',
}
OUTCOMES = {
    'net_damage_per_swap': 'Net damage per swap R',
    'damage_rate': 'Damage rate',
    'accuracy_ratio': 'Accuracy ratio',
    'accuracy_drop_points': 'Accuracy drop (points)',
}
EXPECTED_PLOTS = [
    'oece_vs_net_damage.png',
    'swap_composition_by_confidence.png',
    'margin_calibration_mechanism.png',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test dataset-model overconfidence against HQQ robustness.')
    parser.add_argument('--prepared-dir', default=PREPARED_DIR)
    parser.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--serial', type=int, default=100004)
    parser.add_argument('--exclude-datasets', nargs='*', default=['inat17'])
    parser.add_argument('--expected-datasets', type=int, default=14)
    parser.add_argument('--permutations', type=int, default=10000)
    parser.add_argument('--model-permutations', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=100004)
    parser.add_argument('--dpi', type=int, default=200)
    return parser.parse_args()


def _require_columns(frame, columns, source):
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{source} is missing columns: {", ".join(missing)}')


def _rank(values):
    return pd.Series(np.asarray(values, dtype=float)).rank(method='average').to_numpy()


def spearman(x, y):
    values = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(values) < 3:
        return np.nan
    x_rank = _rank(values['x'])
    y_rank = _rank(values['y'])
    if np.isclose(x_rank.std(), 0) or np.isclose(y_rank.std(), 0):
        return np.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def overconfidence_metrics(confidence, correct, bins=15):
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if confidence.ndim != 1 or correct.shape != confidence.shape or not len(confidence):
        raise ValueError('confidence and correct must be non-empty aligned vectors')
    if not np.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
        raise ValueError('confidence must contain finite probabilities in [0, 1]')
    if bins < 1:
        raise ValueError('bins must be positive')
    ordered = np.argsort(confidence, kind='stable')
    reliability = []
    for index, selected in enumerate(
            np.array_split(ordered, min(bins, len(ordered))), 1):
        if not len(selected):
            continue
        mean_confidence = float(confidence[selected].mean())
        accuracy = float(correct[selected].mean())
        weight = len(selected) / len(correct)
        reliability.append({
            'bin': index,
            'count': int(len(selected)),
            'mean_confidence': mean_confidence,
            'accuracy': accuracy,
            'overconfidence_contribution': weight * max(mean_confidence - accuracy, 0),
            'ece_contribution': weight * abs(mean_confidence - accuracy),
        })
    wrong = confidence[~correct]
    return {
        'oece': float(sum(row['overconfidence_contribution'] for row in reliability)),
        'ece_uncalibrated': float(sum(row['ece_contribution'] for row in reliability)),
        'mean_top1_probability': float(confidence.mean()),
        'signed_confidence_gap': float(confidence.mean() - correct.mean()),
        'wrong_sample_confidence': float(wrong.mean()) if len(wrong) else np.nan,
    }, reliability


def transition_metrics(baseline_correct, hqq_correct, agreement):
    baseline_correct = np.asarray(baseline_correct, dtype=bool)
    hqq_correct = np.asarray(hqq_correct, dtype=bool)
    agreement = np.asarray(agreement, dtype=bool)
    if (baseline_correct.ndim != 1 or hqq_correct.shape != baseline_correct.shape
            or agreement.shape != baseline_correct.shape or not len(agreement)):
        raise ValueError('transition arrays must be non-empty aligned vectors')
    swap = ~agreement
    damage = baseline_correct & ~hqq_correct
    rescue = ~baseline_correct & hqq_correct
    correct_to_correct_swap = baseline_correct & hqq_correct & swap
    wrong_to_wrong_swap = ~baseline_correct & ~hqq_correct & swap
    swap_count = int(swap.sum())
    n_samples = len(swap)
    net_damage_per_swap = (
        float((damage.sum() - rescue.sum()) / swap_count) if swap_count else np.nan)
    return {
        'n_samples': n_samples,
        'swap_count': swap_count,
        'swap_rate': float(swap.mean()),
        'damage_count': int(damage.sum()),
        'damage_fraction': float(damage.mean()),
        'rescue_count': int(rescue.sum()),
        'rescue_fraction': float(rescue.mean()),
        'correct_to_correct_swap_count': int(correct_to_correct_swap.sum()),
        'wrong_to_wrong_swap_count': int(wrong_to_wrong_swap.sum()),
        'baseline_wrong_swap_fraction': (
            float((wrong_to_wrong_swap.sum() + rescue.sum()) / swap_count)
            if swap_count else np.nan),
        'prediction_agreement': float(agreement.mean()),
        'net_damage_per_swap': net_damage_per_swap,
    }


def _equal_count_deciles(values):
    percentile = pd.Series(np.asarray(values, dtype=float)).rank(
        method='average', pct=True).to_numpy()
    return np.clip(np.ceil(percentile * 10).astype(int), 1, 10)


def _bin_rows(record, pair, bin_type, deciles):
    baseline_correct = pair['baseline_correct'].astype(bool)
    hqq_correct = pair['hqq_correct'].astype(bool)
    agreement = pair['agreement'].astype(bool)
    rows = []
    for decile in range(1, 11):
        selected = deciles == decile
        if not selected.any():
            continue
        metrics = transition_metrics(
            baseline_correct[selected], hqq_correct[selected], agreement[selected])
        baseline_correct_count = int(baseline_correct[selected].sum())
        metrics['baseline_correct_count'] = baseline_correct_count
        metrics['damage_rate'] = (
            metrics['damage_count'] / baseline_correct_count
            if baseline_correct_count else np.nan)
        rows.append({
            'sweep_id': record['sweep_id'],
            'dataset_name': record['dataset_name'],
            'condition': record['condition'],
            'hqq_nbits': record['hqq_nbits'],
            'hqq_group_size': record['hqq_group_size'],
            'bin_type': bin_type,
            'decile': decile,
            **metrics,
        })
    return rows


def _load_inputs(prepared_dir, serial, excluded_datasets, expected_datasets):
    transitions = pd.read_csv(os.path.join(prepared_dir, 'transition_summary.csv'))
    conditions = pd.read_csv(os.path.join(prepared_dir, 'condition_summary.csv'))
    _require_columns(transitions, [
        'sweep_id', 'dataset_name', 'model_name', 'serial', 'condition',
        'hqq_nbits', 'hqq_group_size', 'baseline_accuracy', 'hqq_top1',
        'accuracy_ratio', 'accuracy_drop_points', 'damage_rate', 'rescue_rate',
        'prediction_agreement', 'pair_file',
    ], 'transition_summary.csv')
    _require_columns(conditions, [
        'sweep_id', 'dataset_name', 'model_name', 'serial', 'condition',
        'n_samples', 'num_classes', 'accuracy', 'mean_top1_probability',
        'mean_logit_margin', 'mean_normalized_entropy',
    ], 'condition_summary.csv')
    transitions = transitions[transitions['serial'] == serial].copy()
    conditions = conditions[conditions['serial'] == serial].copy()
    excluded = {str(value).lower() for value in excluded_datasets}
    transitions = transitions[
        ~transitions['dataset_name'].str.lower().isin(excluded)].copy()
    conditions = conditions[
        ~conditions['dataset_name'].str.lower().isin(excluded)].copy()
    if transitions.empty or conditions.empty:
        raise ValueError(f'No prepared rows remain for serial {serial}')
    datasets = sorted(transitions['dataset_name'].unique())
    if len(datasets) != expected_datasets:
        raise ValueError(
            f'Expected {expected_datasets} datasets after exclusions, found {len(datasets)}')
    if any(name.lower() == 'inat17' for name in datasets):
        raise ValueError('inat17 must not be included in this analysis')
    expected_hqq = set(EXPECTED_CONDITIONS[1:])
    for dataset, frame in transitions.groupby('dataset_name'):
        observed = set(frame['condition'])
        if observed != expected_hqq or len(frame) != len(expected_hqq):
            raise ValueError(f'{dataset} does not have exactly 12 HQQ conditions')
    baseline = conditions[conditions['condition'] == 'fp_unquantized']
    if len(baseline) != len(datasets) or baseline['dataset_name'].nunique() != len(datasets):
        raise ValueError('Each dataset must have exactly one unquantized baseline')
    if set(baseline['dataset_name']) != set(datasets):
        raise ValueError('Baseline and HQQ dataset sets differ')
    return transitions.sort_values(['dataset_name', 'hqq_nbits', 'hqq_group_size']), baseline


def prepare_hypothesis_tables(
        prepared_dir, serial=100004, excluded_datasets=('inat17',), expected_datasets=14):
    transitions, baseline_summary = _load_inputs(
        prepared_dir, serial, excluded_datasets, expected_datasets)
    baseline_rows = []
    reliability_rows = []
    robustness_rows = []
    mechanism_rows = []
    for record in transitions.to_dict('records'):
        pair_path = os.path.join(prepared_dir, record['pair_file'])
        with np.load(pair_path, allow_pickle=False) as payload:
            pair = {key: payload[key] for key in payload.files}
        metrics = transition_metrics(
            pair['baseline_correct'], pair['hqq_correct'], pair['agreement'])
        expected_drop = 100 * (metrics['damage_fraction'] - metrics['rescue_fraction'])
        if not np.isclose(expected_drop, record['accuracy_drop_points'], atol=1e-10):
            raise ValueError(f'Accuracy decomposition failed for {record["dataset_name"]}')
        if not np.isclose(
                metrics['swap_rate'], 1 - record['prediction_agreement'], atol=1e-12):
            raise ValueError(f'Swap/agreement identity failed for {record["dataset_name"]}')
        robustness_rows.append({**record, **metrics})
        confidence_deciles = _equal_count_deciles(pair['baseline_top1_probability'])
        mechanism_rows.extend(_bin_rows(record, pair, 'raw_confidence', confidence_deciles))
        mechanism_rows.extend(_bin_rows(
            record, pair, 'margin_percentile',
            np.clip(np.ceil(pair['baseline_margin_percentile'] * 10).astype(int), 1, 10)))

    robustness = pd.DataFrame(robustness_rows)
    mechanisms = pd.DataFrame(mechanism_rows)
    first_by_sweep = robustness.drop_duplicates('sweep_id')
    summaries = baseline_summary.set_index('sweep_id')
    for record in first_by_sweep.to_dict('records'):
        pair_path = os.path.join(prepared_dir, record['pair_file'])
        with np.load(pair_path, allow_pickle=False) as pair:
            confidence = pair['baseline_top1_probability']
            correct = pair['baseline_correct']
        confidence_metrics, reliability = overconfidence_metrics(confidence, correct)
        summary = summaries.loc[record['sweep_id']]
        baseline_rows.append({
            'sweep_id': record['sweep_id'],
            'dataset_name': record['dataset_name'],
            'model_name': record['model_name'],
            'serial': serial,
            'n_samples': int(summary['n_samples']),
            'num_classes': int(summary['num_classes']),
            'baseline_accuracy': float(summary['accuracy']),
            'mean_logit_margin': float(summary['mean_logit_margin']),
            'mean_normalized_entropy': float(summary['mean_normalized_entropy']),
            **confidence_metrics,
        })
        for row in reliability:
            reliability_rows.append({
                'sweep_id': record['sweep_id'],
                'dataset_name': record['dataset_name'],
                **row,
            })
    baseline = pd.DataFrame(baseline_rows).sort_values('dataset_name')
    reliability = pd.DataFrame(reliability_rows)
    robustness = robustness.merge(
        baseline, on=['sweep_id', 'dataset_name', 'model_name', 'serial'],
        how='left', validate='many_to_one', suffixes=('', '_calibration'))
    return baseline, reliability, robustness, mechanisms


def condition_correlations(frame, predictor='oece', outcome='net_damage_per_swap'):
    rows = []
    for condition, group in frame.groupby('condition', sort=True):
        valid = group[[predictor, outcome]].dropna()
        rows.append({
            'condition': condition,
            'hqq_nbits': float(group['hqq_nbits'].iloc[0]),
            'hqq_group_size': int(group['hqq_group_size'].iloc[0]),
            'predictor': predictor,
            'outcome': outcome,
            'n_datasets': len(valid),
            'spearman_rho': spearman(valid[predictor], valid[outcome]),
        })
    return pd.DataFrame(rows).sort_values(['hqq_nbits', 'hqq_group_size'])


def block_permutation_test(
        frame, predictor='oece', outcome='net_damage_per_swap', permutations=10000, seed=0):
    if permutations < 1:
        raise ValueError('permutations must be positive')
    datasets = sorted(frame['dataset_name'].unique())
    x = (frame[['dataset_name', predictor]].drop_duplicates('dataset_name')
         .set_index('dataset_name').loc[datasets, predictor].to_numpy(dtype=float))
    by_condition = []
    for condition, group in frame.groupby('condition', sort=True):
        y = group.set_index('dataset_name').reindex(datasets)[outcome].to_numpy(dtype=float)
        by_condition.append((condition, y))
    observed_rows = condition_correlations(frame, predictor, outcome)
    observed = float(observed_rows['spearman_rho'].median())
    if not np.isfinite(observed):
        raise ValueError('Primary correlation is undefined; check predictor and outcome variance')
    rng = np.random.default_rng(seed)
    permuted = np.empty(permutations, dtype=float)
    for index in range(permutations):
        shuffled = x[rng.permutation(len(x))]
        rhos = [spearman(shuffled, y) for _, y in by_condition]
        finite = [rho for rho in rhos if np.isfinite(rho)]
        permuted[index] = float(np.median(finite)) if finite else np.nan
    permuted = permuted[np.isfinite(permuted)]
    if not len(permuted):
        raise ValueError('All permuted correlations are undefined')
    p_value = float((1 + np.count_nonzero(permuted <= observed)) / (permutations + 1))
    return {
        'predictor': predictor,
        'outcome': outcome,
        'alternative': 'negative',
        'n_datasets': len(datasets),
        'n_conditions': len(by_condition),
        'median_spearman_rho': observed,
        'permutations': permutations,
        'permutation_p_value': p_value,
        'seed': seed,
    }, observed_rows


def _design_matrix(frame, model, condition_levels, scaling=None):
    condition = pd.Categorical(frame['condition'], categories=condition_levels)
    dummies = pd.get_dummies(condition, drop_first=True, dtype=float).to_numpy()
    continuous = []
    if model in ('M1', 'M2', 'M3'):
        continuous.extend(['baseline_accuracy', 'log_num_classes'])
    if model in ('M2', 'M3'):
        continuous.append('mean_logit_margin')
    if model == 'M3':
        continuous.append('oece')
    matrix = np.column_stack([np.ones(len(frame)), dummies])
    learned = {} if scaling is None else scaling
    for name in continuous:
        values = frame[name].to_numpy(dtype=float)
        if scaling is None:
            mean = float(values.mean())
            std = float(values.std()) or 1.0
            learned[name] = (mean, std)
        else:
            mean, std = learned[name]
        matrix = np.column_stack([matrix, (values - mean) / std])
    return matrix, learned


def lodo_predictions(frame, model, outcome='net_damage_per_swap'):
    working = frame.dropna(subset=[outcome, 'oece']).copy()
    working['log_num_classes'] = np.log(working['num_classes'])
    if working['dataset_name'].nunique() < 3:
        raise ValueError('LODO analysis requires at least three datasets')
    levels = sorted(working['condition'].unique())
    predictions = []
    for dataset in sorted(working['dataset_name'].unique()):
        train = working[working['dataset_name'] != dataset]
        test = working[working['dataset_name'] == dataset]
        x_train, scaling = _design_matrix(train, model, levels)
        x_test, _ = _design_matrix(test, model, levels, scaling)
        coefficients = np.linalg.lstsq(
            x_train, train[outcome].to_numpy(dtype=float), rcond=None)[0]
        for row_index, predicted in zip(test.index, x_test @ coefficients):
            predictions.append({
                'row_index': row_index,
                'dataset_name': dataset,
                'condition': test.loc[row_index, 'condition'],
                'model': model,
                'observed': float(test.loc[row_index, outcome]),
                'predicted': float(predicted),
            })
    result = pd.DataFrame(predictions)
    result['squared_error'] = np.square(result['observed'] - result['predicted'])
    return result


def nested_model_analysis(frame, permutations=1000, seed=0):
    if permutations < 1:
        raise ValueError('model permutations must be positive')
    prediction_frames = [lodo_predictions(frame, model) for model in ('M0', 'M1', 'M2', 'M3')]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    rows = []
    for model, group in predictions.groupby('model'):
        rows.append({'model': model, 'lodo_rmse': float(np.sqrt(group['squared_error'].mean()))})
    models = pd.DataFrame(rows).sort_values('model')
    rmse = models.set_index('model')['lodo_rmse']
    observed_delta = float(rmse['M2'] - rmse['M3'])
    baseline_rmse = float(rmse['M2'])
    datasets = sorted(frame['dataset_name'].unique())
    oece = (frame[['dataset_name', 'oece']].drop_duplicates('dataset_name')
            .set_index('dataset_name').loc[datasets, 'oece'].to_numpy())
    rng = np.random.default_rng(seed)
    null_deltas = np.empty(permutations, dtype=float)
    for index in range(permutations):
        permuted = frame.copy()
        mapping = dict(zip(datasets, oece[rng.permutation(len(oece))]))
        permuted['oece'] = permuted['dataset_name'].map(mapping)
        m3 = lodo_predictions(permuted, 'M3')
        null_deltas[index] = baseline_rmse - float(np.sqrt(m3['squared_error'].mean()))
    p_value = float(
        (1 + np.count_nonzero(null_deltas >= observed_delta)) / (permutations + 1))
    models['delta_rmse_vs_previous'] = models['lodo_rmse'].shift(1) - models['lodo_rmse']
    nested = {
        'm2_rmse': baseline_rmse,
        'm3_rmse': float(rmse['M3']),
        'm2_minus_m3_rmse': observed_delta,
        'permutations': permutations,
        'permutation_p_value': p_value,
        'seed': seed,
    }
    return nested, models, predictions


def sensitivity_analysis(frame):
    subsets = {
        'confirmatory_all': np.ones(len(frame), dtype=bool),
        'sensitivity_without_1bit_g128': ~(
            np.isclose(frame['hqq_nbits'], 1) & (frame['hqq_group_size'] == 128)),
        'sensitivity_without_8bit': ~np.isclose(frame['hqq_nbits'], 8),
        'posthoc_without_soy': ~frame['dataset_name'].str.startswith('soy'),
    }
    rows = []
    for subset_name, selected in subsets.items():
        subset = frame[selected]
        for predictor in PREDICTORS:
            for outcome in OUTCOMES:
                correlations = condition_correlations(subset, predictor, outcome)
                rows.append({
                    'subset': subset_name,
                    'predictor': predictor,
                    'outcome': outcome,
                    'n_datasets': subset['dataset_name'].nunique(),
                    'n_conditions': subset['condition'].nunique(),
                    'median_spearman_rho': float(correlations['spearman_rho'].median()),
                })
    return pd.DataFrame(rows)


def influence_analysis(frame):
    if frame['dataset_name'].nunique() < 4:
        raise ValueError('Influence analysis requires at least four datasets')
    full = float(condition_correlations(frame)['spearman_rho'].median())
    rows = []
    for dataset in sorted(frame['dataset_name'].unique()):
        reduced = frame[frame['dataset_name'] != dataset]
        rho = float(condition_correlations(reduced)['spearman_rho'].median())
        rows.append({
            'excluded_dataset': dataset,
            'median_spearman_rho': rho,
            'change_from_full': rho - full,
        })
    return pd.DataFrame(rows).sort_values('change_from_full', key=abs, ascending=False)


def _aggregate_mechanisms(mechanisms):
    keys = ['bin_type', 'condition', 'hqq_nbits', 'hqq_group_size', 'decile']
    rows = []
    for key, group in mechanisms.groupby(keys, sort=True):
        totals = group[[
            'n_samples', 'swap_count', 'damage_count', 'rescue_count',
            'correct_to_correct_swap_count', 'wrong_to_wrong_swap_count',
            'baseline_correct_count',
        ]].sum()
        swap_count = int(totals['swap_count'])
        n_samples = int(totals['n_samples'])
        rows.append({
            **dict(zip(keys, key)),
            'n_datasets': group['dataset_name'].nunique(),
            **{name: int(value) for name, value in totals.items()},
            'swap_rate': swap_count / n_samples,
            'net_damage_per_swap': (
                (totals['damage_count'] - totals['rescue_count']) / swap_count
                if swap_count else np.nan),
            'baseline_wrong_swap_fraction': (
                (totals['wrong_to_wrong_swap_count'] + totals['rescue_count']) / swap_count
                if swap_count else np.nan),
        })
    return pd.DataFrame(rows)


def _plot_oece(robustness, path, dpi):
    order = (robustness[['condition', 'hqq_nbits', 'hqq_group_size']]
             .drop_duplicates().sort_values(['hqq_nbits', 'hqq_group_size']))
    figure, axes = plt.subplots(3, 4, figsize=(16, 11), sharex=True, sharey=True)
    for axis, condition in zip(axes.flat, order['condition']):
        panel = robustness[robustness['condition'] == condition]
        sns.regplot(data=panel, x='oece', y='net_damage_per_swap', ci=None, ax=axis,
                    scatter_kws={'s': 25}, line_kws={'linewidth': 1})
        for row in panel.itertuples():
            axis.annotate(row.dataset_name, (row.oece, row.net_damage_per_swap), fontsize=6)
        axis.set_title(condition)
        axis.set_xlabel('OECE')
        axis.set_ylabel('Net damage per swap R')
    figure.suptitle('Dataset–model overconfidence vs HQQ net damage per swap')
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)


def _plot_swap_composition(aggregated, path, dpi):
    frame = aggregated[aggregated['bin_type'] == 'raw_confidence'].groupby('decile').sum(
        numeric_only=True)
    swap = frame['swap_count'].replace(0, np.nan)
    fractions = pd.DataFrame({
        'C→W': frame['damage_count'] / swap,
        'W→C': frame['rescue_count'] / swap,
        'C→C swap': frame['correct_to_correct_swap_count'] / swap,
        'W→W swap': frame['wrong_to_wrong_swap_count'] / swap,
    })
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(frame.index, frame['swap_count'] / frame['n_samples'], marker='o')
    axes[0].set(title='Swap rate by raw-confidence decile', xlabel='Confidence decile',
                ylabel='Swap rate', xticks=range(1, 11))
    fractions.plot(kind='bar', stacked=True, ax=axes[1])
    axes[1].set(title='Composition of swaps', xlabel='Confidence decile',
                ylabel='Fraction of swaps')
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)


def _plot_mechanism(aggregated, robustness, path, dpi):
    margin = aggregated[aggregated['bin_type'] == 'margin_percentile'].groupby('decile').sum(
        numeric_only=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(margin.index, margin['swap_count'] / margin['n_samples'],
                 marker='o', label='All-sample swap rate')
    axes[0].plot(margin.index, margin['damage_count'] / margin['baseline_correct_count'],
                 marker='o', label='Damage rate among baseline-correct')
    axes[0].set(title='Margin mechanism', xlabel='Baseline margin decile', ylabel='Rate',
                xticks=range(1, 11))
    axes[0].legend(fontsize=8)
    dataset_mechanism = robustness.groupby('dataset_name', as_index=False).agg(
        oece=('oece', 'first'),
        baseline_wrong_swap_fraction=('baseline_wrong_swap_fraction', 'mean'),
        net_damage_per_swap=('net_damage_per_swap', 'mean'))
    sns.scatterplot(data=dataset_mechanism, x='oece', y='baseline_wrong_swap_fraction',
                    size='net_damage_per_swap', ax=axes[1], legend=False)
    for row in dataset_mechanism.itertuples():
        axes[1].annotate(row.dataset_name,
                         (row.oece, row.baseline_wrong_swap_fraction), fontsize=7)
    axes[1].set(title='Calibration mechanism', xlabel='Dataset–model OECE',
                ylabel='Baseline-wrong fraction among swaps')
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)


def _decision(primary, nested):
    associated = (primary['median_spearman_rho'] < 0
                  and primary['permutation_p_value'] < .05)
    incremental = (nested['m2_minus_m3_rmse'] > 0
                   and nested['permutation_p_value'] < .05)
    if associated and incremental:
        return 'supports_independent_overconfidence_association'
    if associated:
        return 'raw_association_explained_by_margin_or_covariates'
    return 'hypothesis_not_supported'


def _write_report(
        results_dir, primary, nested, correlations, models, sensitivity,
        influence, decision):
    strongest = correlations.loc[correlations['spearman_rho'].idxmin()]
    weakest = correlations.loc[correlations['spearman_rho'].idxmax()]
    influential = influence.iloc[0]
    robustness_sensitivity = sensitivity[
        (sensitivity['predictor'] == 'oece')
        & (sensitivity['outcome'] == 'net_damage_per_swap')
    ]
    alternate_predictors = sensitivity[
        (sensitivity['subset'] == 'confirmatory_all')
        & (sensitivity['outcome'] == 'net_damage_per_swap')
    ]
    lines = [
        '# Dataset–model overconfidence and HQQ robustness', '',
        'This confirmatory analysis uses raw, uncalibrated probabilities from the matched '
        'AMP-FP16 unquantized baseline. It does not treat overconfidence as an intrinsic '
        'property of a dataset and does not make a causal claim.', '',
        '## Confirmatory result', '',
        f'- Decision: `{decision}`',
        f'- Median condition-wise Spearman rho(OECE, R): '
        f'{primary["median_spearman_rho"]:.6f}',
        f'- One-sided dataset-block permutation p: '
        f'{primary["permutation_p_value"]:.6f} ({primary["permutations"]} permutations)',
        f'- Strongest negative condition: {strongest["condition"]}, '
        f'rho={strongest["spearman_rho"]:.6f}',
        f'- Most positive condition: {weakest["condition"]}, '
        f'rho={weakest["spearman_rho"]:.6f}', '',
        '## Incremental OECE test', '',
        models.to_string(index=False), '',
        f'- M2 minus M3 LODO RMSE: {nested["m2_minus_m3_rmse"]:.6f}',
        f'- Dataset-block permutation p: {nested["permutation_p_value"]:.6f} '
        f'({nested["permutations"]} permutations)', '',
        '## Pre-specified sensitivity analyses', '',
        robustness_sensitivity[
            robustness_sensitivity['subset'] != 'posthoc_without_soy'
        ][['subset', 'n_datasets', 'n_conditions', 'median_spearman_rho']]
        .to_string(index=False), '',
        'Alternate overconfidence predictors on the full cohort:', '',
        alternate_predictors[
            ['predictor', 'n_datasets', 'n_conditions', 'median_spearman_rho']
        ].to_string(index=False), '',
        '## Post-hoc analysis', '',
        robustness_sensitivity[
            robustness_sensitivity['subset'] == 'posthoc_without_soy'
        ][['subset', 'n_datasets', 'n_conditions', 'median_spearman_rho']]
        .to_string(index=False), '',
        '## Anomaly, mechanism, and falsifiable probe', '',
        f'- Anomaly: excluding `{influential["excluded_dataset"]}` changes the median rho '
        f'by {influential["change_from_full"]:.6f}, the largest leave-one-dataset-out shift.',
        '- Mechanism: logit margin controls whether quantization crosses a decision boundary; '
        'calibration controls how often swapped predictions were already wrong, which changes R.',
        '- Probe: train the same ViT-B/16 on one selected dataset with the same seed, optimizer, '
        'schedule, data, and evaluation protocol, changing only label smoothing. If reducing '
        'overconfidence raises R under the same HQQ grid, the proposed mechanism is supported; '
        'if R does not change, independent overconfidence is disfavored.', '',
        '## Interpretation rules', '',
        '- Primary support requires negative median rho and one-sided permutation p < 0.05.',
        '- Independent support additionally requires M3 to improve M2 LODO RMSE with p < 0.05.',
        '- Sensitivity and soy-excluded results are not promoted to confirmatory evidence.',
    ]
    with open(os.path.join(results_dir, 'report.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def analyze_hypothesis(
        prepared_dir, results_dir, serial=100004, excluded_datasets=('inat17',),
        expected_datasets=14, permutations=10000, model_permutations=1000,
        seed=100004, dpi=200):
    if dpi < 1:
        raise ValueError('dpi must be positive')
    os.makedirs(results_dir, exist_ok=True)
    baseline, reliability, robustness, mechanisms = prepare_hypothesis_tables(
        prepared_dir, serial, excluded_datasets, expected_datasets)
    primary, correlations = block_permutation_test(
        robustness, permutations=permutations, seed=seed)
    nested, models, predictions = nested_model_analysis(
        robustness, permutations=model_permutations, seed=seed + 1)
    sensitivity = sensitivity_analysis(robustness)
    influence = influence_analysis(robustness)
    aggregated = _aggregate_mechanisms(mechanisms)
    decision = _decision(primary, nested)

    tables = {
        'dataset_baseline_calibration.csv': baseline,
        'baseline_reliability_bins.csv': reliability,
        'dataset_condition_robustness.csv': robustness,
        'mechanism_bins_by_dataset.csv': mechanisms,
        'mechanism_bins.csv': aggregated,
        'condition_correlations.csv': correlations,
        'nested_model_performance.csv': models,
        'nested_model_predictions.csv': predictions,
        'sensitivity_statistics.csv': sensitivity,
        'leave_one_dataset_out_influence.csv': influence,
    }
    for filename, table in tables.items():
        table.to_csv(os.path.join(results_dir, filename), index=False)
    with open(os.path.join(results_dir, 'primary_test.json'), 'w', encoding='utf-8') as handle:
        json.dump({**primary, 'decision': decision}, handle, indent=2)
    with open(os.path.join(results_dir, 'nested_model_test.json'), 'w', encoding='utf-8') as handle:
        json.dump(nested, handle, indent=2)

    sns.set_theme(style='whitegrid')
    _plot_oece(robustness, os.path.join(results_dir, EXPECTED_PLOTS[0]), dpi)
    _plot_swap_composition(aggregated, os.path.join(results_dir, EXPECTED_PLOTS[1]), dpi)
    _plot_mechanism(
        aggregated, robustness, os.path.join(results_dir, EXPECTED_PLOTS[2]), dpi)
    pd.DataFrame([
        {'plot_file': EXPECTED_PLOTS[0], 'source': 'dataset_condition_robustness.csv'},
        {'plot_file': EXPECTED_PLOTS[1], 'source': 'mechanism_bins.csv'},
        {'plot_file': EXPECTED_PLOTS[2],
         'source': 'mechanism_bins.csv; dataset_condition_robustness.csv'},
    ]).to_csv(os.path.join(results_dir, 'plots_manifest.csv'), index=False)
    _write_report(
        results_dir, primary, nested, correlations, models, sensitivity,
        influence, decision)
    return {
        'decision': decision,
        'median_spearman_rho': primary['median_spearman_rho'],
        'primary_permutation_p_value': primary['permutation_p_value'],
        'm2_minus_m3_rmse': nested['m2_minus_m3_rmse'],
        'model_permutation_p_value': nested['permutation_p_value'],
    }


def main():
    args = parse_args()
    args.prepared_dir = os.path.abspath(args.prepared_dir)
    args.results_dir = os.path.abspath(args.results_dir)
    write_run_config(args.results_dir, args)
    result = analyze_hypothesis(
        args.prepared_dir, args.results_dir, args.serial, args.exclude_datasets,
        args.expected_datasets, args.permutations, args.model_permutations,
        args.seed, args.dpi)
    print(f'decision={result["decision"]}')
    print(f'median_spearman_rho={result["median_spearman_rho"]:.6f}')
    print(f'primary_permutation_p={result["primary_permutation_p_value"]:.6f}')
    print(f'm2_minus_m3_rmse={result["m2_minus_m3_rmse"]:.6f}')
    print(f'Wrote hypothesis analysis to {args.results_dir}')


if __name__ == '__main__':
    main()
