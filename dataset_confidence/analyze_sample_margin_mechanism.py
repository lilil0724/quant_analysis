"""Test whether low baseline margins identify samples that swap after HQQ."""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

if __package__:
    from .analyze_dataset_difficulty import normalized_decision_margins
    from .common import PREPARED_DIR, write_run_config
else:
    from analyze_dataset_difficulty import normalized_decision_margins
    from common import PREPARED_DIR, write_run_config


PRIMARY_BITS = (3.0, 4.0)
PRIMARY_GROUP_SIZES = (8, 128)
PRIMARY_CONDITIONS = tuple(
    f'hqq_n{bits:g}_g{group}'
    for bits in PRIMARY_BITS for group in PRIMARY_GROUP_SIZES
)
DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results', 'sample_margin_mechanism')


def binary_auroc(scores, outcomes):
    """Return rank-based AUROC and an explicit identifiability status."""
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=bool)
    if scores.ndim != 1 or outcomes.shape != scores.shape or not len(scores):
        raise ValueError('scores and outcomes must be non-empty aligned vectors')
    if not np.isfinite(scores).all():
        raise ValueError('scores must be finite')
    positive = int(outcomes.sum())
    negative = int(len(outcomes) - positive)
    if not positive or not negative:
        return np.nan, 'undefined_single_class'
    ranks = pd.Series(scores).rank(method='average').to_numpy(dtype=float)
    rank_sum = float(ranks[outcomes].sum())
    auc = (
        rank_sum - positive * (positive + 1) / 2
    ) / (positive * negative)
    return float(auc), 'ok'


def assign_margin_deciles(normalized_margins):
    """Assign within-dataset deciles without splitting tied values."""
    values = np.asarray(normalized_margins, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError('normalized margins must be a non-empty vector')
    if not np.isfinite(values).all():
        raise ValueError('normalized margins must be finite')
    ranks = pd.Series(values).rank(method='average').to_numpy(dtype=float)
    deciles = np.floor((ranks - 1) * 10 / len(values)).astype(int) + 1
    return np.clip(deciles, 1, 10)


def _require_columns(frame, columns, source):
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{source} is missing columns: {", ".join(missing)}')


def _primary_transitions(
        prepared_dir, serial, excluded_datasets, expected_datasets):
    path = os.path.join(prepared_dir, 'transition_summary.csv')
    transitions = pd.read_csv(path)
    _require_columns(transitions, {
        'dataset_name', 'serial', 'condition', 'hqq_nbits',
        'hqq_group_size', 'pair_file',
    }, path)
    excluded = {str(value).lower() for value in excluded_datasets}
    primary = transitions[
        (transitions['serial'].astype(int) == int(serial))
        & transitions['hqq_nbits'].astype(float).isin(PRIMARY_BITS)
        & transitions['hqq_group_size'].astype(int).isin(PRIMARY_GROUP_SIZES)
        & ~transitions['dataset_name'].astype(str).str.lower().isin(excluded)
    ].copy()
    if primary.empty:
        raise ValueError('no primary 3-bit or 4-bit transition rows were found')
    if primary.duplicated(['dataset_name', 'condition']).any():
        raise ValueError('primary dataset-condition transition rows must be unique')
    datasets = sorted(primary['dataset_name'].astype(str).unique())
    if len(datasets) != int(expected_datasets):
        raise ValueError(
            f'expected {expected_datasets} datasets, found {len(datasets)}')
    expected = set(PRIMARY_CONDITIONS)
    for dataset, group in primary.groupby('dataset_name', sort=True):
        found = set(group['condition'].astype(str))
        if found != expected:
            missing = sorted(expected.difference(found))
            extra = sorted(found.difference(expected))
            raise ValueError(
                f'{dataset} primary conditions mismatch; '
                f'missing={missing}, extra={extra}')
    return primary.sort_values(['dataset_name', 'condition']).reset_index(drop=True)


def _baseline_manifest_rows(
        manifest_rows, serial, excluded_datasets, expected_datasets):
    _require_columns(manifest_rows, {
        'dataset_name', 'serial', 'condition', 'npz_path',
    }, 'manifest')
    excluded = {str(value).lower() for value in excluded_datasets}
    rows = manifest_rows[
        (manifest_rows['serial'].astype(int) == int(serial))
        & (manifest_rows['condition'] == 'fp_unquantized')
        & ~manifest_rows['dataset_name'].astype(str).str.lower().isin(excluded)
    ].copy()
    if rows.duplicated('dataset_name').any():
        raise ValueError('each dataset must have exactly one baseline artifact')
    if len(rows) != int(expected_datasets):
        raise ValueError(
            f'expected {expected_datasets} baseline artifacts, found {len(rows)}')
    indexed = rows.set_index('dataset_name')
    if not indexed.index.is_unique:
        raise ValueError('each dataset must have exactly one baseline artifact')
    return indexed


def _resolve_pair_path(prepared_dir, pair_file):
    path = os.fspath(pair_file)
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(prepared_dir, path))


def build_sample_mechanism_tables(
        prepared_dir, manifest_rows, serial=100004,
        excluded_datasets=('inat17',), expected_datasets=14, chunk_size=1024):
    """Build condition AUROC and decile rows while loading one baseline at a time."""
    transitions = _primary_transitions(
        prepared_dir, serial, excluded_datasets, expected_datasets)
    baselines = _baseline_manifest_rows(
        manifest_rows, serial, excluded_datasets, expected_datasets)
    transition_datasets = set(transitions['dataset_name'].astype(str))
    baseline_datasets = set(baselines.index.astype(str))
    if transition_datasets != baseline_datasets:
        raise ValueError('prepared pairs and baseline artifact datasets do not match')

    auroc_rows = []
    decile_rows = []
    for dataset, dataset_transitions in transitions.groupby(
            'dataset_name', sort=True):
        baseline_path = os.fspath(baselines.loc[dataset, 'npz_path'])
        with np.load(baseline_path, allow_pickle=False) as payload:
            missing = sorted({'sample_id', 'logits'}.difference(payload.files))
            if missing:
                raise ValueError(
                    f'{baseline_path} is missing arrays: {", ".join(missing)}')
            baseline_ids = np.asarray(payload['sample_id']).astype(str)
            if baseline_ids.ndim != 1 or len(np.unique(baseline_ids)) != len(
                    baseline_ids):
                raise ValueError(
                    f'{baseline_path} sample IDs must be a unique vector')
            margins = normalized_decision_margins(
                np.asarray(payload['logits']), chunk_size=chunk_size)
            if len(margins) != len(baseline_ids):
                raise ValueError(
                    f'{baseline_path} sample IDs and logits are not aligned')
        margin_by_id = dict(zip(baseline_ids, margins))

        for transition in dataset_transitions.to_dict('records'):
            pair_path = _resolve_pair_path(
                prepared_dir, transition['pair_file'])
            with np.load(pair_path, allow_pickle=False) as pair:
                missing = sorted({
                    'sample_id', 'baseline_correct', 'agreement',
                }.difference(pair.files))
                if missing:
                    raise ValueError(
                        f'{pair_path} is missing arrays: {", ".join(missing)}')
                sample_ids = np.asarray(pair['sample_id']).astype(str)
                baseline_correct = np.asarray(
                    pair['baseline_correct'], dtype=bool)
                agreement = np.asarray(pair['agreement'], dtype=bool)
            if (
                    sample_ids.ndim != 1
                    or baseline_correct.shape != sample_ids.shape
                    or agreement.shape != sample_ids.shape
                    or not len(sample_ids)):
                raise ValueError(f'{pair_path} contains incompatible sample vectors')
            if len(np.unique(sample_ids)) != len(sample_ids):
                raise ValueError(f'{pair_path} contains duplicate sample IDs')
            missing_ids = sorted(set(sample_ids).difference(margin_by_id))
            extra_ids = sorted(set(margin_by_id).difference(sample_ids))
            if missing_ids or extra_ids:
                raise ValueError(
                    f'{pair_path} and its baseline artifact have different '
                    'sample ID sets')
            aligned_margins = np.asarray(
                [margin_by_id[value] for value in sample_ids], dtype=float)
            swapped = ~agreement
            identity = {
                'dataset_name': str(dataset),
                'condition': str(transition['condition']),
                'hqq_nbits': float(transition['hqq_nbits']),
                'hqq_group_size': int(transition['hqq_group_size']),
            }
            for population, selected in (
                    ('all_matched', np.ones(len(sample_ids), dtype=bool)),
                    ('baseline_correct', baseline_correct)):
                selected_margins = aligned_margins[selected]
                selected_swaps = swapped[selected]
                if not len(selected_margins):
                    auc, status = np.nan, 'undefined_empty_population'
                else:
                    auc, status = binary_auroc(
                        -selected_margins, selected_swaps)
                auroc_rows.append({
                    **identity,
                    'population': population,
                    'n_samples': int(len(selected_margins)),
                    'swap_count': int(selected_swaps.sum()),
                    'non_swap_count': int(
                        len(selected_swaps) - selected_swaps.sum()),
                    'swap_rate': (
                        float(selected_swaps.mean())
                        if len(selected_swaps) else np.nan),
                    'auroc': auc,
                    'status': status,
                })

            deciles = assign_margin_deciles(aligned_margins)
            for decile in sorted(np.unique(deciles)):
                selected = deciles == decile
                decile_rows.append({
                    **identity,
                    'decile': int(decile),
                    'n_samples': int(selected.sum()),
                    'swap_count': int(swapped[selected].sum()),
                    'swap_rate': float(swapped[selected].mean()),
                    'mean_normalized_margin': float(
                        aligned_margins[selected].mean()),
                    'min_normalized_margin': float(
                        aligned_margins[selected].min()),
                    'max_normalized_margin': float(
                        aligned_margins[selected].max()),
                })
    return pd.DataFrame(auroc_rows), pd.DataFrame(decile_rows)


def _exact_positive_sign_test(values, chance=0.5):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    positive = int(np.count_nonzero(values > chance))
    negative = int(np.count_nonzero(values < chance))
    ties = int(np.count_nonzero(values == chance))
    n = positive + negative
    if not n:
        return {
            'positive_count': positive,
            'negative_count': negative,
            'tie_count': ties,
            'n_non_ties': n,
            'p_value': np.nan,
        }
    tail = sum(math.comb(n, k) for k in range(positive, n + 1))
    return {
        'positive_count': positive,
        'negative_count': negative,
        'tie_count': ties,
        'n_non_ties': n,
        'p_value': float(tail / (2 ** n)),
    }


def aggregate_auroc_results(auroc_rows, expected_datasets=14):
    """Create equal-weight dataset scores and evaluate the prespecified gate."""
    _require_columns(auroc_rows, {
        'dataset_name', 'condition', 'population', 'auroc', 'status',
    }, 'AUROC rows')
    primary = auroc_rows[
        auroc_rows['population'] == 'all_matched'].copy()
    if primary.duplicated(['dataset_name', 'condition']).any():
        raise ValueError('primary AUROC rows must be unique by dataset-condition')
    datasets = sorted(primary['dataset_name'].astype(str).unique())
    dataset_rows = []
    for dataset in datasets:
        group = primary[primary['dataset_name'] == dataset]
        found = set(group['condition'].astype(str))
        if found != set(PRIMARY_CONDITIONS):
            raise ValueError(f'{dataset} does not have all primary AUROC rows')
        identifiable = group[
            (group['status'] == 'ok') & np.isfinite(group['auroc'])]
        complete = len(identifiable) == len(PRIMARY_CONDITIONS)
        dataset_rows.append({
            'dataset_name': dataset,
            'identifiable_conditions': int(len(identifiable)),
            'complete': bool(complete),
            'mean_auroc': (
                float(identifiable['auroc'].mean()) if complete else np.nan),
            'min_condition_auroc': (
                float(identifiable['auroc'].min()) if complete else np.nan),
            'max_condition_auroc': (
                float(identifiable['auroc'].max()) if complete else np.nan),
        })
    dataset_scores = pd.DataFrame(dataset_rows)
    condition_summary = primary.groupby(
        'condition', as_index=False).agg(
            n_datasets=('dataset_name', 'nunique'),
            identifiable_datasets=('auroc', 'count'),
            median_auroc=('auroc', 'median'),
            mean_auroc=('auroc', 'mean'),
        )
    complete_scores = dataset_scores.loc[
        dataset_scores['complete'], 'mean_auroc'].to_numpy(dtype=float)
    sign = _exact_positive_sign_test(complete_scores)
    median_score = (
        float(np.median(complete_scores)) if len(complete_scores) else np.nan)
    all_complete = (
        len(dataset_scores) == int(expected_datasets)
        and bool(dataset_scores['complete'].all()))
    all_condition_medians_above_chance = bool(
        len(condition_summary) == len(PRIMARY_CONDITIONS)
        and (condition_summary['median_auroc'] > 0.5).all())
    if not all_complete:
        status = 'INCOMPLETE'
    elif (
            median_score >= 0.60
            and sign['p_value'] < 0.05
            and all_condition_medians_above_chance):
        status = 'PASS'
    else:
        status = 'FAIL'
    decision = {
        'status': status,
        'expected_datasets': int(expected_datasets),
        'observed_datasets': int(len(dataset_scores)),
        'complete_datasets': int(dataset_scores['complete'].sum()),
        'median_dataset_auroc': median_score,
        'median_threshold': 0.60,
        'chance_auroc': 0.50,
        'condition_medians_above_chance': all_condition_medians_above_chance,
        'sign_test_alternative': 'greater',
        'sign_test_p_value': sign['p_value'],
        'sign_test_positive_count': sign['positive_count'],
        'sign_test_negative_count': sign['negative_count'],
        'sign_test_tie_count': sign['tie_count'],
        'sign_test_n_non_ties': sign['n_non_ties'],
    }
    return dataset_scores, condition_summary, decision


def _bootstrap_mean_interval(values, bootstrap_samples, rng):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    indexes = rng.integers(
        0, len(values), size=(int(bootstrap_samples), len(values)))
    means = values[indexes].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_decile_curves(decile_rows, bootstrap_samples=2000, seed=0):
    """Summarize decile swap rates without weighting datasets by sample count."""
    if bootstrap_samples < 1:
        raise ValueError('bootstrap_samples must be positive')
    _require_columns(decile_rows, {
        'dataset_name', 'condition', 'decile', 'n_samples', 'swap_rate',
    }, 'decile rows')
    dataset_curves = decile_rows.groupby(
        ['dataset_name', 'decile'], as_index=False).agg(
            mean_swap_rate=('swap_rate', 'mean'),
            identifiable_conditions=('condition', 'nunique'),
            total_samples_across_conditions=('n_samples', 'sum'),
        )
    rng = np.random.default_rng(seed)
    aggregate_rows = []
    for decile, group in dataset_curves.groupby('decile', sort=True):
        low, high = _bootstrap_mean_interval(
            group['mean_swap_rate'], bootstrap_samples, rng)
        aggregate_rows.append({
            'decile': int(decile),
            'contributing_datasets': int(group['dataset_name'].nunique()),
            'mean_swap_rate': float(group['mean_swap_rate'].mean()),
            'median_swap_rate': float(group['mean_swap_rate'].median()),
            'ci95_low': low,
            'ci95_high': high,
        })

    condition_rows = []
    for (condition, decile), group in decile_rows.groupby(
            ['condition', 'decile'], sort=True):
        low, high = _bootstrap_mean_interval(
            group['swap_rate'], bootstrap_samples, rng)
        condition_rows.append({
            'condition': condition,
            'decile': int(decile),
            'contributing_datasets': int(group['dataset_name'].nunique()),
            'mean_swap_rate': float(group['swap_rate'].mean()),
            'median_swap_rate': float(group['swap_rate'].median()),
            'ci95_low': low,
            'ci95_high': high,
        })
    return (
        dataset_curves,
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(condition_rows),
    )


def _sensitivity_dataset_scores(auroc_rows):
    sensitivity = auroc_rows[
        auroc_rows['population'] == 'baseline_correct'].copy()
    return sensitivity.groupby('dataset_name', as_index=False).agg(
        identifiable_conditions=('auroc', 'count'),
        mean_auroc=('auroc', 'mean'),
        median_auroc=('auroc', 'median'),
    )


def _markdown_table(frame):
    columns = [str(column) for column in frame.columns]
    lines = [
        '| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join(['---'] * len(columns)) + ' |',
    ]
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                rendered.append(
                    'NA' if not np.isfinite(value) else f'{float(value):.6f}')
            else:
                rendered.append(str(value).replace('|', '\\|'))
        lines.append('| ' + ' | '.join(rendered) + ' |')
    return '\n'.join(lines)


def _plot_outputs(
        dataset_scores, condition_summary, dataset_curves, aggregate_curves,
        condition_curves, results_dir, dpi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plots = []
    fig, axis = plt.subplots(figsize=(7.5, 5.2))
    error = np.vstack([
        aggregate_curves['mean_swap_rate'] - aggregate_curves['ci95_low'],
        aggregate_curves['ci95_high'] - aggregate_curves['mean_swap_rate'],
    ])
    axis.errorbar(
        aggregate_curves['decile'], aggregate_curves['mean_swap_rate'],
        yerr=error, marker='o', capsize=3, color='#4c78a8')
    axis.set_xlabel('Within-dataset normalized-margin decile (low to high)')
    axis.set_ylabel('Dataset-equal mean prediction swap rate')
    axis.set_title('Low-margin samples are more vulnerable to HQQ swaps')
    axis.set_xticks(range(1, 11))
    axis.set_ylim(bottom=0)
    fig.tight_layout()
    name = 'aggregate_decile_swap_rate.png'
    fig.savefig(os.path.join(results_dir, name), dpi=dpi)
    plt.close(fig)
    plots.append(name)

    fig, axis = plt.subplots(figsize=(8, 5.5))
    for condition, group in condition_curves.groupby('condition', sort=True):
        axis.plot(
            group['decile'], group['mean_swap_rate'], marker='o',
            label=condition)
    axis.set_xlabel('Within-dataset normalized-margin decile (low to high)')
    axis.set_ylabel('Dataset-equal mean prediction swap rate')
    axis.set_title('Margin-decile swap curves by primary HQQ condition')
    axis.set_xticks(range(1, 11))
    axis.set_ylim(bottom=0)
    axis.legend(fontsize=8)
    fig.tight_layout()
    name = 'condition_decile_swap_rate.png'
    fig.savefig(os.path.join(results_dir, name), dpi=dpi)
    plt.close(fig)
    plots.append(name)

    fig, axis = plt.subplots(figsize=(9, 6))
    for dataset, group in dataset_curves.groupby('dataset_name', sort=True):
        axis.plot(
            group['decile'], group['mean_swap_rate'], alpha=.65,
            linewidth=1.2, label=dataset)
    axis.set_xlabel('Within-dataset normalized-margin decile (low to high)')
    axis.set_ylabel('Mean swap rate across primary conditions')
    axis.set_title('Within-dataset margin-decile curves')
    axis.set_xticks(range(1, 11))
    axis.set_ylim(bottom=0)
    axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    name = 'dataset_decile_swap_rate.png'
    fig.savefig(os.path.join(results_dir, name), dpi=dpi)
    plt.close(fig)
    plots.append(name)

    ordered = dataset_scores.sort_values('mean_auroc')
    fig, axis = plt.subplots(figsize=(8, 5.5))
    colors = [
        '#59a14f' if value >= .5 else '#e15759'
        for value in ordered['mean_auroc']
    ]
    axis.barh(ordered['dataset_name'], ordered['mean_auroc'], color=colors)
    axis.axvline(.5, color='black', linestyle='--', linewidth=1)
    axis.set_xlabel('Mean AUROC across four primary conditions')
    axis.set_ylabel('Dataset-model pair')
    axis.set_title('Sample-level margin discrimination by dataset')
    fig.tight_layout()
    name = 'dataset_auroc.png'
    fig.savefig(os.path.join(results_dir, name), dpi=dpi)
    plt.close(fig)
    plots.append(name)
    return plots


def _write_report(path, decision, condition_summary, dataset_scores):
    lines = [
        '# Sample-level normalized-margin mechanism test', '',
        '## Scope', '',
        '- Primary population: all matched samples.',
        '- Sensitivity population: baseline-correct samples.',
        '- Primary conditions: 3/4-bit HQQ at group sizes 8 and 128.',
        '- Each dataset contributes one equal-weight mean of four condition AUROCs.',
        '- The result is associative and does not establish a causal margin effect.',
        '', '## Prespecified result', '',
        f'- Gate status: **{decision["status"]}**.',
        f'- Median dataset-level AUROC: '
        f'{decision["median_dataset_auroc"]:.6f}.',
        f'- One-sided exact sign-test p: '
        f'{decision["sign_test_p_value"]:.6f}.',
        f'- Dataset scores above/below/equal to 0.50: '
        f'{decision["sign_test_positive_count"]}/'
        f'{decision["sign_test_negative_count"]}/'
        f'{decision["sign_test_tie_count"]}.',
        f'- Complete datasets: {decision["complete_datasets"]}/'
        f'{decision["expected_datasets"]}.',
        f'- Every condition median above 0.50: '
        f'{decision["condition_medians_above_chance"]}.',
        '', '## Condition results', '',
        _markdown_table(condition_summary), '',
        '## Dataset-level scores', '',
        _markdown_table(dataset_scores), '',
        '## Decision rule', '',
        '- PASS requires median dataset AUROC at least 0.60, one-sided exact '
        'sign-test p below 0.05, and all four condition medians above 0.50.',
        '- A dataset score requires identifiable AUROC in all four primary '
        'conditions; otherwise the overall gate is INCOMPLETE.',
        '- Baseline-correct AUROC is sensitivity evidence and cannot change '
        'the primary gate.',
    ]
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def _json_write(path, value):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, default=str)


def analyze_sample_margin_mechanism(
        prepared_dir, manifest_file, results_dir, serial=100004,
        excluded_datasets=('inat17',), expected_datasets=14,
        bootstrap_samples=2000, seed=100004, chunk_size=1024, dpi=180):
    """Run the prespecified within-dataset sample margin analysis."""
    if bootstrap_samples < 1:
        raise ValueError('bootstrap_samples must be positive')
    os.makedirs(results_dir, exist_ok=True)
    manifest = pd.read_csv(manifest_file)
    auroc_rows, decile_rows = build_sample_mechanism_tables(
        prepared_dir, manifest, serial=serial,
        excluded_datasets=excluded_datasets,
        expected_datasets=expected_datasets, chunk_size=chunk_size)
    dataset_scores, condition_summary, decision = aggregate_auroc_results(
        auroc_rows, expected_datasets=expected_datasets)
    sensitivity_scores = _sensitivity_dataset_scores(auroc_rows)
    dataset_curves, aggregate_curves, condition_curves = summarize_decile_curves(
        decile_rows, bootstrap_samples=bootstrap_samples, seed=seed)

    auroc_rows.to_csv(
        os.path.join(results_dir, 'condition_auroc.csv'), index=False)
    dataset_scores.to_csv(
        os.path.join(results_dir, 'dataset_auroc.csv'), index=False)
    condition_summary.to_csv(
        os.path.join(results_dir, 'condition_auroc_summary.csv'), index=False)
    sensitivity_scores.to_csv(
        os.path.join(results_dir, 'baseline_correct_sensitivity.csv'),
        index=False)
    decile_rows.to_csv(
        os.path.join(results_dir, 'dataset_condition_deciles.csv'), index=False)
    dataset_curves.to_csv(
        os.path.join(results_dir, 'dataset_decile_curves.csv'), index=False)
    aggregate_curves.to_csv(
        os.path.join(results_dir, 'aggregate_decile_curve.csv'), index=False)
    condition_curves.to_csv(
        os.path.join(results_dir, 'condition_decile_curves.csv'), index=False)
    _json_write(os.path.join(results_dir, 'decision.json'), decision)
    plots = _plot_outputs(
        dataset_scores, condition_summary, dataset_curves, aggregate_curves,
        condition_curves, results_dir, dpi)
    pd.DataFrame({'plot_file': plots}).to_csv(
        os.path.join(results_dir, 'plots_manifest.csv'), index=False)
    _write_report(
        os.path.join(results_dir, 'report.md'),
        decision, condition_summary, dataset_scores)
    write_run_config(results_dir, {
        'prepared_dir': prepared_dir,
        'manifest_file': manifest_file,
        'results_dir': results_dir,
        'serial': serial,
        'excluded_datasets': list(excluded_datasets),
        'expected_datasets': expected_datasets,
        'bootstrap_samples': bootstrap_samples,
        'seed': seed,
        'chunk_size': chunk_size,
        'dpi': dpi,
    })
    return decision


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Test whether low baseline normalized margins identify samples '
            'whose top-1 prediction swaps after HQQ.'))
    parser.add_argument('--prepared-dir', default=PREPARED_DIR)
    parser.add_argument('--manifest-file', required=True)
    parser.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--serial', type=int, default=100004)
    parser.add_argument('--exclude-datasets', nargs='*', default=['inat17'])
    parser.add_argument('--expected-datasets', type=int, default=14)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=100004)
    parser.add_argument('--chunk-size', type=int, default=1024)
    parser.add_argument('--dpi', type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    result = analyze_sample_margin_mechanism(
        args.prepared_dir, args.manifest_file, args.results_dir,
        serial=args.serial, excluded_datasets=args.exclude_datasets,
        expected_datasets=args.expected_datasets,
        bootstrap_samples=args.bootstrap_samples, seed=args.seed,
        chunk_size=args.chunk_size, dpi=args.dpi)
    print(f'sample_margin_mechanism_status={result["status"]}')
    print(f'median_dataset_auroc={result["median_dataset_auroc"]:.6f}')
    print(f'sign_test_p={result["sign_test_p_value"]:.6f}')


if __name__ == '__main__':
    main()
