import argparse
import json
import os

import numpy as np
import pandas as pd

if __package__:
    from .analyze_overconfidence import prepare_hypothesis_tables, spearman
    from .common import PREPARED_DIR, write_run_config
else:
    from analyze_overconfidence import prepare_hypothesis_tables, spearman
    from common import PREPARED_DIR, write_run_config


PRIMARY_BITS = (3.0, 4.0)
PRIMARY_GROUP_SIZES = (8, 128)
DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results', 'dataset_difficulty')


def normalized_decision_margins(logits, chunk_size=1024):
    """Return top-1/top-2 logit gaps normalized by per-row logit spread."""
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError('logits must be a two-dimensional array with at least two classes')
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    if not np.isfinite(values).all():
        raise ValueError('logits contain non-finite values')

    result = np.empty(values.shape[0], dtype=np.float64)
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        block = values[start:stop]
        top_two = np.partition(block, block.shape[1] - 2, axis=1)[:, -2:]
        gap = np.abs(top_two[:, 1] - top_two[:, 0]).astype(np.float64)
        spread = block.astype(np.float64, copy=False).std(axis=1)
        if np.any(spread == 0):
            raise ValueError('cannot normalize logits with zero within-sample spread')
        result[start:stop] = gap / spread
    return result


def extract_baseline_logit_features(
        manifest_rows, serial=100004, excluded_datasets=('inat17',),
        chunk_size=1024):
    """Read one baseline artifact at a time and summarize its logit geometry."""
    required = {'dataset_name', 'serial', 'condition', 'npz_path'}
    missing = sorted(required.difference(manifest_rows.columns))
    if missing:
        raise ValueError(f'manifest is missing columns: {", ".join(missing)}')
    excluded = {str(value).lower() for value in excluded_datasets}
    baselines = manifest_rows[
        (manifest_rows['serial'].astype(int) == int(serial))
        & (manifest_rows['condition'] == 'fp_unquantized')
        & ~manifest_rows['dataset_name'].str.lower().isin(excluded)
    ].copy()
    if baselines.empty:
        raise ValueError('no matched baseline artifacts were found')
    if baselines.duplicated('dataset_name').any():
        raise ValueError('each dataset must have exactly one baseline artifact')

    rows = []
    identity_columns = [
        name for name in (
            'sweep_id', 'project', 'dataset_name', 'model_name',
            'checkpoint_sha256', 'serial', 'seed', 'run_id',
        ) if name in baselines.columns
    ]
    for _, baseline in baselines.sort_values('dataset_name').iterrows():
        path = baseline['npz_path']
        with np.load(path, allow_pickle=False) as payload:
            needed = {
                'logits', 'logit_margin', 'top1_probability',
                'top2_probability', 'normalized_entropy',
            }
            missing_arrays = sorted(needed.difference(payload.files))
            if missing_arrays:
                raise ValueError(
                    f'{path} is missing arrays: {", ".join(missing_arrays)}')
            logits = np.asarray(payload['logits'])
            normalized = normalized_decision_margins(logits, chunk_size=chunk_size)
            raw_margin = np.asarray(payload['logit_margin'], dtype=float)
            probability_gap = (
                np.asarray(payload['top1_probability'], dtype=float)
                - np.asarray(payload['top2_probability'], dtype=float)
            )
            normalized_entropy = np.asarray(
                payload['normalized_entropy'], dtype=float)
            if not (
                    len(raw_margin) == len(normalized)
                    == len(probability_gap) == len(normalized_entropy)):
                raise ValueError(f'{path} contains incompatible sample vectors')
            rows.append({
                **{name: baseline[name] for name in identity_columns},
                'npz_path': path,
                'n_samples': int(len(normalized)),
                'num_classes': int(logits.shape[1]),
                'normalized_margin_q05': float(np.quantile(normalized, .05)),
                'normalized_margin_q10': float(np.quantile(normalized, .10)),
                'normalized_margin_q25': float(np.quantile(normalized, .25)),
                'median_normalized_margin': float(np.median(normalized)),
                'mean_normalized_margin': float(np.mean(normalized)),
                'mean_raw_margin': float(np.mean(raw_margin)),
                'mean_probability_gap': float(np.mean(probability_gap)),
                'mean_normalized_entropy': float(np.mean(normalized_entropy)),
            })
    return pd.DataFrame(rows).sort_values('dataset_name').reset_index(drop=True)


def build_difficulty_table(
        condition_rows, primary_bits=PRIMARY_BITS,
        primary_group_sizes=PRIMARY_GROUP_SIZES):
    """Build the prespecified condition-adjusted dataset difficulty score."""
    required = {
        'dataset_name', 'condition', 'hqq_nbits', 'hqq_group_size',
        'accuracy_ratio',
    }
    missing = sorted(required.difference(condition_rows.columns))
    if missing:
        raise ValueError(f'condition rows are missing columns: {", ".join(missing)}')
    primary = condition_rows[
        condition_rows['hqq_nbits'].astype(float).isin(primary_bits)
        & condition_rows['hqq_group_size'].astype(int).isin(primary_group_sizes)
    ].copy()
    if primary.empty:
        raise ValueError('no primary 3-bit or 4-bit conditions were found')
    if primary.duplicated(['dataset_name', 'condition']).any():
        raise ValueError('dataset-condition rows must be unique')

    expected_conditions = len(primary_bits) * len(primary_group_sizes)
    counts = primary.groupby('dataset_name')['condition'].nunique()
    if counts.nunique() != 1 or int(counts.iloc[0]) != expected_conditions:
        raise ValueError('every dataset must have all four primary conditions')
    if primary['accuracy_ratio'].isna().any():
        raise ValueError('primary accuracy ratios must be finite')

    primary['relative_accuracy_loss'] = 1 - primary['accuracy_ratio'].astype(float)
    primary['difficulty_percentile'] = primary.groupby('condition')[
        'relative_accuracy_loss'].rank(method='average', pct=True)
    difficulty = primary.groupby('dataset_name', as_index=False).agg(
        difficulty_score=('difficulty_percentile', 'mean'),
        median_relative_accuracy_loss=('relative_accuracy_loss', 'median'),
        mean_relative_accuracy_loss=('relative_accuracy_loss', 'mean'),
    )

    rank_matrix = primary.pivot(
        index='dataset_name', columns='condition',
        values='relative_accuracy_loss').rank(method='average', axis=0)
    n_datasets, n_conditions = rank_matrix.shape
    if n_datasets < 2 or n_conditions < 2:
        raise ValueError('Kendall concordance needs at least two datasets and conditions')
    rank_sums = rank_matrix.sum(axis=1).to_numpy(dtype=float)
    squared_deviation = np.square(rank_sums - rank_sums.mean()).sum()
    tie_sum = 0.0
    for condition in rank_matrix.columns:
        counts_for_rank = rank_matrix[condition].value_counts().to_numpy(dtype=float)
        tie_sum += np.sum(counts_for_rank ** 3 - counts_for_rank)
    denominator = (
        n_conditions ** 2 * (n_datasets ** 3 - n_datasets)
        - n_conditions * tie_sum
    )
    if denominator <= 0:
        raise ValueError('Kendall concordance is undefined for constant rankings')
    kendalls_w = float(12 * squared_deviation / denominator)
    concordance = {
        'n_datasets': int(n_datasets),
        'n_conditions': int(n_conditions),
        'kendalls_w': kendalls_w,
    }
    return difficulty.sort_values('dataset_name'), primary, concordance


def _condition_correlations(frame, predictor, outcome):
    rows = []
    for condition, group in frame.groupby('condition', sort=True):
        valid = group[['dataset_name', predictor, outcome]].dropna()
        row = {
            'condition': condition,
            'predictor': predictor,
            'outcome': outcome,
            'n_datasets': int(len(valid)),
            'spearman_rho': spearman(valid[predictor], valid[outcome]),
        }
        if 'hqq_nbits' in group:
            row['hqq_nbits'] = float(group['hqq_nbits'].iloc[0])
        if 'hqq_group_size' in group:
            row['hqq_group_size'] = int(group['hqq_group_size'].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def dataset_block_permutation_test(
        frame, predictor, outcome, alternative, permutations=10000, seed=0):
    """Permute one dataset-level predictor while preserving condition blocks."""
    if alternative not in ('negative', 'positive'):
        raise ValueError('alternative must be negative or positive')
    if permutations < 1:
        raise ValueError('permutations must be positive')
    required = {'dataset_name', 'condition', predictor, outcome}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f'analysis rows are missing columns: {", ".join(missing)}')
    predictor_matrix = frame.pivot(
        index='dataset_name', columns='condition', values=predictor)
    outcome_matrix = frame.pivot(
        index='dataset_name', columns='condition', values=outcome)
    if list(predictor_matrix.columns) != list(outcome_matrix.columns):
        raise ValueError('predictor and outcome conditions do not match')
    datasets = sorted(set(predictor_matrix.index).intersection(outcome_matrix.index))
    x_matrix = predictor_matrix.reindex(datasets).to_numpy(dtype=float)
    y_matrix = outcome_matrix.reindex(datasets).to_numpy(dtype=float)
    valid = np.isfinite(x_matrix).all(axis=1) & np.isfinite(y_matrix).all(axis=1)
    datasets = [dataset for dataset, keep in zip(datasets, valid) if keep]
    x_matrix = x_matrix[valid]
    y_matrix = y_matrix[valid]
    if len(x_matrix) < 3:
        raise ValueError('at least three complete dataset blocks are required')

    complete = frame[frame['dataset_name'].isin(datasets)].copy()
    correlations = _condition_correlations(complete, predictor, outcome)
    observed = float(correlations['spearman_rho'].median())
    if not np.isfinite(observed):
        raise ValueError('observed median correlation is undefined')
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=float)
    for index in range(permutations):
        shuffled = x_matrix[rng.permutation(len(x_matrix))]
        null[index] = float(np.median([
            spearman(shuffled[:, column], y_matrix[:, column])
            for column in range(y_matrix.shape[1])
        ]))
    if alternative == 'negative':
        extreme = np.count_nonzero(null <= observed)
    else:
        extreme = np.count_nonzero(null >= observed)
    result = {
        'predictor': predictor,
        'outcome': outcome,
        'alternative': alternative,
        'n_datasets': int(len(x_matrix)),
        'n_conditions': int(y_matrix.shape[1]),
        'median_spearman_rho': observed,
        'permutations': int(permutations),
        'permutation_p_value': float((1 + extreme) / (permutations + 1)),
        'seed': int(seed),
    }
    return result, correlations


def _lodo_predictions(ranked, q10_values=None):
    q10 = (ranked['normalized_margin_q10_rank'].to_numpy(dtype=float)
           if q10_values is None else np.asarray(q10_values, dtype=float))
    y = ranked['difficulty_rank'].to_numpy(dtype=float)
    baseline = ranked['baseline_accuracy_rank'].to_numpy(dtype=float)
    classes = ranked['log_num_classes_rank'].to_numpy(dtype=float)
    rows = []
    for held_out in range(len(ranked)):
        train = np.arange(len(ranked)) != held_out
        m0_train = np.column_stack([
            np.ones(train.sum()), baseline[train], classes[train],
        ])
        m1_train = np.column_stack([
            np.ones(train.sum()), baseline[train], classes[train], q10[train],
        ])
        beta0 = np.linalg.lstsq(m0_train, y[train], rcond=None)[0]
        beta1 = np.linalg.lstsq(m1_train, y[train], rcond=None)[0]
        x0 = np.asarray([1, baseline[held_out], classes[held_out]], dtype=float)
        x1 = np.asarray(
            [1, baseline[held_out], classes[held_out], q10[held_out]], dtype=float)
        rows.append({
            'dataset_name': ranked.iloc[held_out]['dataset_name'],
            'observed_difficulty_rank': y[held_out],
            'm0_prediction': float(x0 @ beta0),
            'm1_prediction': float(x1 @ beta1),
        })
    predictions = pd.DataFrame(rows)
    mae0 = float(np.mean(np.abs(
        predictions['observed_difficulty_rank'] - predictions['m0_prediction'])))
    mae1 = float(np.mean(np.abs(
        predictions['observed_difficulty_rank'] - predictions['m1_prediction'])))
    return predictions, mae0, mae1


def nested_lodo_comparison(frame, permutations=10000, seed=0):
    """Compare prespecified baseline controls with and without Q10 margin."""
    required = {
        'dataset_name', 'difficulty_score', 'baseline_accuracy',
        'num_classes', 'normalized_margin_q10',
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f'dataset table is missing columns: {", ".join(missing)}')
    if len(frame) < 6:
        raise ValueError('nested leave-one-out analysis needs at least six datasets')
    if permutations < 1:
        raise ValueError('permutations must be positive')
    ranked = frame.sort_values('dataset_name').reset_index(drop=True).copy()
    ranked['difficulty_rank'] = ranked['difficulty_score'].rank(pct=True)
    ranked['baseline_accuracy_rank'] = ranked['baseline_accuracy'].rank(pct=True)
    ranked['log_num_classes_rank'] = np.log(
        ranked['num_classes'].astype(float)).rank(pct=True)
    ranked['normalized_margin_q10_rank'] = ranked[
        'normalized_margin_q10'].rank(pct=True)
    predictions, mae0, mae1 = _lodo_predictions(ranked)
    observed_improvement = mae0 - mae1

    rng = np.random.default_rng(seed)
    q10 = ranked['normalized_margin_q10_rank'].to_numpy(dtype=float)
    null = np.empty(permutations, dtype=float)
    for index in range(permutations):
        _, _, permuted_mae1 = _lodo_predictions(
            ranked, q10_values=q10[rng.permutation(len(q10))])
        null[index] = mae0 - permuted_mae1
    p_value = float(
        (1 + np.count_nonzero(null >= observed_improvement)) / (permutations + 1))
    result = {
        'n_datasets': int(len(ranked)),
        'm0_predictors': ['baseline_accuracy', 'log_num_classes'],
        'm1_added_predictor': 'normalized_margin_q10',
        'm0_mae': mae0,
        'm1_mae': mae1,
        'mae_improvement_m0_minus_m1': float(observed_improvement),
        'permutations': int(permutations),
        'permutation_p_value': p_value,
        'seed': int(seed),
    }
    return result, predictions


def leave_one_dataset_out_correlations(frame, predictor, outcome):
    rows = []
    full = spearman(frame[predictor], frame[outcome])
    for dataset in sorted(frame['dataset_name'].unique()):
        reduced = frame[frame['dataset_name'] != dataset]
        rho = spearman(reduced[predictor], reduced[outcome])
        rows.append({
            'excluded_dataset': dataset,
            'spearman_rho': rho,
            'delta_from_full': rho - full,
        })
    return pd.DataFrame(rows)


def _benjamini_hochberg(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for reverse_index in range(len(values) - 1, -1, -1):
        original_index = order[reverse_index]
        candidate = values[original_index] * len(values) / (reverse_index + 1)
        running = min(running, candidate)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def _json_write(path, value):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, default=str)


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
                rendered.append(f'{float(value):.6f}')
            else:
                rendered.append(str(value).replace('|', '\\|'))
        lines.append('| ' + ' | '.join(rendered) + ' |')
    return '\n'.join(lines)


def _plot_outputs(dataset_table, primary_rows, results_dir, dpi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plots = []
    ordered = dataset_table.sort_values('difficulty_score')
    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.barh(ordered['dataset_name'], ordered['difficulty_score'], color='#4c78a8')
    axis.set_xlabel('Condition-adjusted difficulty percentile')
    axis.set_ylabel('Dataset-model pair')
    axis.set_title('Exploratory 3/4-bit HQQ difficulty ranking')
    fig.tight_layout()
    name = 'difficulty_ranking.png'
    fig.savefig(os.path.join(results_dir, name), dpi=dpi)
    plt.close(fig)
    plots.append(name)

    fig, axis = plt.subplots(figsize=(7.5, 6))
    axis.scatter(
        dataset_table['normalized_margin_q10'], dataset_table['difficulty_score'],
        color='#e45756', s=42)
    for row in dataset_table.to_dict('records'):
        axis.annotate(
            row['dataset_name'],
            (row['normalized_margin_q10'], row['difficulty_score']),
            xytext=(4, 3), textcoords='offset points', fontsize=7)
    axis.set_xlabel('Baseline normalized logit margin Q10')
    axis.set_ylabel('Condition-adjusted difficulty percentile')
    axis.set_title('Vulnerable-tail margin vs quantization difficulty')
    fig.tight_layout()
    name = 'q10_vs_difficulty.png'
    fig.savefig(os.path.join(results_dir, name), dpi=dpi)
    plt.close(fig)
    plots.append(name)

    heatmap = primary_rows.pivot(
        index='dataset_name', columns='condition', values='difficulty_percentile')
    heatmap = heatmap.loc[ordered['dataset_name']]
    fig, axis = plt.subplots(figsize=(7.5, 6))
    image = axis.imshow(heatmap.to_numpy(), aspect='auto', vmin=0, vmax=1,
                        cmap='magma')
    axis.set_xticks(range(len(heatmap.columns)), heatmap.columns, rotation=30,
                    ha='right')
    axis.set_yticks(range(len(heatmap.index)), heatmap.index)
    axis.set_title('Within-condition difficulty percentiles')
    fig.colorbar(image, ax=axis, label='Difficulty percentile')
    fig.tight_layout()
    name = 'condition_difficulty_heatmap.png'
    fig.savefig(os.path.join(results_dir, name), dpi=dpi)
    plt.close(fig)
    plots.append(name)

    conditions = sorted(primary_rows['condition'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for axis, condition in zip(axes.ravel(), conditions):
        group = primary_rows[primary_rows['condition'] == condition]
        axis.scatter(group['normalized_margin_q10'], group['swap_rate'], s=28)
        axis.set_title(condition)
        axis.set_xlabel('Normalized margin Q10')
        axis.set_ylabel('Swap rate')
    fig.suptitle('Vulnerable-tail margin and prediction swaps')
    fig.tight_layout()
    name = 'q10_vs_swap_by_condition.png'
    fig.savefig(os.path.join(results_dir, name), dpi=dpi)
    plt.close(fig)
    plots.append(name)
    return plots


def _write_report(
        path, dataset_table, concordance, primary, condition_correlations,
        influence, nested, margin_swap, swap_loss, mechanism_supported,
        promising, boundary, sensitivity):
    hardest = dataset_table.nlargest(5, 'difficulty_score')[
        ['dataset_name', 'difficulty_score', 'median_relative_accuracy_loss']]
    easiest = dataset_table.nsmallest(5, 'difficulty_score')[
        ['dataset_name', 'difficulty_score', 'median_relative_accuracy_loss']]
    influential = influence.loc[influence['delta_from_full'].abs().idxmax()]
    negative_conditions = int((condition_correlations['spearman_rho'] < 0).sum())
    lines = [
        '# Dataset–model quantization difficulty screening', '',
        '## Scope', '',
        '- Serial `100004`; `inat17` excluded.',
        '- Primary: 3/4-bit HQQ at group sizes 8 and 128.',
        '- Boundary sensitivity: 2-bit and 8-bit; 1/1.58-bit excluded.',
        '- One fine-tuned ViT-B/16 checkpoint per dataset; results are exploratory ',
        '  dataset–model pair associations, not intrinsic or causal dataset effects.', '',
        '## Prespecified primary result', '',
        f'- Kendall’s W across primary conditions: {concordance["kendalls_w"]:.6f}.',
        f'- Q10 margin vs difficulty Spearman rho: '
        f'{primary["median_spearman_rho"]:.6f}.',
        f'- One-sided dataset permutation p: '
        f'{primary["permutation_p_value"]:.6f} '
        f'({primary["permutations"]} permutations).',
        f'- Negative condition-wise directions: {negative_conditions}/4.',
        f'- Most influential deletion: `{influential["excluded_dataset"]}` '
        f'(delta rho={influential["delta_from_full"]:.6f}).',
        f'- Prespecified promising gate: **{"PASS" if promising else "FAIL"}**.', '',
        '## Baseline-control predictive check', '',
        f'- M0 LODO MAE (baseline accuracy + log classes): {nested["m0_mae"]:.6f}.',
        f'- M1 LODO MAE (+ normalized margin Q10): {nested["m1_mae"]:.6f}.',
        f'- M0 minus M1 improvement: '
        f'{nested["mae_improvement_m0_minus_m1"]:.6f}.',
        f'- Dataset permutation p: {nested["permutation_p_value"]:.6f}.', '',
        '## Mechanism consistency', '',
        f'- Q10 margin → swap median rho: '
        f'{margin_swap["median_spearman_rho"]:.6f}, '
        f'p={margin_swap["permutation_p_value"]:.6f}.',
        f'- Swap → relative accuracy loss median rho: '
        f'{swap_loss["median_spearman_rho"]:.6f}, '
        f'p={swap_loss["permutation_p_value"]:.6f}.',
        f'- Margin mechanism gate: '
        f'**{"PASS" if mechanism_supported else "FAIL"}**.', '',
        '## Hardest dataset–model pairs', '',
        _markdown_table(hardest), '',
        '## Easiest dataset–model pairs', '',
        _markdown_table(easiest), '',
        '## Boundary and sensitivity status', '',
        '- The 2-bit and 8-bit results below are stress tests, not independent validation.',
    ]
    for item in boundary:
        lines.append(
            f'- {item["subset"]}: rho={item["median_spearman_rho"]:.6f}, '
            f'p={item["permutation_p_value"]:.6f}.')
    lines.extend([
        '', '- Alternate logit predictors are sensitivity analyses with BH-FDR; '
        '  none may replace normalized-margin Q10 as the primary predictor.', '',
        _markdown_table(sensitivity), '',
        '## Decision rule', '',
        '- `promising` requires W ≥ 0.6, primary rho ≤ -0.5 with p < 0.05, '
        '  negative direction in at least 3/4 conditions, and negative rho after '
        '  every leave-one-dataset-out deletion.',
        '- Mechanism support additionally requires Q10→swap rho ≤ -0.5 and '
        '  swap→loss rho ≥ 0.5, each with p < 0.05 and expected direction in '
        '  at least 3/4 conditions.',
    ])
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def analyze_dataset_difficulty(
        prepared_dir, manifest_file, results_dir, serial=100004,
        excluded_datasets=('inat17',), expected_datasets=14,
        permutations=10000, model_permutations=1000, seed=100004,
        chunk_size=1024, dpi=180):
    """Run the prespecified exploratory dataset-difficulty analysis."""
    os.makedirs(results_dir, exist_ok=True)
    baseline, _, robustness, _ = prepare_hypothesis_tables(
        prepared_dir, serial=serial, excluded_datasets=excluded_datasets,
        expected_datasets=expected_datasets)
    manifest = pd.read_csv(manifest_file)
    logit_features = extract_baseline_logit_features(
        manifest, serial=serial, excluded_datasets=excluded_datasets,
        chunk_size=chunk_size)
    if set(baseline['dataset_name']) != set(logit_features['dataset_name']):
        raise ValueError('prepared pairs and baseline artifact datasets do not match')

    artifact_classes = logit_features.set_index('dataset_name')['num_classes']
    prepared_classes = baseline.set_index('dataset_name')['num_classes']
    if not artifact_classes.equals(prepared_classes.reindex(artifact_classes.index)):
        raise ValueError('baseline artifact and prepared class counts do not match')
    feature_columns = [
        'dataset_name', 'normalized_margin_q05', 'normalized_margin_q10',
        'normalized_margin_q25', 'median_normalized_margin',
        'mean_normalized_margin', 'mean_raw_margin', 'mean_probability_gap',
    ]
    baseline_features = baseline.merge(
        logit_features[feature_columns], on='dataset_name', how='inner',
        validate='one_to_one')
    difficulty, primary_rows, concordance = build_difficulty_table(robustness)
    dataset_table = difficulty.merge(
        baseline_features, on='dataset_name', how='inner', validate='one_to_one')
    primary_rows = primary_rows.merge(
        baseline_features, on='dataset_name', how='inner', validate='many_to_one',
        suffixes=('', '_baseline'))

    aggregate = dataset_table[[
        'dataset_name', 'normalized_margin_q10', 'difficulty_score',
    ]].copy()
    aggregate['condition'] = 'aggregate'
    primary_test, _ = dataset_block_permutation_test(
        aggregate, 'normalized_margin_q10', 'difficulty_score',
        alternative='negative', permutations=permutations, seed=seed)
    _, primary_condition_correlations = dataset_block_permutation_test(
        primary_rows, 'normalized_margin_q10', 'relative_accuracy_loss',
        alternative='negative', permutations=permutations, seed=seed + 1)
    influence = leave_one_dataset_out_correlations(
        dataset_table, 'normalized_margin_q10', 'difficulty_score')
    nested, nested_predictions = nested_lodo_comparison(
        dataset_table, permutations=model_permutations, seed=seed + 2)

    margin_swap, margin_swap_correlations = dataset_block_permutation_test(
        primary_rows, 'normalized_margin_q10', 'swap_rate',
        alternative='negative', permutations=permutations, seed=seed + 3)
    swap_loss, swap_loss_correlations = dataset_block_permutation_test(
        primary_rows, 'swap_rate', 'relative_accuracy_loss',
        alternative='positive', permutations=permutations, seed=seed + 4)
    margin_swap_directions = int(
        (margin_swap_correlations['spearman_rho'] < 0).sum())
    swap_loss_directions = int(
        (swap_loss_correlations['spearman_rho'] > 0).sum())
    mechanism_supported = bool(
        margin_swap['median_spearman_rho'] <= -.5
        and margin_swap['permutation_p_value'] < .05
        and margin_swap_directions >= 3
        and swap_loss['median_spearman_rho'] >= .5
        and swap_loss['permutation_p_value'] < .05
        and swap_loss_directions >= 3)

    negative_conditions = int(
        (primary_condition_correlations['spearman_rho'] < 0).sum())
    promising = bool(
        concordance['kendalls_w'] >= .6
        and primary_test['median_spearman_rho'] <= -.5
        and primary_test['permutation_p_value'] < .05
        and negative_conditions >= 3
        and (influence['spearman_rho'] < 0).all())

    boundary = []
    boundary_correlations = []
    for bits in (2.0, 8.0):
        subset = robustness[np.isclose(robustness['hqq_nbits'], bits)].copy()
        subset['relative_accuracy_loss'] = 1 - subset['accuracy_ratio']
        subset = subset.merge(
            baseline_features[['dataset_name', 'normalized_margin_q10']],
            on='dataset_name', how='inner', validate='many_to_one')
        test, correlations = dataset_block_permutation_test(
            subset, 'normalized_margin_q10', 'relative_accuracy_loss',
            alternative='negative', permutations=permutations,
            seed=seed + 10 + int(bits))
        test['subset'] = f'{bits:g}-bit boundary'
        boundary.append(test)
        correlations['subset'] = test['subset']
        boundary_correlations.append(correlations)

    sensitivity_specs = [
        ('normalized_margin_q05', 'negative'),
        ('normalized_margin_q25', 'negative'),
        ('median_normalized_margin', 'negative'),
        ('mean_normalized_margin', 'negative'),
        ('mean_raw_margin', 'negative'),
        ('mean_probability_gap', 'negative'),
        ('mean_normalized_entropy', 'positive'),
        ('oece', 'negative'),
    ]
    sensitivity_rows = []
    for index, (predictor, alternative) in enumerate(sensitivity_specs):
        sensitivity_frame = dataset_table[[
            'dataset_name', predictor, 'difficulty_score',
        ]].copy()
        sensitivity_frame['condition'] = 'aggregate'
        test, _ = dataset_block_permutation_test(
            sensitivity_frame, predictor, 'difficulty_score', alternative,
            permutations=permutations, seed=seed + 20 + index)
        sensitivity_rows.append(test)
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity['bh_fdr_p_value'] = _benjamini_hochberg(
        sensitivity['permutation_p_value'])

    dataset_table.to_csv(
        os.path.join(results_dir, 'dataset_difficulty.csv'), index=False)
    primary_rows.to_csv(
        os.path.join(results_dir, 'primary_condition_rows.csv'), index=False)
    logit_features.to_csv(
        os.path.join(results_dir, 'baseline_logit_features.csv'), index=False)
    primary_condition_correlations.to_csv(
        os.path.join(results_dir, 'primary_condition_correlations.csv'), index=False)
    influence.to_csv(
        os.path.join(results_dir, 'leave_one_dataset_out.csv'), index=False)
    nested_predictions.to_csv(
        os.path.join(results_dir, 'nested_lodo_predictions.csv'), index=False)
    margin_swap_correlations.to_csv(
        os.path.join(results_dir, 'margin_swap_correlations.csv'), index=False)
    swap_loss_correlations.to_csv(
        os.path.join(results_dir, 'swap_loss_correlations.csv'), index=False)
    pd.concat(boundary_correlations, ignore_index=True).to_csv(
        os.path.join(results_dir, 'boundary_correlations.csv'), index=False)
    sensitivity.to_csv(
        os.path.join(results_dir, 'sensitivity_predictors.csv'), index=False)
    _json_write(os.path.join(results_dir, 'concordance.json'), concordance)
    _json_write(os.path.join(results_dir, 'primary_test.json'), primary_test)
    _json_write(os.path.join(results_dir, 'nested_lodo_test.json'), nested)
    _json_write(os.path.join(results_dir, 'margin_swap_test.json'), margin_swap)
    _json_write(os.path.join(results_dir, 'swap_loss_test.json'), swap_loss)
    _json_write(os.path.join(results_dir, 'boundary_tests.json'), boundary)
    decision = {
        'promising_gate_passed': promising,
        'margin_mechanism_gate_passed': mechanism_supported,
        'negative_primary_conditions': negative_conditions,
        'negative_margin_swap_conditions': margin_swap_directions,
        'positive_swap_loss_conditions': swap_loss_directions,
    }
    _json_write(os.path.join(results_dir, 'decision.json'), decision)
    plots = _plot_outputs(dataset_table, primary_rows, results_dir, dpi)
    pd.DataFrame({'plot_file': plots}).to_csv(
        os.path.join(results_dir, 'plots_manifest.csv'), index=False)
    _write_report(
        os.path.join(results_dir, 'report.md'), dataset_table, concordance,
        primary_test, primary_condition_correlations, influence, nested,
        margin_swap, swap_loss, mechanism_supported, promising, boundary,
        sensitivity)
    write_run_config(results_dir, {
        'prepared_dir': prepared_dir,
        'manifest_file': manifest_file,
        'results_dir': results_dir,
        'serial': serial,
        'excluded_datasets': list(excluded_datasets),
        'expected_datasets': expected_datasets,
        'permutations': permutations,
        'model_permutations': model_permutations,
        'seed': seed,
        'chunk_size': chunk_size,
        'dpi': dpi,
    })
    return {
        **decision,
        'kendalls_w': concordance['kendalls_w'],
        'primary_rho': primary_test['median_spearman_rho'],
        'primary_p_value': primary_test['permutation_p_value'],
        'margin_swap_rho': margin_swap['median_spearman_rho'],
        'margin_swap_p_value': margin_swap['permutation_p_value'],
        'swap_loss_rho': swap_loss['median_spearman_rho'],
        'swap_loss_p_value': swap_loss['permutation_p_value'],
        'm0_mae': nested['m0_mae'],
        'm1_mae': nested['m1_mae'],
        'nested_p_value': nested['permutation_p_value'],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Screen dataset-model quantization difficulty from baseline logits.')
    parser.add_argument('--prepared-dir', default=PREPARED_DIR)
    parser.add_argument('--manifest-file', required=True)
    parser.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--serial', type=int, default=100004)
    parser.add_argument('--exclude-datasets', nargs='*', default=['inat17'])
    parser.add_argument('--expected-datasets', type=int, default=14)
    parser.add_argument('--permutations', type=int, default=10000)
    parser.add_argument('--model-permutations', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=100004)
    parser.add_argument('--chunk-size', type=int, default=1024)
    parser.add_argument('--dpi', type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    result = analyze_dataset_difficulty(
        args.prepared_dir, args.manifest_file, args.results_dir,
        serial=args.serial, excluded_datasets=args.exclude_datasets,
        expected_datasets=args.expected_datasets,
        permutations=args.permutations,
        model_permutations=args.model_permutations, seed=args.seed,
        chunk_size=args.chunk_size, dpi=args.dpi)
    print(f'kendalls_w={result["kendalls_w"]:.6f}')
    print(
        f'primary_rho={result["primary_rho"]:.6f} '
        f'primary_p={result["primary_p_value"]:.6f}')
    print(
        f'margin_swap_rho={result["margin_swap_rho"]:.6f} '
        f'margin_swap_p={result["margin_swap_p_value"]:.6f}')
    print(
        f'swap_loss_rho={result["swap_loss_rho"]:.6f} '
        f'swap_loss_p={result["swap_loss_p_value"]:.6f}')
    print(
        f'nested_m0_mae={result["m0_mae"]:.6f} '
        f'nested_m1_mae={result["m1_mae"]:.6f} '
        f'nested_p={result["nested_p_value"]:.6f}')
    print(f'promising_gate_passed={result["promising_gate_passed"]}')
    print(
        'margin_mechanism_gate_passed='
        f'{result["margin_mechanism_gate_passed"]}')


if __name__ == '__main__':
    main()
