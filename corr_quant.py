"""Describe and model factors associated with HQQ accuracy retention."""

import argparse
import json
import os
import subprocess
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTCOME = 'accuracy_ratio'
CONTEXT_COLUMNS = ['series', 'dataset_name', 'model_name', 'ckpt_kind']
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, 'results_all', 'quant', 'corr')
DEFAULT_OUTPUT = os.path.join(DEFAULT_RESULTS_DIR, 'model_summary.csv')


def parse_args():
    parser = argparse.ArgumentParser(description='Model HQQ accuracy-ratio factors.')
    parser.add_argument(
        '--input-file', default=os.path.join(BASE_DIR, 'results_all', 'quant', 'quant_summary.csv'),
        help='full path of the summary CSV produced by summarize_quant.py')
    parser.add_argument(
        '--output-file', default=DEFAULT_OUTPUT,
        help='full path of the primary HC3 model summary CSV')
    parser.add_argument(
        '--results-dir', default=DEFAULT_RESULTS_DIR,
        help='directory for statistical tables, Markdown reports, and run_config.json')
    parser.add_argument(
        '--serials', nargs='+', type=int,
        help='explicit HQQ serial numbers to retain; default uses every summary row')
    parser.add_argument('--bootstrap-replicates', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20260801)
    return parser.parse_args()


def write_run_config(path, args):
    config = vars(args).copy()
    config['started_at'] = datetime.now(timezone.utc).isoformat()
    try:
        config['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        config['git_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        config['git_commit'] = None
        config['git_dirty'] = None
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2)


def bootstrap_interval(frame, value_column, replicates, rng):
    context_values = frame.groupby(CONTEXT_COLUMNS, as_index=False)[value_column].median()[value_column].to_numpy(float)
    if len(context_values) < 2:
        return np.nan, np.nan
    samples = rng.choice(context_values, size=(replicates, len(context_values)), replace=True)
    medians = np.median(samples, axis=1)
    return tuple(np.quantile(medians, [0.025, 0.975]))


def factor_summary(data, factor, replicates, rng):
    rows = []
    for value, frame in data.groupby(factor, dropna=False):
        low, high = bootstrap_interval(frame, OUTCOME, replicates, rng)
        context_values = frame.groupby(CONTEXT_COLUMNS, as_index=False)[OUTCOME].median()[OUTCOME]
        rows.append({
            'factor': factor,
            'level': value,
            'n_cells': len(frame),
            'n_contexts': frame[CONTEXT_COLUMNS].drop_duplicates().shape[0],
            'mean_accuracy_ratio': frame[OUTCOME].mean(),
            # Match the point estimate to the context-level bootstrap unit.
            'median_accuracy_ratio': context_values.median(),
            'bootstrap_ci_low': low,
            'bootstrap_ci_high': high,
        })
    return pd.DataFrame(rows)


def correlations(data):
    rows = []
    numeric_factors = {'hqq_nbits': data['hqq_nbits'], 'log2_group_size': np.log2(data['hqq_group_size'])}
    for name, values in numeric_factors.items():
        for method in ('pearson', 'spearman'):
            rows.append({'scope': 'global', 'factor': name, 'method': method, 'correlation': values.corr(data[OUTCOME], method=method)})
    for name, values in numeric_factors.items():
        values = pd.Series(values, index=data.index)
        grouped = []
        for _, frame in data.assign(_factor=values).groupby(CONTEXT_COLUMNS):
            if frame['_factor'].nunique() > 1 and frame[OUTCOME].nunique() > 1:
                grouped.append(frame['_factor'].corr(frame[OUTCOME], method='spearman'))
        rows.append({
            'scope': 'median_within_context', 'factor': name, 'method': 'spearman',
            'correlation': np.median(grouped) if grouped else np.nan,
        })
    return pd.DataFrame(rows)


def original_correlation_rows(data):
    """Preserve the legacy global and within-context correlation supplement."""
    rows = []
    factors = {'hqq_nbits': data['hqq_nbits'], 'log2_group_size': np.log2(data['hqq_group_size'])}
    for factor, values in factors.items():
        for method in ('pearson', 'spearman'):
            rows.append({'scope': 'global', 'factor': factor, 'method': method,
                         'context': 'all', 'n': len(data),
                         'correlation': values.corr(data[OUTCOME], method=method)})
    specs = [
        ('nbits_within_context', CONTEXT_COLUMNS + ['hqq_group_size'], 'hqq_nbits'),
        ('group_within_context', CONTEXT_COLUMNS + ['hqq_nbits'], 'log2_group_size'),
    ]
    for scope, keys, factor in specs:
        values = factors[factor]
        for context, frame in data.assign(_factor=values).groupby(keys):
            correlation = frame['_factor'].corr(frame[OUTCOME], method='spearman')
            rows.append({'scope': scope, 'factor': factor, 'method': 'spearman',
                         'context': ' | '.join(map(str, context)), 'n': len(frame),
                         'correlation': correlation})
    result = pd.DataFrame(rows)
    within = result[result['scope'].str.contains('_within_', na=False)].groupby(['scope', 'factor', 'method'])['correlation'].median().reset_index()
    within['scope'] = 'within_summary:' + within['scope']
    within['context'] = 'summary'
    within['n'] = np.nan
    return pd.concat([result, within[result.columns]], ignore_index=True)


def legacy_design_matrix(data, levels):
    records = [('intercept', np.ones(len(data)), 'intercept')]
    categorical = {}
    if levels:
        for column, group in [('hqq_nbits', 'nbits'), ('hqq_group_size', 'group_size')]:
            values = sorted(data[column].astype(str).unique())
            categorical[group] = []
            for value in values[1:]:
                vector = (data[column].astype(str) == value).astype(float).to_numpy()
                records.append((f'{column}[{value}]', vector, group))
                categorical[group].append((f'{column}[{value}]', vector))
    else:
        categorical['nbits'] = [('nbits', data['hqq_nbits'].to_numpy(float))]
        categorical['group_size'] = [('log2_group_size', np.log2(data['hqq_group_size']).to_numpy(float))]
        records.extend([(name, values, group) for group in ('nbits', 'group_size') for name, values in categorical[group]])
    for column, group in [('dataset_name', 'dataset'), ('model_name', 'model'), ('ckpt_kind', 'ckpt_kind')]:
        categorical[group] = []
        for value in sorted(data[column].astype(str).unique())[1:]:
            vector = (data[column].astype(str) == value).astype(float).to_numpy()
            records.append((f'{column}[{value}]', vector, group))
            categorical[group].append((f'{column}[{value}]', vector))
    for right, label in [('group_size', 'nbits:group_size'), ('dataset', 'nbits:dataset'), ('model', 'nbits:model'), ('ckpt_kind', 'nbits:ckpt_kind')]:
        for left_name, left_values in categorical['nbits']:
            for right_name, right_values in categorical[right]:
                records.append((f'{left_name}:{right_name}', left_values * right_values, label))
    return np.column_stack([values for _, values, _ in records]), [name for name, _, _ in records], [group for _, _, group in records]


def legacy_factor_model(data, levels):
    matrix, names, groups = legacy_design_matrix(data, levels)
    outcome = data[OUTCOME].to_numpy(float)
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, outcome, rcond=None)
    fitted = matrix @ coefficients
    sst = float(np.dot(outcome - outcome.mean(), outcome - outcome.mean()))
    sse = float(np.dot(outcome - fitted, outcome - fitted))
    r2 = 1 - sse / sst if sst else np.nan
    rows = []
    for group in sorted(set(groups) - {'intercept'}):
        keep = np.array([value != group for value in groups])
        reduced_coef, _, _, _ = np.linalg.lstsq(matrix[:, keep], outcome, rcond=None)
        reduced_sse = float(np.dot(outcome - matrix[:, keep] @ reduced_coef, outcome - matrix[:, keep] @ reduced_coef))
        rows.append({'term_group': group, 'full_r2': r2,
                     'reduced_r2': 1 - reduced_sse / sst if sst else np.nan,
                     'delta_r2': r2 - (1 - reduced_sse / sst if sst else np.nan)})
    return pd.DataFrame({'column': names, 'term_group': groups, 'coefficient': coefficients}), pd.DataFrame(rows), {
        'model': 'levels' if levels else 'trend', 'n_cells': int(len(data)),
        'n_columns': int(matrix.shape[1]), 'rank': int(rank), 'r2': float(r2),
        'condition_number': float(np.linalg.cond(matrix)),
    }


def fit_models(data):
    formula = (
        'accuracy_ratio ~ C(hqq_nbits) * C(hqq_group_size) + '
        'C(ckpt_kind) + C(dataset_name) + C(model_name)'
    )
    fixed = smf.ols(formula, data=data).fit(cov_type='HC3')
    fixed_table = pd.DataFrame({
        'term': fixed.params.index,
        'coefficient': fixed.params.values,
        'std_error': fixed.bse.values,
        'ci_low': fixed.conf_int().iloc[:, 0].values,
        'ci_high': fixed.conf_int().iloc[:, 1].values,
        'p_value': fixed.pvalues.values,
    })
    fixed_metadata = {
        'formula': formula,
        'n_cells': int(fixed.nobs),
        'r_squared': float(fixed.rsquared),
        'adjusted_r_squared': float(fixed.rsquared_adj),
        'condition_number': float(fixed.condition_number),
        'covariance': 'HC3 heteroskedasticity robust',
    }

    mixed_metadata = {'status': 'not_run'}
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter('always')
            mixed = smf.mixedlm(
                formula, data=data, groups=data[CONTEXT_COLUMNS].astype(str).agg('|'.join, axis=1)
            ).fit(reml=False, method='lbfgs')
        random_variance = float(mixed.cov_re.iloc[0, 0])
        warning_text = ' | '.join(str(item.message) for item in caught_warnings)
        status = 'fit' if mixed.converged and random_variance > 1e-10 else 'singular_or_unconverged'
        mixed_metadata = {
            'status': status, 'converged': bool(mixed.converged), 'random_intercept_variance': random_variance,
            'aic': float(mixed.aic), 'bic': float(mixed.bic), 'log_likelihood': float(mixed.llf),
            'warning': warning_text or None,
        }
    except Exception as error:
        mixed_metadata = {'status': 'failed', 'reason': str(error)}
    return fixed_table, fixed_metadata, mixed_metadata


def main():
    args = parse_args()
    args.results_dir = os.path.abspath(args.results_dir)
    args.output_file = os.path.abspath(args.output_file)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    write_run_config(os.path.join(args.results_dir, 'run_config.json'), args)
    data = pd.read_csv(args.input_file)
    required = [OUTCOME, 'baseline_matched', 'serial', 'hqq_nbits', 'hqq_group_size', *CONTEXT_COLUMNS]
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f'Summary is missing columns: {", ".join(missing)}')
    if args.serials:
        data = data[data['serial'].isin(args.serials)].copy()
    data = data[data['baseline_matched']].dropna(subset=[OUTCOME]).copy()
    if data.empty:
        raise ValueError('No baseline-matched accuracy cells remain.')
    data.to_csv(os.path.join(args.results_dir, 'prepared_accuracy_cells.csv'), index=False)

    rng = np.random.default_rng(args.seed)
    summaries = pd.concat([
        factor_summary(data, factor, args.bootstrap_replicates, rng)
        for factor in ('hqq_nbits', 'hqq_group_size', 'ckpt_kind', 'dataset_name', 'model_name')
    ], ignore_index=True)
    summaries.to_csv(os.path.join(args.results_dir, 'factor_summary.csv'), index=False)
    correlations(data).to_csv(os.path.join(args.results_dir, 'correlations.csv'), index=False)
    legacy_correlations = original_correlation_rows(data)
    legacy_correlations.to_csv(os.path.join(args.results_dir, 'correlations_accuracy.csv'), index=False)
    trend_effects, trend_ablation, trend_metadata = legacy_factor_model(data, levels=False)
    level_effects, level_ablation, level_metadata = legacy_factor_model(data, levels=True)
    trend_effects.to_csv(os.path.join(args.results_dir, 'factor_effects_accuracy_trend.csv'), index=False)
    level_effects.to_csv(os.path.join(args.results_dir, 'factor_effects_accuracy_levels.csv'), index=False)
    trend_ablation.to_csv(os.path.join(args.results_dir, 'factor_ablation_accuracy_trend.csv'), index=False)
    level_ablation.to_csv(os.path.join(args.results_dir, 'factor_ablation_accuracy_levels.csv'), index=False)
    level_ablation.to_csv(os.path.join(args.results_dir, 'factor_ablation_accuracy.csv'), index=False)
    with open(os.path.join(args.results_dir, 'model_fit_accuracy_trend.json'), 'w', encoding='utf-8') as handle:
        json.dump(trend_metadata, handle, indent=2)
    with open(os.path.join(args.results_dir, 'model_fit_accuracy_levels.json'), 'w', encoding='utf-8') as handle:
        json.dump(level_metadata, handle, indent=2)
    fixed_table, fixed_metadata, mixed_metadata = fit_models(data)
    fixed_table.to_csv(args.output_file, index=False)
    with open(os.path.join(args.results_dir, 'model_metadata.json'), 'w', encoding='utf-8') as handle:
        json.dump({'fixed_effects': fixed_metadata, 'mixed_effects': mixed_metadata}, handle, indent=2)
    quality_lines = ['# Data quality', '', f'- Prepared accuracy cells: {len(data)}',
                     f'- Baseline-matched cells: {int(data["baseline_matched"].sum())}',
                     f'- Contexts: {data[CONTEXT_COLUMNS].drop_duplicates().shape[0]}']
    with open(os.path.join(args.results_dir, 'data_quality.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(quality_lines) + '\n')
    mixed_status = mixed_metadata['status']
    mixed_detail = mixed_metadata.get('reason') or mixed_metadata.get('warning') or 'No diagnostic warning.'
    report_lines = ['# Quant accuracy factor report', '', f'- Prepared cells: {len(data)}', '',
                    '## Factor contribution ranking (delta R2 ablation)', '',
                    level_ablation.sort_values('delta_r2', ascending=False).to_string(index=False), '',
                    '## Spearman rank correlation supplement', '',
                    legacy_correlations[legacy_correlations['method'].eq('spearman')].tail(2).to_string(index=False), '',
                    '## Additional inference', '',
                    'HC3 fixed-effects estimates and a mixed-effects diagnostic are in `model_summary.csv` and `model_metadata.json`.',
                    f'Mixed-effects diagnostic status: `{mixed_status}`.',
                    f'Diagnostic detail: {mixed_detail}']
    with open(os.path.join(args.results_dir, 'report.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(report_lines) + '\n')
    print(f'Modelled {len(data)} baseline-matched cells in {args.results_dir}')


if __name__ == '__main__':
    main()
