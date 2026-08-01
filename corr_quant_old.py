"""Descriptive factor analysis for pre-cleanup HQQ WandB CSVs."""

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


CKPT_PATTERN = re.compile(r'/(cal|ft|fz)_ckpts/')
CONTEXT = ['dataset_name', 'model_name', 'ckpt_kind']
OUTCOMES = {
    'accuracy': 'accuracy_ratio',
    'vram': 'vram_reduction_pct',
    'params': 'params_reduction_pct',
    'time': 'time_delta_pct',
}
HQQ_METRICS = ['hqq_top1', 'hqq_vram_gb', 'hqq_params_m', 'time_total_s']
FP32_METRICS = ['top1', 'vram_gb', 'params_m', 'time_total_s']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyse pre-cleanup HQQ factor effects and resource trade-offs.')
    parser.add_argument('--input-file', default=os.path.join('data', 'backbones_quant.csv'))
    parser.add_argument(
        '--baseline-overrides',
        default=os.path.join('data', 'quant_fp32_baseline_overrides.csv'))
    parser.add_argument('--results-dir', default=os.path.join(BASE_DIR, 'results_all', 'quant_old', 'corr'))
    parser.add_argument('--output-name',
                        help='output folder name under results_all')
    parser.add_argument('--subset-datasets', nargs='+')
    parser.add_argument('--subset-models', nargs='+')
    parser.add_argument('--subset-ckpt-kinds', nargs='+', default=['ft', 'fz', 'cal'])
    parser.add_argument('--metrics', nargs='+', choices=tuple(OUTCOMES), default=list(OUTCOMES))
    parser.add_argument('--save-format', choices=('png', 'pdf'), default='png')
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def is_true(series):
    return series.astype(str).str.lower().isin(('true', '1'))


def add_ckpt_kind(df):
    kinds = df['ckpt_path'].astype(str).str.extract(CKPT_PATTERN)[0]
    invalid = df.loc[kinds.isna(), 'ckpt_path'].unique()
    if len(invalid):
        raise ValueError('Cannot classify ckpt_path: ' + ', '.join(map(str, invalid[:3])))
    result = df.copy()
    result['ckpt_kind'] = kinds
    return result


def apply_filters(df, args):
    if args.subset_datasets:
        df = df[df['dataset_name'].isin(args.subset_datasets)]
    if args.subset_models:
        df = df[df['model_name'].isin(args.subset_models)]
    return df


def require_columns(df, columns, label):
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f'{label} is missing columns: {", ".join(missing)}')


def load_baselines(raw, args):
    fp32 = raw[(~is_true(raw['hqq'])) & raw['serial'].eq(4999)].copy()
    fp32 = apply_filters(fp32, args)
    fp32 = add_ckpt_kind(fp32)
    fp32 = fp32.dropna(subset=FP32_METRICS)
    raw_baselines = fp32.groupby(CONTEXT, as_index=False).agg(
        fp32_top1=('top1', 'median'),
        fp32_vram_gb=('vram_gb', 'median'),
        fp32_params_m=('params_m', 'median'),
        fp32_time_total_s=('time_total_s', 'median'),
    )
    raw_baselines['baseline_source'] = 'raw_4999'

    if not os.path.exists(args.baseline_overrides):
        raise FileNotFoundError(f'Baseline override file not found: {args.baseline_overrides}')
    overrides = pd.read_csv(args.baseline_overrides)
    require_columns(overrides, CONTEXT + ['serial', *FP32_METRICS, 'source'], 'baseline overrides')
    overrides = apply_filters(overrides, args)
    overrides = overrides[overrides['ckpt_kind'].isin(args.subset_ckpt_kinds)].copy()
    if overrides.duplicated(CONTEXT).any():
        raise ValueError('Baseline overrides contain duplicate context keys.')
    overrides = overrides.rename(columns={
        'top1': 'fp32_top1', 'vram_gb': 'fp32_vram_gb',
        'params_m': 'fp32_params_m', 'time_total_s': 'fp32_time_total_s',
        'source': 'baseline_source',
    })
    overrides = overrides[CONTEXT + [
        'fp32_top1', 'fp32_vram_gb', 'fp32_params_m', 'fp32_time_total_s',
        'baseline_source',
    ]]

    combined = pd.concat([raw_baselines, overrides], ignore_index=True)
    # Overrides are appended last and therefore take precedence per context.
    return combined.drop_duplicates(CONTEXT, keep='last')


def prepare_cells(raw, baselines, args):
    hqq = raw[is_true(raw['hqq'])].copy()
    hqq = apply_filters(hqq, args)
    hqq = add_ckpt_kind(hqq)
    hqq = hqq[hqq['ckpt_kind'].isin(args.subset_ckpt_kinds)]
    complete = hqq.dropna(subset=HQQ_METRICS).copy()
    keys = ['dataset_name', 'model_name', 'ckpt_path', 'ckpt_kind', 'hqq_nbits', 'hqq_group_size']
    numeric = HQQ_METRICS
    cells = complete.groupby(keys, as_index=False).agg(
        **{column: (column, 'median') for column in numeric},
        n_runs=('serial', 'size'),
        time_min_s=('time_total_s', 'min'),
        time_max_s=('time_total_s', 'max'),
    )
    cells['time_range_s'] = cells['time_max_s'] - cells['time_min_s']
    cells['log2_group_size'] = np.log2(cells['hqq_group_size'])
    cells = cells.merge(baselines, on=CONTEXT, how='left')
    cells['baseline_matched'] = cells['fp32_top1'].notna()
    for denominator in ['fp32_top1', 'fp32_vram_gb', 'fp32_params_m', 'fp32_time_total_s']:
        if (cells[denominator].dropna() == 0).any():
            raise ValueError(f'Baseline contains zero denominator: {denominator}')
    cells['accuracy_ratio'] = cells['hqq_top1'] / cells['fp32_top1']
    cells['vram_reduction_pct'] = 100 * (cells['fp32_vram_gb'] - cells['hqq_vram_gb']) / cells['fp32_vram_gb']
    cells['params_reduction_pct'] = 100 * (cells['fp32_params_m'] - cells['hqq_params_m']) / cells['fp32_params_m']
    cells['time_delta_pct'] = 100 * (cells['time_total_s'] - cells['fp32_time_total_s']) / cells['fp32_time_total_s']
    return hqq, complete, cells


def correlation_rows(df, outcome):
    def correlation(left, right, method):
        pair = pd.DataFrame({'left': left, 'right': right}).dropna()
        if len(pair) < 2 or pair['left'].nunique() < 2 or pair['right'].nunique() < 2:
            return np.nan
        return pair['left'].corr(pair['right'], method=method)

    rows = []
    for factor in ['hqq_nbits', 'log2_group_size']:
        for method in ['pearson', 'spearman']:
            rows.append({
                'scope': 'global', 'factor': factor, 'method': method,
                'context': 'all', 'n': len(df), 'correlation': correlation(df[factor], df[outcome], method),
            })
    specs = [
        ('nbits_within_context', ['dataset_name', 'model_name', 'ckpt_kind', 'hqq_group_size'], 'hqq_nbits'),
        ('group_within_context', ['dataset_name', 'model_name', 'ckpt_kind', 'hqq_nbits'], 'log2_group_size'),
    ]
    for scope, keys, factor in specs:
        for values, group in df.groupby(keys):
            label = ' | '.join(map(str, values if isinstance(values, tuple) else (values,)))
            for method in ['pearson', 'spearman']:
                rows.append({
                    'scope': scope, 'factor': factor, 'method': method,
                    'context': label, 'n': len(group),
                    'correlation': correlation(group[factor], group[outcome], method),
                })
    result = pd.DataFrame(rows)
    within = result[result['scope'] != 'global'].groupby(['scope', 'factor', 'method'])['correlation'].agg(
        ['median', 'min', 'max', 'count']).reset_index()
    within['scope'] = 'within_summary:' + within['scope']
    within['context'] = 'summary'
    within['n'] = within['count']
    within['correlation'] = within['median']
    return pd.concat([result, within[result.columns]], ignore_index=True)


def categorical_columns(df, column, group):
    values = sorted(df[column].astype(str).unique())
    reference = values[0]
    records = []
    for value in values[1:]:
        records.append((f'{column}[{value}]', (df[column].astype(str) == value).astype(float).to_numpy(), group))
    return records, reference


def design_matrix(df, kind):
    records = [('intercept', np.ones(len(df)), 'intercept')]
    categorical = {}
    if kind == 'trend':
        nbits = df['hqq_nbits'].to_numpy(float)
        group_size = df['log2_group_size'].to_numpy(float)
        records.extend([('nbits', nbits, 'nbits'), ('log2_group_size', group_size, 'group_size')])
        categorical['nbits'] = [('nbits', nbits)]
        categorical['group_size'] = [('log2_group_size', group_size)]
    else:
        n_records, _ = categorical_columns(df, 'hqq_nbits', 'nbits')
        g_records, _ = categorical_columns(df, 'hqq_group_size', 'group_size')
        records.extend(n_records + g_records)
        categorical['nbits'] = [(name, values) for name, values, _ in n_records]
        categorical['group_size'] = [(name, values) for name, values, _ in g_records]

    for column, group in [('dataset_name', 'dataset'), ('model_name', 'model'), ('ckpt_kind', 'ckpt_kind')]:
        cat_records, _ = categorical_columns(df, column, group)
        records.extend(cat_records)
        categorical[group] = [(name, values) for name, values, _ in cat_records]

    interactions = [('group_size', 'nbits:group_size'), ('dataset', 'nbits:dataset'),
                    ('model', 'nbits:model'), ('ckpt_kind', 'nbits:ckpt_kind')]
    for right_key, term in interactions:
        for left_name, left_values in categorical['nbits']:
            for right_name, right_values in categorical[right_key]:
                records.append((f'{left_name}:{right_name}', left_values * right_values, term))

    names = [name for name, _, _ in records]
    groups = [group for _, _, group in records]
    matrix = np.column_stack([values for _, values, _ in records])
    return matrix, names, groups


def fit_model(df, outcome, kind):
    data = df.dropna(subset=[outcome]).copy()
    matrix, names, groups = design_matrix(data, kind)
    y = data[outcome].to_numpy(float)
    coefficients, _, rank, singular = np.linalg.lstsq(matrix, y, rcond=None)
    if rank < matrix.shape[1]:
        raise ValueError(f'{kind} design matrix is rank deficient: {rank}/{matrix.shape[1]}')
    prediction = matrix @ coefficients
    residual = y - prediction
    sse = float(np.dot(residual, residual))
    sst = float(np.dot(y - y.mean(), y - y.mean()))
    r2 = 1 - sse / sst if sst else np.nan
    n, p = matrix.shape
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p) if n > p and np.isfinite(r2) else np.nan
    pseudoinverse = np.linalg.pinv(matrix)
    leverage = np.einsum('ij,ji->i', matrix, pseudoinverse)
    loo_errors = residual / np.maximum(1 - leverage, np.finfo(float).eps)
    result = pd.DataFrame({'column': names, 'term_group': groups, 'coefficient': coefficients})
    ablation = []
    for group in sorted(set(groups) - {'intercept'}):
        keep = np.array([item != group for item in groups])
        beta, _, _, _ = np.linalg.lstsq(matrix[:, keep], y, rcond=None)
        reduced = matrix[:, keep] @ beta
        reduced_sse = float(np.dot(y - reduced, y - reduced))
        reduced_r2 = 1 - reduced_sse / sst if sst else np.nan
        ablation.append({'term_group': group, 'full_r2': r2, 'reduced_r2': reduced_r2, 'delta_r2': r2 - reduced_r2})
    metadata = {
        'outcome': outcome, 'model': kind, 'n_cells': int(n), 'n_columns': int(p),
        'rank': int(rank), 'condition_number': float(np.linalg.cond(matrix)),
        'r2': float(r2), 'adjusted_r2': float(adjusted_r2),
        'loo_mae': float(np.mean(np.abs(loo_errors))),
        'loo_rmse': float(np.sqrt(np.mean(np.square(loo_errors)))),
        'references': {
            'nbits': str(sorted(data['hqq_nbits'].unique())[0]),
            'group_size': str(sorted(data['hqq_group_size'].unique())[0]),
            'dataset': sorted(data['dataset_name'].astype(str).unique())[0],
            'model': sorted(data['model_name'].astype(str).unique())[0],
            'ckpt_kind': sorted(data['ckpt_kind'].astype(str).unique())[0],
        },
    }
    return result, pd.DataFrame(ablation), metadata


def write_quality(path, raw, hqq, complete, cells, baselines):
    duplicate = cells[cells['n_runs'] > 1]
    lines = [
        '# Data quality', '',
        f'- Raw rows: {len(raw)}',
        f'- Raw HQQ rows: {len(hqq)}',
        f'- Complete HQQ rows: {len(complete)}',
        f'- Prepared factor cells: {len(cells)}',
        f'- Duplicate prepared cells: {len(duplicate)}',
        f'- Baseline contexts: {len(baselines)}',
        f'- Baseline-matched cells: {int(cells["baseline_matched"].sum())}',
        f'- Unmatched cells: {int((~cells["baseline_matched"]).sum())}',
        f'- Max duplicate time range (s): {duplicate["time_range_s"].max() if len(duplicate) else 0:.2f}', '',
        '## Baseline sources', '',
    ]
    lines.extend(f'- `{row.baseline_source}`: {row.count}' for row in baselines['baseline_source'].value_counts().reset_index(name='count').itertuples(index=False))
    lines.extend(['', '## Excluded fixed config', '',
                  '- hqq_quant_zero=True, hqq_quant_scale=True, hqq_offload_meta=False,',
                  '  hqq_compute_dtype=float16, hqq_exclude=head, image_size=224,',
                  '  batch_size=64, host=server-3090.', '',
                  'Time results are exploratory: repeated same-seed runs were collapsed by median.'])
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def save_plot(fig, path, args):
    fig.savefig(path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)


def plot_outputs(cells, ablations, args):
    plot_dir = os.path.join(args.results_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    sns.set_theme(style='whitegrid')
    for ckpt in sorted(cells['ckpt_kind'].unique()):
        for key in ['accuracy', 'vram']:
            outcome = OUTCOMES[key]
            pivot = cells[cells['ckpt_kind'] == ckpt].pivot_table(
                index='hqq_group_size', columns='hqq_nbits', values=outcome, aggfunc='median')
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=1 if key == 'accuracy' else None,
                center=None if key == 'accuracy' else 0, ax=ax)
            ax.set_title(f'Median {outcome}: {ckpt} checkpoints')
            ax.set_xlabel('nbits')
            ax.set_ylabel('group size')
            save_plot(fig, os.path.join(plot_dir, f'{key}_heatmap_{ckpt}.{args.save_format}'), args)

    for column, label, filename_prefix in [
            ('model_name', 'model', 'accuracy_heatmap_model'),
            ('dataset_name', 'dataset', 'accuracy_heatmap_dataset')]:
        for value in sorted(cells[column].unique()):
            pivot = cells[cells[column] == value].pivot_table(
                index='hqq_group_size', columns='hqq_nbits',
                values=OUTCOMES['accuracy'], aggfunc='median')
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=1, ax=ax)
            ax.set_title(f'Median accuracy_ratio: {label} {value}')
            ax.set_xlabel('nbits')
            ax.set_ylabel('group size')
            save_plot(
                fig,
                os.path.join(plot_dir, f'{filename_prefix}_{value}.{args.save_format}'),
                args)
    for key in args.metrics:
        outcome = OUTCOMES[key]
        fig, ax = plt.subplots(figsize=(7, 4))
        for group_size, group in cells.groupby('hqq_group_size'):
            summary = group.groupby('hqq_nbits')[outcome].median().sort_index()
            ax.plot(summary.index, summary.values, marker='o', label=f'group {group_size}')
        hline = 1 if key == 'accuracy' else 0
        ax.axhline(hline, color='black', linewidth=.7)
        ax.set(title=f'{outcome} by nbits and group size', xlabel='nbits', ylabel=outcome)
        ax.legend()
        save_plot(fig, os.path.join(plot_dir, f'interaction_nbits_group_{key}.{args.save_format}'), args)
        fig, ax = plt.subplots(figsize=(7, 4))
        for ckpt, group in cells.groupby('ckpt_kind'):
            summary = group.groupby('hqq_nbits')[outcome].median().sort_index()
            ax.plot(summary.index, summary.values, marker='o', label=ckpt)
        ax.axhline(hline, color='black', linewidth=.7)
        ax.set(title=f'{outcome} by nbits and checkpoint', xlabel='nbits', ylabel=outcome)
        ax.legend(title='checkpoint')
        save_plot(fig, os.path.join(plot_dir, f'interaction_nbits_context_{key}.{args.save_format}'), args)
        trend = ablations[key]['levels'].sort_values('delta_r2', ascending=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(trend['term_group'], trend['delta_r2'])
        ax.set(title=f'Descriptive factor contribution: {outcome}', xlabel='Ablation delta R2')
        save_plot(fig, os.path.join(plot_dir, f'factor_contribution_{key}.{args.save_format}'), args)


def write_report(path, cells, baselines, correlation_summaries, ablation_summaries):
    lines = [
        '# Quant factor report', '',
        f'- Prepared cells: {len(cells)}',
        f'- Baseline-matched cells: {int(cells["baseline_matched"].sum())}',
        f'- Baseline contexts: {len(baselines)}', '',
        '## Factor contribution ranking (delta R2 ablation)', '',
        'Bars in `factor_contribution_<metric>.png` are delta R2 from the ',
        'levels (categorical) model: each nbits and group_size level is a ',
        'separate dummy column. This avoids forcing a linear relationship and ',
        'captures non-linear patterns like "low bit collapses, high bit ',
        'plateaus".  Delta R2 = how much the full-model R2 drops when that ',
        'term group is removed.  This is a ranking, not an additive allocation.',
        '',
    ]
    for key, ablation in sorted(ablation_summaries.items()):
        lines.append(f'### {OUTCOMES[key]}')
        top = ablation.sort_values('delta_r2', ascending=False).head(6)
        lines.append(top.to_string(index=False))
        lines.append('')

    lines.extend([
        '## Spearman rank correlation supplement', '',
        'Pearson r measures linear association; Spearman rho measures monotonic ',
        'association (is the rank order preserved?). When rho substantially exceeds ',
        'r, the relationship exists but is non-linear — a common pattern with ',
        'nbits-driven accuracy drop. Spearman is then the more reliable first-pass ',
        'summary of association strength.',
        '',
        'Below are median within-context Spearman correlations: each value fixes ',
        'dataset, model, checkpoint kind, and (for nbits) group size, or ',
        '(for group size) nbits.  This controls confounding from task difficulty ',
        'and isolates the monotonic association between the factor and the outcome.',
        '',
        'All values are dimensionless (-1 to 1).',
        '',
    ])
    for key, corr_df in sorted(correlation_summaries.items()):
        lines.append(f'### {OUTCOMES[key]}')
        lines.append(corr_df.to_string(index=False))
        lines.append('')

    lines.extend([
        '## Limitations', '',
        '- This is a single-seed complete factor grid. Results are descriptive effect decompositions, not significance tests.',
        '- Time is exploratory because repeated same-seed rows have variable elapsed time.',
        '- Metrics are host-specific (`server-3090`) and fixed to the observed HQQ configuration.',
    ])
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    if args.output_name:
        args.results_dir = os.path.join(BASE_DIR, 'results_all', args.output_name, 'corr')
    os.makedirs(args.results_dir, exist_ok=True)
    config = vars(args).copy()
    config['started_at'] = datetime.now(timezone.utc).isoformat()
    try:
        config['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip()
        config['git_dirty'] = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        config['git_commit'] = None
        config['git_dirty'] = None
    with open(os.path.join(args.results_dir, 'run_config.json'), 'w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2, default=str)
    raw = pd.read_csv(args.input_file)
    require_columns(raw, ['serial', 'hqq', 'dataset_name', 'model_name', 'ckpt_path', 'hqq_nbits', 'hqq_group_size', *HQQ_METRICS, *FP32_METRICS], 'input CSV')
    baselines = load_baselines(raw, args)
    hqq, complete, cells = prepare_cells(raw, baselines, args)
    cells.to_csv(os.path.join(args.results_dir, 'prepared_quant_cells.csv'), index=False)
    baselines.to_csv(os.path.join(args.results_dir, 'baseline_contexts.csv'), index=False)
    write_quality(os.path.join(args.results_dir, 'data_quality.md'), raw, hqq, complete, cells, baselines)

    ablations = {}
    corr_summaries = {}
    for key in args.metrics:
        outcome = OUTCOMES[key]
        data = cells.dropna(subset=[outcome])
        correlations = correlation_rows(data, outcome)
        correlations.to_csv(os.path.join(args.results_dir, f'correlations_{key}.csv'), index=False)
        summary = correlations[correlations['scope'].str.startswith('within_summary:', na=False)].copy()
        summary = summary[summary['method'].eq('spearman')]
        if not summary.empty:
            corr_summaries[key] = summary[['scope', 'factor', 'correlation']]
        ablations[key] = {}
        for kind in ['trend', 'levels']:
            effects, ablation, metadata = fit_model(data, outcome, kind)
            effects.to_csv(os.path.join(args.results_dir, f'factor_effects_{key}_{kind}.csv'), index=False)
            ablation.to_csv(os.path.join(args.results_dir, f'factor_ablation_{key}_{kind}.csv'), index=False)
            if kind == 'levels':
                ablation.to_csv(os.path.join(args.results_dir, f'factor_ablation_{key}.csv'), index=False)
            ablations[key][kind] = ablation
            with open(os.path.join(args.results_dir, f'model_fit_{key}_{kind}.json'), 'w', encoding='utf-8') as handle:
                json.dump(metadata, handle, indent=2)

    plot_outputs(cells, ablations, args)
    write_report(os.path.join(args.results_dir, 'report.md'), cells, baselines, corr_summaries, {k: v['levels'] for k, v in ablations.items()})
    print(f'Prepared {len(cells)} cells and wrote analysis to {args.results_dir}')


if __name__ == '__main__':
    main()
