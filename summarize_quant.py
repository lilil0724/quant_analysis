"""Prepare condition-keyed, baseline-matched HQQ accuracy summaries.

The analysis uses recorded HQQ settings and exact checkpoint identities rather
than inferring experimental conditions or FP32 baselines from serial numbers.
Serials remain optional input filters and output provenance only.
"""

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, 'results_all', 'quant')
DEFAULT_OUTPUT = os.path.join(DEFAULT_RESULTS_DIR, 'quant_summary.csv')
CKPT_PATTERN = re.compile(r'/(?:ckpt/)?(cal|ft|fz)(?:_ckpts)?/', re.IGNORECASE)
IDENTITY_COLUMNS = ['dataset_name', 'model_name', 'ckpt_path', 'seed_key']
CONTEXT_COLUMNS = ['dataset_name', 'model_name', 'ckpt_path', 'ckpt_kind']
SEED_CONDITION_COLUMNS = IDENTITY_COLUMNS + ['ckpt_kind', 'hqq_nbits', 'hqq_group_size']
CONDITION_COLUMNS = CONTEXT_COLUMNS + ['hqq_nbits', 'hqq_group_size']
HQQ_COLUMNS = ['hqq_top1', 'hqq_nbits', 'hqq_group_size']
BASELINE_COLUMNS = ['top1']


def is_true(values):
    return values.astype(str).str.lower().isin(('true', '1'))


def serial_values(values):
    numeric = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
    return sorted({int(value) for value in numeric})


def serials_text(values):
    return ', '.join(map(str, serial_values(values)))


def serials_count(values):
    return len(serial_values(values))


def combine_serial_text(values):
    serials = []
    for value in values:
        if pd.isna(value):
            continue
        serials.extend(str(value).split(','))
    return serials_text(serials)


def combined_serial_count(values):
    text = combine_serial_text(values)
    return serials_count(text.split(',')) if text else 0


def checkpoint_kind(paths):
    normalized = paths.astype(str).str.replace('\\', '/', regex=False)
    kinds = normalized.str.extract(CKPT_PATTERN)[0].str.lower()
    invalid = paths[kinds.isna()].unique()
    if len(invalid):
        examples = ', '.join(map(str, invalid[:3]))
        raise ValueError(f'Cannot classify ckpt_path as cal, ft, or fz: {examples}')
    return kinds


def rename_swin(model_name):
    """Adapted from BackbonesAnalysis/utils.py to normalize model names."""
    return {
        'swin_large_patch4_window12_384_in22k': 'swin_large_patch4_window7_224_in22k',
        'swin_base_patch4_window12_384_in22k': 'swin_base_patch4_window7_224_in22k',
        'swin_base_patch4_window12_384': 'swin_base_patch4_window7_224',
        'beitv2_base_patch16_224': 'beitv2_base_patch16_224_in22k',
    }.get(model_name, model_name)


def highlight_top_k(series, largest=True, k=5, color='lightgreen'):
    """Small local replacement for the historical utils.highlight_top_k helper."""
    if not pd.api.types.is_numeric_dtype(series):
        return [''] * len(series)
    selected = (series.nlargest(k) if largest else series.nsmallest(k)).dropna().index
    return [f'background-color: {color}' if index in selected else '' for index in series.index]


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize HQQ accuracy against exact checkpoint-and-seed FP32 baselines.')
    parser.add_argument(
        '--input-file', default=os.path.join(BASE_DIR, 'data', 'backbones_quant.csv'),
        help='full path of the downloaded HQQ CSV')
    parser.add_argument(
        '--output-file', default=DEFAULT_OUTPUT,
        help='full path of the primary summary CSV')
    parser.add_argument(
        '--results-dir', default=DEFAULT_RESULTS_DIR,
        help='directory for validation CSVs, XLSX workbooks, and run_config.json')
    parser.add_argument(
        '--serials', nargs='+', type=int,
        help='optional raw-run serials to retain; matching baselines must be included in the selected rows')
    parser.add_argument('--subset-datasets', nargs='+')
    parser.add_argument('--subset-models', nargs='+')
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


def apply_filters(frame, args):
    if args.subset_datasets:
        frame = frame[frame['dataset_name'].isin(args.subset_datasets)]
    if args.subset_models:
        frame = frame[frame['model_name'].isin(args.subset_models)]
    return frame


def require_columns(frame, columns):
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'Input CSV is missing required columns: {", ".join(missing)}')


def normalize_raw(raw, args):
    require_columns(raw, ['hqq', 'dataset_name', 'model_name', 'ckpt_path', *HQQ_COLUMNS, *BASELINE_COLUMNS])
    normalized = raw.copy()
    if 'serial' not in normalized:
        if args.serials:
            raise ValueError('--serials requires a serial column in the input CSV.')
        normalized['serial'] = pd.NA
    else:
        normalized['serial'] = pd.to_numeric(normalized['serial'], errors='raise')
        if args.serials:
            normalized = normalized[normalized['serial'].isin(args.serials)].copy()
    if 'seed' not in normalized:
        normalized['seed'] = pd.NA
    normalized['seed_key'] = normalized['seed'].where(normalized['seed'].notna(), '__missing_seed__').astype(str)
    normalized['model_name'] = normalized['model_name'].map(rename_swin)
    normalized = apply_filters(normalized, args)
    normalized['ckpt_kind'] = checkpoint_kind(normalized['ckpt_path'])
    return normalized


def aggregate_baselines(raw):
    baseline = raw[~is_true(raw['hqq'])].dropna(subset=BASELINE_COLUMNS).copy()
    baseline['top1'] = pd.to_numeric(baseline['top1'], errors='raise')
    grouped = baseline.groupby(IDENTITY_COLUMNS, as_index=False, dropna=False).agg(
        fp32_top1=('top1', 'median'),
        baseline_n_runs=('top1', 'size'),
        baseline_source_serials=('serial', serials_text),
        baseline_source_serial_count=('serial', serials_count),
    )
    grouped['baseline_source'] = 'raw_checkpoint_seed_baseline'
    return grouped


def aggregate_hqq_seed_conditions(raw):
    hqq = raw[is_true(raw['hqq'])].dropna(subset=HQQ_COLUMNS).copy()
    for column in HQQ_COLUMNS:
        hqq[column] = pd.to_numeric(hqq[column], errors='raise')
    grouped = hqq.groupby(SEED_CONDITION_COLUMNS, as_index=False, dropna=False).agg(
        hqq_top1=('hqq_top1', 'mean'),
        hqq_top1_median=('hqq_top1', 'median'),
        hqq_top1_std=('hqq_top1', 'std'),
        n_runs=('hqq_top1', 'size'),
        hqq_source_serials=('serial', serials_text),
        hqq_source_serial_count=('serial', serials_count),
    )
    return hqq, grouped


def aggregate_conditions(seed_conditions):
    grouped = seed_conditions.groupby(CONDITION_COLUMNS, as_index=False, dropna=False).agg(
        hqq_top1=('hqq_top1', 'mean'),
        hqq_top1_median=('hqq_top1', 'median'),
        hqq_top1_std=('hqq_top1', 'std'),
        fp32_top1=('fp32_top1', 'mean'),
        accuracy_ratio=('accuracy_ratio', 'mean'),
        n_runs=('n_runs', 'sum'),
        n_seeds=('seed_key', 'nunique'),
        matched_seed_count=('baseline_matched', 'sum'),
        baseline_matched=('baseline_matched', 'all'),
        hqq_source_serials=('hqq_source_serials', combine_serial_text),
        hqq_source_serial_count=('hqq_source_serials', combined_serial_count),
        baseline_source_serials=('baseline_source_serials', combine_serial_text),
        baseline_source_serial_count=('baseline_source_serials', combined_serial_count),
        baseline_n_runs=('baseline_n_runs', 'sum'),
    )
    grouped['unmatched_seed_count'] = grouped['n_seeds'] - grouped['matched_seed_count']
    grouped['baseline_source'] = 'raw_checkpoint_seed_baseline'
    unmatched = ~grouped['baseline_matched']
    grouped.loc[unmatched, ['fp32_top1', 'accuracy_ratio']] = np.nan
    return grouped


def prepare_summary(raw, args):
    normalized = normalize_raw(raw, args)
    hqq, seed_conditions = aggregate_hqq_seed_conditions(normalized)
    baseline = aggregate_baselines(normalized)
    seed_conditions = seed_conditions.merge(
        baseline, on=IDENTITY_COLUMNS, how='left', validate='many_to_one')
    seed_conditions['baseline_matched'] = seed_conditions['fp32_top1'].notna()
    seed_conditions['accuracy_ratio'] = seed_conditions['hqq_top1'] / seed_conditions['fp32_top1']
    cells = aggregate_conditions(seed_conditions)
    return normalized, hqq, cells


def completeness_table(cells):
    rows = []
    for context, group in cells.groupby(CONTEXT_COLUMNS, dropna=False):
        rows.append({
            **dict(zip(CONTEXT_COLUMNS, context)),
            'observed_hqq_conditions': len(group),
            'observed_nbits': ', '.join(map(str, sorted(group['hqq_nbits'].unique()))),
            'observed_group_sizes': ', '.join(map(str, sorted(group['hqq_group_size'].unique()))),
            'n_seeds': int(group['n_seeds'].sum()),
            'matched_conditions': int(group['baseline_matched'].sum()),
            'unmatched_conditions': int((~group['baseline_matched']).sum()),
            'unmatched_seed_count': int(group['unmatched_seed_count'].sum()),
            'raw_hqq_rows': int(group['n_runs'].sum()),
            'hqq_source_serials': combine_serial_text(group['hqq_source_serials']),
        })
    return pd.DataFrame(rows)


def configuration_quality(hqq):
    columns = [
        'hqq_exclude', 'hqq_offload_meta', 'hqq_compute_dtype',
        'hqq_quant_zero', 'hqq_quant_scale', 'image_size', 'batch_size', 'host',
    ]
    rows = []
    for column in columns:
        if column not in hqq:
            continue
        values = hqq[column].dropna().astype(str)
        rows.append({
            'column': column,
            'n_unique': values.nunique(),
            'values': ', '.join(sorted(values.unique())),
            'fixed': values.nunique() <= 1,
        })
    return pd.DataFrame(rows)


def write_dataset_workbooks(cells, results_dir):
    workbook_columns = [
        'dataset_name', 'model_name', 'ckpt_path', 'ckpt_kind', 'hqq_nbits',
        'hqq_group_size', 'hqq_top1', 'fp32_top1', 'accuracy_ratio',
        'hqq_top1_std', 'n_runs', 'n_seeds', 'hqq_source_serials',
        'baseline_source_serials', 'baseline_source', 'baseline_matched',
        'unmatched_seed_count',
    ]
    for dataset, frame in cells.groupby('dataset_name'):
        output = os.path.join(results_dir, f'quant_{dataset}.xlsx')
        selected = frame[[column for column in workbook_columns if column in frame]].sort_values(
            ['model_name', 'ckpt_kind', 'ckpt_path', 'hqq_nbits', 'hqq_group_size'])
        selected.style.apply(
            highlight_top_k, largest=True, k=5, color='lightgreen', axis=0,
            subset=['accuracy_ratio']).to_excel(output, index=False, engine='openpyxl')


def main():
    args = parse_args()
    args.results_dir = os.path.abspath(args.results_dir)
    args.output_file = os.path.abspath(args.output_file)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    write_run_config(os.path.join(args.results_dir, 'run_config.json'), args)
    raw = pd.read_csv(args.input_file)
    _, hqq, cells = prepare_summary(raw, args)
    if hqq.empty:
        raise ValueError('No valid HQQ rows remain after applying the selected filters.')
    completeness = completeness_table(cells)
    quality = configuration_quality(hqq)
    cells.to_csv(args.output_file, index=False)
    completeness.to_csv(os.path.join(args.results_dir, 'completeness.csv'), index=False)
    quality.to_csv(os.path.join(args.results_dir, 'configuration_quality.csv'), index=False)
    write_dataset_workbooks(cells, args.results_dir)
    unmatched = int((~cells['baseline_matched']).sum())
    print(f'Prepared {len(cells)} observed HQQ conditions; unmatched baseline conditions: {unmatched}')


if __name__ == '__main__':
    main()
