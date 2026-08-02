"""Prepare baseline-matched accuracy summaries from legacy HQQ WandB exports.

This replaces summarize_quant_old.py. The small helper functions below are
adapted from the historical BackbonesAnalysis utilities so this repository no
longer depends on that sibling project or its unrelated analysis code.
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
CKPT_PATTERN = re.compile(r'/(cal|ft|fz)_ckpts/')
CONTEXT_COLUMNS = ['series', 'dataset_name', 'model_name', 'ckpt_kind']
HQQ_COLUMNS = ['hqq_top1', 'hqq_nbits', 'hqq_group_size']
BASELINE_COLUMNS = ['top1']


def scan_serials(series):
    return [series + nbits * 100 + group for nbits in range(8) for group in range(4)]


EXPECTED_SERIALS = {3000: scan_serials(3000), 4000: scan_serials(4000)}
BASELINE_SERIALS = {3000: 3999, 4000: 4999}


def is_true(values):
    return values.astype(str).str.lower().isin(('true', '1'))


def series_from_serial(values):
    numeric = pd.to_numeric(values, errors='coerce')
    return np.select([numeric.between(3000, 3999), numeric.between(4000, 4999)], [3000, 4000], default=np.nan)


def checkpoint_kind(paths):
    kinds = paths.astype(str).str.extract(CKPT_PATTERN)[0]
    invalid = paths[kinds.isna()].unique()
    if len(invalid):
        examples = ', '.join(map(str, invalid[:3]))
        raise ValueError(f'Cannot classify ckpt_path as cal, ft, or fz: {examples}')
    return kinds


def rename_swin(model_name):
    """Adapted from BackbonesAnalysis/utils.py to normalize Swin model names."""
    return {
        'swin_large_patch4_window12_384_in22k': 'swin_large_patch4_window7_224_in22k',
        'swin_base_patch4_window12_384_in22k': 'swin_base_patch4_window7_224_in22k',
        'swin_base_patch4_window12_384': 'swin_base_patch4_window7_224',
    }.get(model_name, model_name)


def highlight_top_k(series, largest=True, k=5, color='lightgreen'):
    """Small local replacement for the historical utils.highlight_top_k helper."""
    if not pd.api.types.is_numeric_dtype(series):
        return [''] * len(series)
    selected = (series.nlargest(k) if largest else series.nsmallest(k)).dropna().index
    return [f'background-color: {color}' if index in selected else '' for index in series.index]


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize HQQ accuracy against series-specific FP32 baselines.')
    parser.add_argument(
        '--input-file', default=os.path.join(BASE_DIR, 'data', 'backbones_quant.csv'),
        help='full path of the downloaded legacy HQQ CSV')
    parser.add_argument(
        '--output-file', default=DEFAULT_OUTPUT,
        help='full path of the primary summary CSV')
    parser.add_argument(
        '--results-dir', default=DEFAULT_RESULTS_DIR,
        help='directory for validation CSVs, XLSX workbooks, and run_config.json')
    parser.add_argument(
        '--serials', nargs='+', type=int, default=sum(EXPECTED_SERIALS.values(), []) + list(BASELINE_SERIALS.values()),
        help='explicit serial numbers to retain; include each required FP32 baseline')
    parser.add_argument(
        '--baseline-overrides', default=os.path.join(BASE_DIR, 'data', 'quant_fp32_baseline_overrides.csv'),
        help='full path of the optional FP32 baseline override CSV')
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


def load_overrides(path, args):
    if not os.path.exists(path):
        return pd.DataFrame(columns=CONTEXT_COLUMNS + ['fp32_top1', 'baseline_source'])
    overrides = pd.read_csv(path)
    require_columns(overrides, ['dataset_name', 'model_name', 'ckpt_kind', 'serial', 'top1', 'source'])
    overrides = overrides[overrides['serial'].isin(args.serials)].copy()
    overrides['series'] = series_from_serial(overrides['serial'])
    if overrides['series'].isna().any():
        raise ValueError('Baseline overrides must use serial 3999 or 4999.')
    overrides = apply_filters(overrides, args)
    overrides = overrides.rename(columns={'top1': 'fp32_top1', 'source': 'baseline_source'})
    if overrides.duplicated(CONTEXT_COLUMNS).any():
        raise ValueError('Baseline overrides contain duplicate series/context keys.')
    return overrides[CONTEXT_COLUMNS + ['fp32_top1', 'baseline_source']]


def prepare_summary(raw, args):
    require_columns(raw, ['serial', 'hqq', 'dataset_name', 'model_name', 'ckpt_path', *HQQ_COLUMNS, *BASELINE_COLUMNS])
    raw = raw.copy()
    raw['serial'] = pd.to_numeric(raw['serial'], errors='raise')
    raw = raw[raw['serial'].isin(args.serials)].copy()
    raw['series'] = series_from_serial(raw['serial'])
    raw = raw[raw['series'].notna()].copy()
    raw['series'] = raw['series'].astype(int)
    raw['model_name'] = raw['model_name'].map(rename_swin)
    raw = apply_filters(raw, args)
    raw['ckpt_kind'] = checkpoint_kind(raw['ckpt_path'])

    hqq = raw[is_true(raw['hqq']) & raw['serial'].isin(sum(EXPECTED_SERIALS.values(), []))].copy()
    hqq = hqq.dropna(subset=HQQ_COLUMNS)
    hqq['hqq_nbits'] = pd.to_numeric(hqq['hqq_nbits'], errors='raise')
    hqq['hqq_group_size'] = pd.to_numeric(hqq['hqq_group_size'], errors='raise')
    baseline = raw[(~is_true(raw['hqq'])) & raw['serial'].isin(BASELINE_SERIALS.values())].copy()
    baseline = baseline.dropna(subset=BASELINE_COLUMNS)
    baseline = baseline.groupby(CONTEXT_COLUMNS, as_index=False).agg(fp32_top1=('top1', 'median'))
    baseline['baseline_source'] = 'raw_series_baseline'
    overrides = load_overrides(args.baseline_overrides, args)
    baseline = pd.concat([baseline, overrides], ignore_index=True).drop_duplicates(CONTEXT_COLUMNS, keep='last')

    cell_keys = CONTEXT_COLUMNS + ['ckpt_path', 'serial', 'hqq_nbits', 'hqq_group_size']
    cells = hqq.groupby(cell_keys, as_index=False).agg(
        hqq_top1=('hqq_top1', 'mean'),
        hqq_top1_median=('hqq_top1', 'median'),
        hqq_top1_std=('hqq_top1', 'std'),
        n_runs=('serial', 'size'),
        n_seeds=('seed', 'nunique'),
    )
    cells = cells.merge(baseline, on=CONTEXT_COLUMNS, how='left')
    cells['baseline_matched'] = cells['fp32_top1'].notna()
    cells['accuracy_ratio'] = cells['hqq_top1'] / cells['fp32_top1']
    return raw, hqq, cells


def completeness_table(hqq, cells, serials):
    selected = set(serials)
    rows = []
    for context, group in hqq.groupby(CONTEXT_COLUMNS, dropna=False):
        series = int(context[0])
        expected = set(EXPECTED_SERIALS[series]).intersection(selected)
        observed = set(group['serial'].astype(int))
        missing = sorted(expected.difference(observed))
        cell_context = cells
        for column, value in zip(CONTEXT_COLUMNS, context):
            cell_context = cell_context[cell_context[column] == value]
        baseline_present = bool(cell_context['baseline_matched'].any())
        rows.append({
            **dict(zip(CONTEXT_COLUMNS, context)),
            'expected_hqq_cells': len(expected),
            'observed_hqq_cells': len(observed),
            'missing_hqq_serials': ', '.join(map(str, missing)),
            'complete_hqq_grid': not missing,
            'expected_baseline_serial': BASELINE_SERIALS[series],
            'baseline_present': baseline_present,
            'complete_context': not missing and baseline_present,
            'raw_hqq_rows': len(group),
        })
    observed_series = set(hqq['series'].astype(int).unique())
    for series, expected_serials in EXPECTED_SERIALS.items():
        expected = set(expected_serials).intersection(selected)
        if expected and series not in observed_series:
            rows.append({
                'series': series,
                'dataset_name': '__series_missing__',
                'model_name': '__series_missing__',
                'ckpt_kind': '__series_missing__',
                'expected_hqq_cells': len(expected),
                'observed_hqq_cells': 0,
                'missing_hqq_serials': ', '.join(map(str, sorted(expected))),
                'complete_hqq_grid': False,
                'expected_baseline_serial': BASELINE_SERIALS[series],
                'baseline_present': False,
                'complete_context': False,
                'raw_hqq_rows': 0,
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


def serial_mapping(hqq):
    mapping = hqq[['series', 'serial', 'hqq_nbits', 'hqq_group_size']].copy()
    mapping['nbits_code'] = (mapping['serial'] % 1000) // 100
    mapping['group_code'] = mapping['serial'] % 10
    mapping = mapping.groupby(['series', 'nbits_code', 'group_code'], as_index=False).agg(
        hqq_nbits=('hqq_nbits', 'first'),
        hqq_group_size=('hqq_group_size', 'first'),
        nbits_values=('hqq_nbits', 'nunique'),
        group_values=('hqq_group_size', 'nunique'),
    )
    if (mapping['nbits_values'] > 1).any() or (mapping['group_values'] > 1).any():
        raise ValueError('A serial code maps to inconsistent HQQ bit or group-size values.')
    if mapping['series'].nunique() == 2:
        compared = mapping.drop(columns='series').duplicated(keep=False)
        if not compared.all():
            raise ValueError('3X0X and 4X0X use different serial-to-HQQ mappings.')
    return mapping


def write_dataset_workbooks(cells, results_dir):
    workbook_columns = [
        'series', 'dataset_name', 'model_name', 'ckpt_kind', 'hqq_nbits',
        'hqq_group_size', 'hqq_top1', 'fp32_top1', 'accuracy_ratio',
        'hqq_top1_std', 'n_runs', 'n_seeds', 'baseline_source', 'baseline_matched',
    ]
    for dataset, frame in cells.groupby('dataset_name'):
        output = os.path.join(results_dir, f'quant_{dataset}.xlsx')
        selected = frame[[column for column in workbook_columns if column in frame]].sort_values(
            ['series', 'model_name', 'ckpt_kind', 'hqq_nbits', 'hqq_group_size'])
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
        raise ValueError('No HQQ rows remain after applying --serials and subset filters.')
    completeness = completeness_table(hqq, cells, args.serials)
    quality = configuration_quality(hqq)
    mapping = serial_mapping(hqq)
    cells.to_csv(args.output_file, index=False)
    completeness.to_csv(os.path.join(args.results_dir, 'completeness.csv'), index=False)
    quality.to_csv(os.path.join(args.results_dir, 'configuration_quality.csv'), index=False)
    mapping.to_csv(os.path.join(args.results_dir, 'serial_mapping.csv'), index=False)
    write_dataset_workbooks(cells, args.results_dir)
    missing = int((~completeness['complete_context']).sum()) if len(completeness) else 0
    unmatched = int((~cells['baseline_matched']).sum())
    print(f'Prepared {len(cells)} HQQ cells; incomplete contexts: {missing}; unmatched baselines: {unmatched}')


if __name__ == '__main__':
    main()
