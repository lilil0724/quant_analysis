"""Compatibility wrapper for pre-cleanup eval_quant.py wandb logs.

The current eval_quant.py (TGDA-2) still writes hqq_top1 / hqq_loss /
hqq_vram_gb / hqq_params_m (not bare top1/loss/vram_gb/params_m) for
HQQ-quantized runs.  This wrapper overrides _long_from_hqq to read those
hqq_* keys, then delegates everything else to summarize_quant.py.

Usage (from quant_analysis/):
    conda activate opencode_env
    $env:PYTHONPATH = "$env:PYTHONPATH;C:\\iCloudDrive\\project_2\reference\\BackbonesAnalysis"
    python download_save_wandb_data_old.py --quant
    python summarize_quant_old.py
"""

import os
import argparse
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from summarize_quant import (
    QUANT_MODELS_NATIVE,
    make_quant_setting, rename_swin,
    _long_from_native, _long_from_plain_fp32,
    aggregate_results, highlight_per_ds, count_best_per_ds,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _long_from_hqq(row):
    """Exploded HQQ-quantized row from a pre-cleanup eval_quant run (hqq_* keys)."""
    return {
        'dataset_name': row['dataset_name'],
        'model_name': row['model_name'],
        'seed': row['seed'],
        'serial': row['serial'],
        'image_size': row['image_size'],
        'hqq_nbits': row['hqq_nbits'],
        'hqq_group_size': row['hqq_group_size'],
        'hqq_quant_zero': row['hqq_quant_zero'],
        'hqq_quant_scale': row['hqq_quant_scale'],
        'hqq_offload_meta': row['hqq_offload_meta'],
        'hqq_compute_dtype': row['hqq_compute_dtype'],
        'quant_setting': make_quant_setting(
            row['hqq'], row['hqq_nbits'], row['hqq_group_size'],
            row['hqq_quant_zero'], row['hqq_quant_scale'],
            row['hqq_offload_meta'], row['model_name']),
        'acc': row.get('hqq_top1'),
        'vram_gb': row.get('hqq_vram_gb'),
        'params_m': row.get('hqq_params_m'),
        'loss': row.get('hqq_loss'),
        'time_total_s': row.get('time_total_s'),
        'kind': 'hqq',
    }


def load_quant_df(args):
    """Read the downloaded quant CSV (pre-cleanup hqq_* keys) and explode each
       run into long-format rows."""
    df = pd.read_csv(args.input_file_quant)

    if args.main_serials:
        df = df[df['serial'].isin(args.main_serials)]

    if args.subset_datasets:
        df = df[df['dataset_name'].isin(args.subset_datasets)]
    if args.subset_models:
        df = df[df['model_name'].isin(args.subset_models)]

    rows = []
    for _, row in df.iterrows():
        is_hqq = row['hqq'] in (True, 'True', 'true', 1)
        if is_hqq:
            rows.append(_long_from_hqq(row))
        elif row['model_name'] in QUANT_MODELS_NATIVE:
            nat_row = _long_from_native(row)
            if nat_row is not None:
                rows.append(nat_row)
        else:
            rows.append(_long_from_plain_fp32(row))

    long = pd.DataFrame([r for r in rows if r is not None])

    long['model_name'] = long['model_name'].apply(rename_swin)

    if args.input_file_stats_data and os.path.exists(args.input_file_stats_data):
        df_stats = pd.read_csv(args.input_file_stats_data)
        long = long.merge(df_stats, on='dataset_name', how='left')

    return long


def summarize_results(args):
    df = load_quant_df(args)
    df = aggregate_results(df, args)
    df.to_csv(f'{args.output_file}.csv', header=True, index=False)
    highlight_per_ds(df, args)
    count_best_per_ds(df, args)
    return df


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file_quant', type=str,
                        default=os.path.join('data', 'backbones_quant.csv'),
                        help='filename for input .csv file from wandb (TiT project)')
    parser.add_argument('--input_file_stats_data', type=str,
                        default=os.path.join('data', 'stats_datasets.csv'),
                        help='optional dataset stats csv to merge in')

    parser.add_argument('--main_serials', nargs='+', type=int,
                        default=[4000, 4001, 4002, 4003,
                                 4100, 4101, 4102, 4103,
                                 4200, 4201, 4202, 4203,
                                 4300, 4301, 4302, 4303,
                                 4400, 4401, 4402, 4403,
                                 4500, 4501, 4502, 4503,
                                 4600, 4601, 4602, 4603,
                                 4700, 4701, 4702, 4703,
                                 4999])
    parser.add_argument('--subset_datasets', nargs='+', type=str, default=None)
    parser.add_argument('--subset_models', nargs='+', type=str, default=None)
    parser.add_argument('--subset_quant_settings', nargs='+', type=str, default=None,
                        help='restrict to e.g. fp32 hqq_n2_g64 hqq_n4_g128 native_BHViT')
    parser.add_argument('--top_k', type=int, default=10,
                        help='how many top k for each dataset to count')

    parser.add_argument('--output_file', type=str, default='quant',
                        help='filename stem for output .csv/.xlsx files (no extension)')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join(BASE_DIR, 'results_all', 'quant_old'),
                        help='the directory where results will be stored')
    parser.add_argument('--output-name', type=str,
                        help='output folder name under results_all')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.output_name:
        args.results_dir = os.path.join(BASE_DIR, 'results_all', args.output_name)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    args.output_file = os.path.join(args.results_dir, args.output_file)
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

    summarize_results(args)


if __name__ == '__main__':
    main()
