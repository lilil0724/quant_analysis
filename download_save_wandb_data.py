"""Download the legacy HQQ quantization runs used by the formal analysis.

The TiT project logs HQQ outcomes under hqq_* fields and independent FP32
baselines under bare metric fields. This schema is intentionally retained until
the upstream experiments are migrated.
"""

import argparse
import json
import netrc
import os
import subprocess
from datetime import datetime, timezone

import pandas as pd
import wandb


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, 'data')
DEFAULT_OUTPUT = os.path.join(DEFAULT_RESULTS_DIR, 'backbones_quant.csv')
CONFIG_COLUMNS = [
    'serial', 'dataset_name', 'model_name', 'seed', 'image_size', 'batch_size',
    'ckpt_path', 'ckpt_path_teacher', 'quant_cfg_path', 'hqq', 'hqq_nbits',
    'hqq_group_size', 'hqq_exclude', 'hqq_offload_meta', 'hqq_compute_dtype',
    'hqq_quant_zero', 'hqq_quant_scale', 'hqq_verbose', 'debugging', 'test_only',
]
SUMMARY_COLUMNS = [
    'hqq_top1', 'hqq_loss', 'hqq_vram_gb', 'hqq_params_m',
    'top1', 'loss', 'vram_gb', 'params_m', 'time_total_s',
]
SORT_COLUMNS = [
    'dataset_name', 'serial', 'model_name', 'ckpt_path', 'hqq_nbits',
    'hqq_group_size', 'seed',
]


def scan_serials(series):
    return [series + nbits * 100 + group for nbits in range(8) for group in range(4)]


DEFAULT_SERIALS = scan_serials(3000) + [3999] + scan_serials(4000) + [4999]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download legacy HQQ runs for the 3X0X and 4X0X scans.')
    parser.add_argument('--project-name', default='nycu_pcs/TiT')
    parser.add_argument(
        '--serials', nargs='+', type=int, default=DEFAULT_SERIALS,
        help='explicit serial numbers to download; include each required FP32 baseline')
    parser.add_argument(
        '--output-file', default=DEFAULT_OUTPUT,
        help='full path of the downloaded CSV')
    parser.add_argument(
        '--results-dir', default=DEFAULT_RESULTS_DIR,
        help='directory for run_config.json')
    return parser.parse_args()


def write_run_config(path, args):
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
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2)


def has_wandb_credentials():
    if os.environ.get('WANDB_API_KEY'):
        return True
    try:
        credentials = netrc.netrc().authenticators('api.wandb.ai')
        return bool(credentials and credentials[2])
    except (FileNotFoundError, netrc.NetrcParseError):
        return False


def main():
    args = parse_args()
    output_file = os.path.abspath(args.output_file)
    results_dir = os.path.abspath(args.results_dir)
    args.output_file = output_file
    args.results_dir = results_dir
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    if not has_wandb_credentials():
        raise RuntimeError(
            'W&B authentication is required. Set WANDB_API_KEY or run "wandb login" before downloading.'
        )
    write_run_config(os.path.join(results_dir, 'run_config.json'), args)
    api = wandb.Api()
    runs = api.runs(
        path=args.project_name,
        per_page=2000,
        filters={
            '$and': [
                {'state': 'finished'},
                {'$or': [{'config.serial': serial} for serial in args.serials]},
            ]
        },
    )
    rows = []
    total = len(runs)
    for run in runs:
        metadata = getattr(run, 'metadata', None) or {}
        row = {'host': metadata.get('host')}
        row.update({column: run.config.get(column) for column in CONFIG_COLUMNS})
        row.update({column: run.summary.get(column) for column in SUMMARY_COLUMNS})
        rows.append(row)

    result = pd.DataFrame(rows, columns=['host', *CONFIG_COLUMNS, *SUMMARY_COLUMNS])
    if not result.empty:
        result = result.sort_values(SORT_COLUMNS, kind='stable')
    result.to_csv(output_file, index=False)
    print(f'Finished runs kept: {len(result)} of {total}')
    print(f'Wrote {output_file}')


if __name__ == '__main__':
    main()
