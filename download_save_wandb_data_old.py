"""Download wandb runs compatible with summarize_quant_old.py.

Same as download_save_wandb_data.py but includes hqq_* summary keys
for pre-cleanup eval_quant.py runs (TGDA-2 still writes hqq_top1 etc.).
"""

import os
import argparse
import json
import subprocess
from datetime import datetime, timezone

from download_save_wandb_data import (
    CONFIG_COLS, SUMMARY_COLS, SORT_COLS,
    QUANT_CONFIG_COLS, QUANT_SORT_COLS,
    get_wandb_project_runs, make_df, sort_save_df,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

QUANT_SUMMARY_COLS_OLD = [
    'hqq_top1', 'hqq_loss', 'hqq_vram_gb', 'hqq_params_m',
    'top1', 'loss', 'vram_gb', 'params_m', 'time_total_s',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str,
                        help='project_entity/project_name (default: per --quant)')
    parser.add_argument('--quant', action='store_true',
                        help='download eval_quant.py runs (nycu_pcs/TiT, '
                             'QUANT_* col sets, output backbones_quant.csv)')
    parser.add_argument('--vit_only', action='store_true')
    parser.add_argument('--serials', nargs='+', type=int, default=[1, 3])
    parser.add_argument('--config_cols', nargs='+', type=str,
                        help='override config columns (default: per --quant)')
    parser.add_argument('--summary_cols', nargs='+', type=str,
                        help='override summary columns (default: per --quant)')
    parser.add_argument('--output_file', type=str,
                        help='File path (default: per --quant)')
    parser.add_argument('--output-name', type=str,
                        help='CSV filename stem under results_dir')
    parser.add_argument('--results_dir', type=str, default=os.path.join(BASE_DIR, 'data'),
                        help='The directory where results will be stored')
    parser.add_argument('--sort_cols', nargs='+', type=str,
                        help='override sort columns (default: per --quant)')

    args = parser.parse_args()

    if args.vit_only:
        args.project_name = 'nycu_pcs/Backbones'
        args.serials = [1]
        args.output_file = args.output_file or 'backbones_vit_stage2.csv'
    elif args.quant:
        args.project_name = args.project_name or 'nycu_pcs/TiT'
        args.config_cols = args.config_cols or QUANT_CONFIG_COLS
        args.summary_cols = args.summary_cols or QUANT_SUMMARY_COLS_OLD
        args.sort_cols = args.sort_cols or QUANT_SORT_COLS
        args.output_file = args.output_file or 'backbones_quant.csv'
    else:
        args.project_name = args.project_name or 'nycu_pcs/FGIRFT'
        args.config_cols = args.config_cols or CONFIG_COLS
        args.summary_cols = args.summary_cols or SUMMARY_COLS
        args.sort_cols = args.sort_cols or SORT_COLS
        args.output_file = args.output_file or 'fgirft_stage2.csv'

    return args


def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    if args.output_name:
        args.output_file = args.output_name
        if not args.output_file.lower().endswith('.csv'):
            args.output_file += '.csv'
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

    runs = get_wandb_project_runs(args.project_name, args.serials, args.vit_only)

    df = make_df(runs, args.config_cols, args.summary_cols)

    sort_save_df(df, args.output_file, args.sort_cols)

    return 0


if __name__ == '__main__':
    main()
