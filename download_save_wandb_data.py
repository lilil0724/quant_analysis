import os
import argparse
import json
import subprocess
from datetime import datetime, timezone
import wandb
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


CONFIG_COLS = [
    'serial', 'dataset_name', 'model_name', 'freeze_backbone', 'transfer_learning', 'epoch'

    'classifier', 'selector', 'prompt', 'adapter',
    'ila', 'ila_padding', 'ila_ds_conv', 'ila_ds_conv_type',
    'ila_locs', 'ila_ds_locs', 'ila_ds_kernel_size',

    'lr', 'base_lr', 'seed', 'epochs', 'image_size', 'batch_size',
    'num_images_train', 'num_images_val',
]

SUMMARY_COLS = [
    'val_acc', 'val_loss', 'train_acc', 'train_loss',
    'time_total', 'max_memory', 'flops',
    'no_params', 'no_params_trainable', 'throughput',
]

SORT_COLS = [
    'dataset_name', 'serial', 'model_name', 'epoch', 'ila',
    'lr', 'seed', 'host', 'batch_size',
]


# For eval_quant.py runs in the sibling TGDA project (project nycu_pcs/TiT).
# After the cleanup, every run logs a single bare set:
#   - HQQ run               -> top1/loss/vram_gb/params_m (the post-quant model)
#   - Native quant model    -> top1/loss/vram_gb/params_m (the bundled quant model)
#   - Plain fp32 run (4999) -> top1/loss/vram_gb/params_m
# The hqq_* / original_* / acc_drop / params_reduction_m keys were removed from
# eval_quant.py's wandb log; they are dropped here too (see fix_eval_quant.md).
QUANT_CONFIG_COLS = [
    'serial', 'dataset_name', 'model_name', 'seed',
    'image_size', 'batch_size', 'ckpt_path', 'ckpt_path_teacher',
    'quant_cfg_path',
    'hqq', 'hqq_nbits', 'hqq_group_size', 'hqq_exclude',
    'hqq_offload_meta', 'hqq_compute_dtype',
    'hqq_quant_zero', 'hqq_quant_scale', 'hqq_verbose',
    'debugging', 'test_only',
]
QUANT_SUMMARY_COLS = [
    'top1', 'loss', 'vram_gb', 'params_m', 'time_total_s',
]
QUANT_SORT_COLS = [
    'dataset_name', 'serial', 'model_name',
    'hqq_nbits', 'hqq_group_size', 'seed',
]


def get_wandb_project_runs(project, serials=None, vit_only=False):
    api = wandb.Api()

    if vit_only:
        runs = api.runs(path=project, per_page=2000, filters={'$and': [
            {'config.serial': 1}, 
            {'config.model_name': 'vit_b16'},
            {'config.test_only': False},
            # {'config.freeze_backbone': False},
        ]})
    elif serials:
        runs = api.runs(path=project, per_page=2000,
                        filters={'$or': [{'config.serial': s} for s in serials]})
    else:
        runs = api.runs(path=project, per_page=2000)

    total_runs = len(runs)
    runs = [
        run for run in runs
        if str(getattr(run, 'state', '')).lower() == 'finished'
    ]
    print(f'Downloaded runs: {total_runs}')
    print(f'Finished runs kept: {len(runs)}')
    print(f'Runs excluded by state: {total_runs - len(runs)}')
    return runs


def make_df(runs, config_cols, summary_cols):
    data_list_dics = []

    for i, run in enumerate(runs):
        run_data = {}
        try:
            host = {'host': run.metadata.get('host')}
        except:
            print(run)
            host = {'host': None}
        cfg = {col: run.config.get(col, None) for col in config_cols}
        summary = {col: run.summary.get(col, None) for col in summary_cols}

        run_data.update(host)
        run_data.update(cfg)
        run_data.update(summary)

        data_list_dics.append(run_data)

        if (i + 1) % 10 == 0:
            print(f'{i}/{len(runs)}')

    df = pd.DataFrame.from_dict(data_list_dics)
    print(df.head())

    return df


def sort_save_df(df, fp, sort_cols=['serial']):
    df = df.sort_values(by=sort_cols, ascending=[True for _ in sort_cols])
    df.to_csv(fp, header=True, index=False)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # input
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
    # output
    parser.add_argument('--output_file', type=str,
                        help='File path (default: per --quant)')
    parser.add_argument('--output-name', type=str,
                        help='CSV filename stem under results_dir')
    parser.add_argument('--results_dir', type=str, default=os.path.join(BASE_DIR, 'data'),
                        help='The directory where results will be stored')
    parser.add_argument('--sort_cols', nargs='+', type=str,
                        help='override sort columns (default: per --quant)')

    args = parser.parse_args()

    # Select grouped defaults based on --quant / --vit_only.
    #
    # All `default=None` flags above let us tell "user left blank" vs "user
    # explicitly passed X"; the user-passed value wins via `or`. A single
    #   arg.set_defaults(...)
    # call wouldn't express a 3-way choice (training vs quant vs vit_only),
    # which is why the fallback lives here rather than in argparse.
    if args.vit_only:
        args.project_name = 'nycu_pcs/Backbones'
        args.serials = [1]
        args.output_file = args.output_file or 'backbones_vit_stage2.csv'
    elif args.quant:
        args.project_name = args.project_name or 'nycu_pcs/TiT'
        args.config_cols = args.config_cols or QUANT_CONFIG_COLS
        args.summary_cols = args.summary_cols or QUANT_SUMMARY_COLS
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
