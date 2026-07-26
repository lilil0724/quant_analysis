import os
import sys
import argparse

import pandas as pd

from summarize_acc import load_acc_df, aggregate_results
from utils import add_setting, min_max_larger_better, min_max_smaller_better, \
    diff_setting, diff_384_224, highlight_top_k


SETTINGS = ['fz_224', 'fz_384', 'ft_224', 'ft_384', 'cal_224', 'cal_384', 'cal_448',
                'cal_ap_224', 'cal_ap_384', 'cal_ap_448', 'cal_cm_224', 'cal_cm_448', 'cal_cm_ap_224', 'cal_cm_ap_448']

DATASETS = ['aircraft', 'cub', 'soygene', 'soylocal']

MODELS_OG = [
    'vit_b16', 'vgg19_bn', 'van_b3', 'swin_base_patch4_window7_224_in22k',
    'resnetv2_101x3_bitm_in21k', 'resnetv2_101', 'resnet101', 
    'convnext_base_in22k', 'beitv2_base_patch16_224_in22k'
]

MODELS_DIC = {
    'swin_large_patch4_window12_384_in22k': 'swin_large_patch4_window7_224_in22k',
    'swin_base_patch4_window12_384_in22k': 'swin_base_patch4_window7_224_in22k',
    'swin_base_patch4_window12_384': 'swin_base_patch4_window7_224',
}

METRICS_UNIFORM = ['acc_norm', 'acc_mean', 'acc_std', 'acc_train_only',
                   'diff_setting_acc', 'diff_is_acc', 'params',
                   'time_train', 'vram_train']
SMALLER_BETTER_LIST = ('diff_setting_vram', 'diff_is_vram',
                       'diff_setting_time', 'diff_is_time',
                       'diff_setting_latency', 'diff_is_latency', 'std')

EPOCHS = 200

def rename_var(x):
    if x in MODELS_DIC.keys():
        return MODELS_DIC[x]
    return x

def duplicate_cal_cm_ap_train(df):
    mapping = {
        'cal_cm_224': 'cal_cm_ap_224',
    }

    new_rows = []

    hosts = df['Hostname'].unique()
    for host in hosts:
        df_host = df[df['Hostname'] == host]

        for model_name in df_host['model_name'].unique():
            df_model = df_host[df_host['model_name'] == model_name]

            for src_setting, tgt_setting in mapping.items():
                df_src = df_model[df_model['setting'] == src_setting]
                df_tgt = df_model[df_model['setting'] == tgt_setting]

                # only duplicate if target missing
                if df_src.empty:
                    continue

                batch_sizes = df_src['batch_size'].unique()
                for batch_size in batch_sizes:
                    df_src_row = df_src[df_src['batch_size'] == batch_size]
                    df_tgt_row = df_tgt[df_tgt['batch_size'] == batch_size]

                    # skip if target row already exists
                    if not df_tgt_row.empty:
                        continue

                    # duplicate row
                    df_copy = df_src_row.copy()
                    df_copy['setting'] = tgt_setting
                    new_rows.append(df_copy)

    if new_rows:
        df = pd.concat([df] + new_rows, ignore_index=True)

    return df

def summarize_train_cost(args):
    df = pd.read_csv(args.input_file_train_cost)
    df = add_setting(df)
    df['model_name'] = df['model_name'].apply(rename_var)

    df['time_train'] = df['time_total'] / 60
    df['tp_train'] = ((df['num_images_val'] + (df['num_images_train'] * EPOCHS)) 
        / (df['time_total'] * 60))
    df['vram_train'] = df['max_memory']

    df = duplicate_cal_cm_ap_train(df)

    df_cal_ap = df[df['setting'].isin(['cal_224', 'cal_384', 'cal_448'])].copy(deep=False)
    df_cal_ap['setting'] = df_cal_ap['setting'].str.replace('cal_', 'cal_ap_')
    df_cal_ap = df_cal_ap.assign(cal_op_only = True)

    df = pd.concat([df, df_cal_ap], axis=0).reset_index(drop=True)
    df = df[['setting', 'model_name', 'no_params', 'no_params_trainable', 'tp_train', 'time_train', 'vram_train', 'Hostname']]

    df = min_max_smaller_better(df, 'no_params')
    df = min_max_smaller_better(df, 'no_params_trainable')
    # df = min_max_larger_better(df, 'tp_train')
    df = min_max_smaller_better(df, 'time_train')
    df = min_max_smaller_better(df, 'vram_train')

    df = diff_384_224(df, 'time_train', hostname=True)
    df = diff_384_224(df, 'vram_train', hostname=True)
    df = diff_setting(df, 'time_train', hostname=True)
    df = diff_setting(df, 'vram_train', hostname=True)

    df['setting_order'] = pd.Categorical(df['setting'], categories=SETTINGS, ordered=True)
    df = df.sort_values(by=['setting_order', 'model_name'], ascending=True)
    df = df.drop(columns=['setting_order', 'Hostname'])

    return df


def duplicate_cal_cm_inference(df):
    mapping = {
        'cal_224': 'cal_cm_224',
        'cal_ap_224': 'cal_cm_ap_224',
    }

    new_rows = []

    hosts = df['Hostname'].unique()
    for host in hosts:
        df_host = df[df['Hostname'] == host]

        for model_name in df_host['model_name'].unique():
            df_model = df_host[df_host['model_name'] == model_name]

            for src_setting, tgt_setting in mapping.items():
                df_src = df_model[df_model['setting'] == src_setting]
                df_tgt = df_model[df_model['setting'] == tgt_setting]

                # only duplicate if target missing
                if df_src.empty:
                    continue

                batch_sizes = df_src['batch_size'].unique()
                for batch_size in batch_sizes:
                    df_src_row = df_src[df_src['batch_size'] == batch_size]
                    df_tgt_row = df_tgt[df_tgt['batch_size'] == batch_size]

                    # skip if target row already exists
                    if not df_tgt_row.empty:
                        continue

                    # duplicate row
                    df_copy = df_src_row.copy()
                    df_copy['setting'] = tgt_setting
                    new_rows.append(df_copy)

    if new_rows:
        df = pd.concat([df] + new_rows, ignore_index=True)

    return df

def summarize_test_cost(args):
    df = pd.read_csv(args.input_file_inference_cost)
    df = add_setting(df)
    df['model_name'] = df['model_name'].apply(rename_var)
    df = duplicate_cal_cm_inference(df)

    best_list = []
    df_best = pd.DataFrame()

    host_list = df['Hostname'].unique()
    for host in host_list:
        df_subset = df[df['Hostname'] == host]

        model_list = df_subset['model_name'].unique()
        for model in model_list:

            df_subset = df[
                (df['Hostname'] == host) & (df['model_name'] == model)
            ]

            setting_list = df_subset['setting'].unique()
            for setting in setting_list:
                df_subset = df[
                    (df['Hostname'] == host) & (df['model_name'] == model) & (df['setting'] == setting)
                ]

                df_subset = df_subset.dropna(subset=['throughput'])

                df_subset_sorted = df_subset.sort_values(by=['throughput'], ascending=False)

                bs_batched = df_subset_sorted['batch_size'].iloc[0]
                tp_batched = df_subset_sorted['throughput'].iloc[0]
                vram_batched = df_subset_sorted['max_memory'].iloc[0]

                df_subset_sorted = df_subset.sort_values(by=['batch_size'], ascending=True)

                bs_stream = df_subset_sorted['batch_size'].iloc[0]
                tp_stream = df_subset_sorted['throughput'].iloc[0]
                latency_stream = 1 / tp_stream
                vram_stream = df_subset_sorted['max_memory'].iloc[0]

                # save tp and vram for each model-setting pair for each host
                best_list.append({
                    'Hostname': host, 'model_name': model, 'setting': setting,
                    'tp_stream': tp_stream, 'vram_stream': vram_stream, 'latency_stream': latency_stream, # 'bs_stream': bs_stream,
                    'tp_batched': tp_batched, 'vram_batched': vram_batched, 'bs_batched': bs_batched
                })

        # normalized results for each host
        df_best_host = pd.DataFrame.from_dict(best_list)
        best_list = []

        df_best_host = min_max_smaller_better(df_best_host, 'latency_stream')
        df_best_host = min_max_smaller_better(df_best_host, 'vram_stream')
        df_best_host = min_max_larger_better(df_best_host, 'tp_batched')
        df_best_host = min_max_smaller_better(df_best_host, 'vram_batched')

        df_best_host = diff_384_224(df_best_host, 'latency_stream', hostname=True)
        df_best_host = diff_384_224(df_best_host, 'vram_stream', hostname=True)
        df_best_host = diff_384_224(df_best_host, 'tp_batched', hostname=True)
        df_best_host = diff_384_224(df_best_host, 'vram_batched', hostname=True)
        df_best_host = diff_setting(df_best_host, 'latency_stream', hostname=True)
        df_best_host = diff_setting(df_best_host, 'vram_stream', hostname=True)
        df_best_host = diff_setting(df_best_host, 'tp_batched', hostname=True)
        df_best_host = diff_setting(df_best_host, 'vram_batched', hostname=True)

        df_best = pd.concat([df_best, df_best_host], axis=0)

    df_best['setting_order'] = pd.Categorical(df_best['setting'], categories=SETTINGS, ordered=True)
    df_best = df_best.sort_values(by=['Hostname', 'setting_order', 'model_name'], ascending=True)
    df_best = df_best.drop(columns=['setting_order'])

    return df_best


def combine_train_test(df_train_cost, df_test_cost):
    # combine the summarized train and test cost to get overall cost
    df_cost = pd.merge(df_test_cost, df_train_cost, how='left', on=['setting', 'model_name'])

    return df_cost


def preprocess_acc_df(args):
    df_acc = load_acc_df(args)
    print(df_acc["dataset_name"].unique())
    df_acc = aggregate_results(df_acc, args)
    
    # mean, std and median across all datasets
    df_std = df_acc.groupby(['setting', 'model_name'], as_index=False).std(numeric_only=True)
    df_median = df_acc.groupby(['setting', 'model_name'], as_index=False).median(numeric_only=True)
    df_acc = df_acc.groupby(['setting', 'model_name'], as_index=False).mean(numeric_only=True)
    df_acc['acc_median'] = df_median['acc']
    df_acc['acc_std'] = df_std['acc']

    df_acc = min_max_larger_better(df_acc, 'acc')

    return df_acc


def highlight_per_host(df, args):
    metric_list = [col for col in df.columns if any(['norm' in col, 'diff_' in col, 'acc_' in col])]

    smaller_better_list = [m for m in metric_list if any([kw in m for kw in SMALLER_BETTER_LIST])]
    larger_better_list = [m for m in metric_list if not any([kw in m for kw in SMALLER_BETTER_LIST])]

    host_list = df['Hostname'].unique()
    for host in host_list:
        df_host = df[df['Hostname'] == host]

        df_host.style.apply(highlight_top_k, largest=True, k=5, color='lightgreen', axis=0, subset=larger_better_list).apply(
            highlight_top_k, largest=False, k=5, color='lightcoral', axis=0, subset=larger_better_list).apply(
            highlight_top_k, largest=False, k=5, color='lightgreen', axis=0, subset=smaller_better_list).apply(
            highlight_top_k, largest=True, k=5, color='lightcoral', axis=0, subset=smaller_better_list).to_excel(
            f'{args.output_file}_{host}.xlsx', header=True, index=False, engine='openpyxl')

    return 0


def count_best_per_host(df, args):
    # which models obtain best metrics across hosts
    df_best = pd.DataFrame()

    metric_list = [col for col in df.columns if any(['norm' in col, 'diff_' in col, 'acc_' in col])]
    smaller_better_list = [m for m in metric_list if any([kw in m for kw in SMALLER_BETTER_LIST])]

    host_list = df['Hostname'].unique()

    with open(f'{args.output_file}_counts.txt', 'w') as sys.stdout:

        for metric in metric_list:
            m = metric.replace('_norm', '')

            for host in host_list:
                df_host = df[df['Hostname'] == host]        

                if metric in smaller_better_list:
                    ascending = True
                else:
                    ascending = False

                df_host_sorted = df_host.sort_values(by=metric, ascending=ascending)
                df_host_sorted = df_host_sorted[['Hostname', 'setting', 'model_name', m]]

                df_host_sorted = df_host_sorted.iloc[:args.top_k]

                df_best = pd.concat([df_best, df_host_sorted], axis=0)

                # these metrics are same across hosts so just show for one host
                if any([m in metric for m in METRICS_UNIFORM]):
                    break

            print(metric)
            print(df_best.to_string())
            print(df_best[['setting', 'model_name']].value_counts().to_string(), '\n')
            print(df_best.model_name.value_counts().to_string(), '\n')
            print(df_best.setting.value_counts().to_string(), '\n\n')

            df_best = pd.DataFrame()

    sys.stdout = sys.__stdout__

    return 0


def diff_host_settings_mean_std(df, args):
    # show how setting increase cost/acc (diff) per server
    metric_list = [col for col in df.columns if any(['diff_' in col])]

    cols = ['Hostname', 'setting'] + metric_list
    df = df[cols]

    mean = df.groupby(['Hostname', 'setting'], as_index=False).mean(numeric_only=True)
    std = df.groupby(['Hostname', 'setting'], as_index=False).std(numeric_only=True)
    std = std.rename(columns={col: f'{col}_std' for col in std.columns if 'diff' in col})

    mean_std = pd.merge(mean, std, how='left', on=['Hostname', 'setting'])
    mean_std.to_csv(f'{args.output_file}_diff_host_setting_mean_std.csv', header=True, index=False)

    return 0


def combine_acc_cost(df_acc, df_cost, args):
    # combine the summarized train and test cost to get overall cost
    df = pd.merge(df_cost, df_acc, how='left', on=['setting', 'model_name'])

    # filter models if needed 
    if args.filter_og_only:
    # (also needs to use image size 224 since 384 is only available for 4 datasets)
        args.image_size = 224
        df = df[df['model_name'].isin(MODELS_OG)]
        df = df.drop(columns=[col for col in df.columns if 'diff_is' in col])

    if args.setting and args.image_size:
        setting = f'{args.setting}_{args.image_size}'
        df = df[df['setting'] == setting]
    elif args.setting:
        settings = [f'{args.setting}_{image_size}' for image_size in (224, 384)]
        df = df[df['setting'].isin(settings)]
    elif args.image_size:
        settings = [f'{setting}_{args.image_size}' for setting in ('ft', 'fz', 'cal', 'cal_ap')]
        df = df[df['setting'].isin(settings)]

    # add acc cost normalized metrics
    df['acc_params'] = (df['acc_norm'] + df['no_params_norm']) / 2
    df['acc_params_trainable'] = (df['acc_norm'] + df['no_params_trainable_norm']) / 2
    df['acc_time_train'] = (df['acc_norm'] + df['time_train_norm']) / 2
    df['acc_vram_train'] = (df['acc_norm'] + df['vram_train_norm']) / 2
    df['acc_train_only'] = (df['acc_norm'] + df['time_train_norm'] + df['vram_train_norm']) / 3
    df['acc_latency_stream'] = (df['acc_norm'] + df['latency_stream_norm']) / 2
    df['acc_vram_stream'] = (df['acc_norm'] + df['vram_stream_norm']) / 2
    df['acc_stream'] = (df['acc_norm'] + df['latency_stream_norm'] + df['vram_stream_norm']) / 3
    df['acc_train_stream'] = (df['acc_norm'] + df['latency_stream_norm'] + df['vram_stream_norm'] + df['time_train_norm'] + df['vram_train_norm']) / 5
    df['acc_tp_batched'] = (df['acc_norm'] + df['tp_batched_norm']) / 2
    df['acc_vram_batched'] = (df['acc_norm'] + df['vram_batched_norm']) / 2
    df['acc_batched'] = (df['acc_norm'] + df['tp_batched_norm'] + df['vram_batched_norm']) / 3
    df['acc_train_batched'] = (df['acc_norm'] + df['tp_batched_norm'] + df['vram_batched_norm'] + df['time_train_norm'] + df['vram_train_norm']) / 5

    df.to_csv(f'{args.output_file}.csv', header=True, index=False)

    highlight_per_host(df, args)

    count_best_per_host(df, args)

    diff_host_settings_mean_std(df, args)

    return df


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--input_file_train_cost', type=str, 
                        default=os.path.join('data', 'backbones_train_cost_cub_3090.csv'),
                        help='filename for input .csv file from wandb')
    parser.add_argument('--input_file_inference_cost', type=str, 
                        default=os.path.join('data', 'backbones_inference_cost.csv'),
                        help='filename for input .csv file from wandb')
    parser.add_argument('--input_file_stage2', type=str, 
                        default=os.path.join('data', 'backbones_stage2.csv'),
                        help='filename for input .csv file from wandb')
    parser.add_argument('--input_file_cal_ap', type=str, 
                        default=os.path.join('data', 'backbones_accuracy_cal_ap.csv'),
                        help='filename for input .csv file from wandb')
    parser.add_argument('--input_file_stats_data', type=str,
                        default=os.path.join('data', 'stats_datasets.csv'),
                        help='filename for input .csv file')

    parser.add_argument('--cal_single_values', action='store_true',
                        help='use cal single acc from input_file_cal_ap instead of avgs')

    # filter
    parser.add_argument('--filter_og_only', action='store_true', help='')
    parser.add_argument('--subset_datasets', nargs='+', type=str, default=None,
                         help='list of datasets to use in case want to focus on subset')

    parser.add_argument('--main_serials', nargs='+', type=int, default=[1,2,3,4,5,6, 8])

    parser.add_argument('--image_size', type=int, default=None,
                        help='if specified then filter by image_size')
    parser.add_argument('--setting', type=str, default=None,
                        help='filter by setting: (ft, fz, cal, cal_ap)')

    parser.add_argument('--top_k', type=int, default=10,
                        help='how many top k for each server to calculate counts/mode')

    # output
    parser.add_argument('--output_file', type=str, default='cost',
                        help='filename for output .csv file')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'cost'),
                        help='The directory where results will be stored')

    # style
    parser.add_argument('--cmap', type=str, default='coolwarm',
                        help='color map for styling (background gradient), also RdYlGn')

    args= parser.parse_args()
    return args


def main():
    args = parse_args()

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    
    args.output_file = os.path.join(args.results_dir, args.output_file)

    df_train_cost = summarize_train_cost(args)

    df_test_cost = summarize_test_cost(args)

    df_cost = combine_train_test(df_train_cost, df_test_cost)

    df_acc = preprocess_acc_df(args)

    df_acc_cost = combine_acc_cost(df_acc, df_cost, args)

    return 0


if __name__ == '__main__':
    main()