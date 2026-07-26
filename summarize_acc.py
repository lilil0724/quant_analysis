import os
import sys
import argparse

import pandas as pd

from utils import add_setting, rename_swin, diff_setting, diff_384_224, highlight_top_k


DATASETS_REDUCED = ['aircraft', 'cub', 'soygene', 'soylocal']

MODELS_OG = [
    'vit_b16', 'vgg19_bn', 'van_b3', 'swin_base_patch4_window7_224_in22k',
    'resnetv2_101x3_bitm_in21k', 'resnetv2_101', 'resnet101', 
    'convnext_base_in22k', 'beitv2_base_patch16_224_in22k'
]

GROUPS = {
    'ric_-3': 'soyglobal soylocal',
    'ric_-2': 'inat17 soygene',
    'ric_-1': 'cars cotton cub dafb flowers nabirds soyageing',
    'ric_0': 'aircraft dogs moe pets vegfru',
    'ric_1': 'food',

    'imb_0': 'aircraft cars cotton cub dogs flowers food pets soyageing soyglobal soylocal vegfru',
    'imb_1': 'moe nabirds soygene',
    'imb_3': 'dafb inat17',

    'train_2': 'cotton',
    'train_3': 'flowers soylocal',
    'train_4': 'aircraft cars cub dogs moe nabirds pets soyageing soygene soyglobal',
    'train_5': 'food vegfru',
    'train_6': 'dafb inat17',

    'machines': 'aircraft cars',

    'birds': 'cub nabirds',
    'pets_all': 'dogs pets',
    'animals_only': 'cub nabirds dogs pets',
    'animals_all': 'cub nabirds dogs pets inat17',

    'leaves': 'cotton soyageing soygene soyglobal soylocal',
    'soy': 'soyageing soygene soyglobal soylocal',
    'plants_only': 'cotton flowers soyageing soygene soyglobal soylocal vegfru',
    'plants_all': 'cotton flowers inat17 soyageing soygene soyglobal soylocal vegfru',
    'food_all': 'food vegfru',

    'natural': 'aircraft cars cotton cub dogs flowers food inat17 nabirds pets soyageing soygene soyglobal soylocal vegfru',
    'anime': 'dafb moe',
    'all': 'aircraft cars cotton cub dafb dogs flowers food inat17 moe nabirds pets soyageing soygene soyglobal soylocal vegfru',
}


def filter_df(args, df):
    #Filter original models (9) and all datasets or all models and 4 datasets
    if args.filter_og_only:
        df = df[df['model_name'].isin(MODELS_OG)]
    else:
        df = df[df['dataset_name'].isin(DATASETS_REDUCED)]

    if hasattr(args, 'subset_datasets') and args.subset_datasets:
        df = df[df['dataset_name'].isin(args.subset_datasets)]

    # filter by setting and sort
    if args.filter_og_only:
        settings = [f'{setting}_224' for setting in ('fz', 'ft', 'cal', 'cal_ap', 'cal_cm', 'cal_cm_ap')]
    else:
        settings =  ['fz_224', 'fz_384', 'ft_224', 'ft_384', 'cal_224', 'cal_384', 'cal_448',
                'cal_ap_224', 'cal_ap_384', 'cal_ap_448', 'cal_cm_224', 'cal_cm_448', 'cal_cm_ap_224', 'cal_cm_ap_448']

    df = df[df['setting'].isin(settings)]

    df['setting_order'] = pd.Categorical(df['setting'], categories=settings, ordered=True)
    df = df.sort_values(by=['dataset_name', 'setting_order', 'model_name'], ascending=True)
    df = df.drop(columns=['setting_order'])

    return df


def add_dataset_groups(df, args):
    # filter based on dataset groups (category: machines, order of magnitude: 3, etc)
    # and add them as new "datasets"
    if args.filter_og_only:
        for name, group in GROUPS.items():
            group = group.split(' ')
            subset = df[df['dataset_name'].isin(group)].copy(deep=False)
            subset['dataset_name'] = name
            df = pd.concat([df, subset], axis=0)

    return df



def load_acc_df(args):
    df_stage2 = pd.read_csv(args.input_file_stage2)
    df_cal_ap = pd.read_csv(args.input_file_cal_ap)
    df_stats = pd.read_csv(args.input_file_stats_data)

    # Filter by serial column
    df_stage2['serial'] = df_stage2['Name'].str.split('_').str[-1].astype(int)
    df_cal_ap['serial'] = df_cal_ap['Name'].str.split('_').str[-1].astype(int)

    # Filter by selected serials
    df_stage2 = df_stage2[df_stage2['serial'].isin(args.main_serials)]
    df_cal_ap = df_cal_ap[df_cal_ap['serial'].isin(args.main_serials)]

    if args.cal_single_values:
        df_stage2 = df_stage2[~df_stage2['selector'].isin('cal')]
    else:
        df_cal_ap = df_cal_ap[df_cal_ap['cal_ap_only'] == True]

    df = pd.concat([df_stage2, df_cal_ap], axis=0)

    df = add_setting(df)
    df['model_name'] = df['model_name'].apply(rename_swin)

    df['acc'] = df[['test_acc']].apply(lambda x: x.fillna(value=df['val_acc']))

    df = df[['dataset_name', 'setting', 'model_name', 'acc']]

    # Merge with dataset statistics
    df = df.merge(df_stats, on='dataset_name', how='left')

    add_dataset_groups(df, args)

    return df


def highlight_per_ds(df, args):
    metric_list = [col for col in df.columns if any(['diff_' in col, 'acc' in col])]

    ds_list = df['dataset_name'].unique()
    for ds in ds_list:
        df_ds = df[df['dataset_name'] == ds]

        df_ds.style.apply(highlight_top_k, largest=True, k=5, color='lightgreen', axis=0, subset=metric_list).apply(
            highlight_top_k, largest=False, k=5, color='lightcoral', axis=0, subset=metric_list).to_excel(
            f'{args.output_file}_{ds}.xlsx', header=True, index=False, engine='openpyxl')

    return 0


def count_best_per_ds(df, args):
    df_best = pd.DataFrame()

    metric_list = [col for col in df.columns if any(['diff_' in col, 'acc' in col])]

    ds_list = df['dataset_name'].unique()

    with open(f'{args.output_file}_counts.txt', 'w') as sys.stdout:

        for metric in metric_list:
            if 'norm' in metric:
                m = metric.replace('_norm', '')
            else:
                m = metric

            for ds in ds_list:
                df_ds = df[df['dataset_name'] == ds]        

                ascending = True if 'std' in metric else False
                df_ds_sorted = df_ds.sort_values(by=metric, ascending=ascending)

                df_ds_sorted = df_ds_sorted.iloc[:args.top_k]

                df_ds_sorted = df_ds_sorted[['dataset_name', 'setting', 'model_name', m]]

                df_best = pd.concat([df_best, df_ds_sorted], axis=0)

            print(metric)
            print(df_best.to_string())
            print(df_best[['setting', 'model_name']].value_counts().to_string(), '\n')
            print(df_best.model_name.value_counts().to_string(), '\n')
            print(df_best.setting.value_counts().to_string(), '\n\n')

            df_best = pd.DataFrame()

    return 0


def aggregate_results(df, args):
    df_std = df.groupby(['dataset_name', 'setting', 'model_name'], as_index=False).std(numeric_only=True)
    df_median = df.groupby(['dataset_name', 'setting', 'model_name'], as_index=False).median(numeric_only=True)
    df = df.groupby(['dataset_name', 'setting', 'model_name'], as_index=False).mean(numeric_only=True)
    df['acc_median'] = df_median['acc']
    df['acc_std'] = df_std['acc']

    if not args.filter_og_only:
        df = diff_384_224(df, 'acc')
    df = diff_setting(df, 'acc')
    
    df = filter_df(args, df)
    print(df["dataset_name"].unique())
    return df


def summarize_results(args):
    df = load_acc_df(args)

    df = aggregate_results(df, args)

    df.to_csv(f'{args.output_file}.csv', header=True, index=False)

    highlight_per_ds(df, args)

    count_best_per_ds(df, args)

    return df


def parse_args():
    parser = argparse.ArgumentParser()

    # input
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

    parser.add_argument('--main_serials', nargs='+', type=int, default=[1,2,3,4,5,6, 8])

    parser.add_argument('--top_k', type=int, default=10,
                        help='how many top k for each server to calculate counts/mode')

    # output
    parser.add_argument('--output_file', type=str, default='acc',
                        help='filename for output .csv file')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'acc'),
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

    df = summarize_results(args)


if __name__ == '__main__':
    main()