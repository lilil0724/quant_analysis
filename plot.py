import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from summarize_acc import load_acc_df, diff_384_224, diff_setting

from utils import rename_vars, input_output_args, filtering_args, plot_args, MODELS_OG, DATASETS_REDUCED, \
    SETTINGS_DIC_IS, SETTINGS_DIC, MODELS_DIC, DATASETS_DIC, VAR_DIC


def filter_df(args, df):
    #Filter original models (9) and all datasets or all models and 4 datasets
    if args.filter_og_only:
        df = df[df['model_name'].isin(MODELS_OG)]
    else:
        df = df[df['dataset_name'].isin(DATASETS_REDUCED)]

    df = df[df['model_name'].isin(args.subset_models)]
    df = df[df['dataset_name'].isin(args.subset_datasets)]

    if args.setting and args.image_size:
        settings = [f'{args.setting}_{args.image_size}']
    elif args.setting:
        settings = [f'{args.setting}_{image_size}' for image_size in (224, 384, 448)]
    elif args.image_size:
        settings = [f'{setting}_{args.image_size}' for setting in ('fz', 'ft', 'cal', 'cal_ap', 'cal_cm', 'cal_cm_ap')]
    else:
        settings = ['fz_224', 'fz_384', 'ft_224', 'ft_384', 'cal_224', 'cal_384', 'cal_448',
                'cal_ap_224', 'cal_ap_384', 'cal_ap_448', 'cal_cm_224', 'cal_cm_448', 'cal_cm_ap_224', 'cal_cm_ap_448']

    df = df[df['setting'].isin(settings)]

    # sort
    df['setting_order'] = pd.Categorical(df['setting'], categories=settings, ordered=True)
    df = df.sort_values(by=['dataset_name', 'model_name', 'setting_order'], ascending=True)
    df = df.drop(columns=['setting_order'])

    return df


def process_acc_df(args):
    df = load_acc_df(args)

    if (not (args.type_plot in ('box', 'violin')) or any(['diff_is' in v for v in args.plot_vars])):
        df = df.groupby(['dataset_name', 'setting', 'model_name'], as_index=False).mean(numeric_only=True)

        if any(['diff_is' in v for v in args.plot_vars]):
            df = diff_384_224(df, 'acc')
        if any(['diff_setting' in v for v in args.plot_vars]):
            df = diff_setting(df, 'acc')

    return df


def load_cost_df(args):
    df = pd.read_csv(args.input_file_cost, delimiter=',')
    # filter by hostname and image size
    df = df[df['Hostname'] == args.host]

    df = df.drop(columns=['acc', 'diff_is_acc', 'diff_setting_acc', 'acc_median', 'acc_std'])
    return df


# ric_to_model = {
#     -3: "Swin B (IN21k)",
#     -2: "Swin B (IN21k)",
#     -1: "Swin B (IN21k)",
#      0: "Swin B (IN21k)",
#      1: "Swin B (IN21k)",
# }

def make_plot(args, df):
    # Seaborn Style Settings
    sns.set_theme(
        context=args.context, style=args.style, palette=args.palette,
        font=args.font_family, font_scale=args.font_scale, rc={
            "font.serif": ["Times New Roman"],
            "grid.linewidth": args.bg_line_width, # mod any of matplotlib rc system
            "figure.figsize": args.fig_size,
        })
    # print(list(df.columns))
    # print(list(df['Model'].unique()))
    # df = df[df['Model'].eq(df['OOM for Ratio of Images per Class^2'].map(ric_to_model))]
    if args.choose_best_ric:
        ric_col = 'OOM for Ratio of Images per Class^2'
        acc_col = 'Accuracy'
        model_col = 'Model'

        # Find best model per RIC bin
        best_models = (
            df.loc[
                df.groupby(ric_col)[acc_col].idxmax(),
                [ric_col, model_col]
            ]
        )

        df = df.merge(best_models, on=[ric_col, model_col], how='inner')


    if args.type_plot == 'bar':
        ax = sns.barplot(x=args.x_var_name, y=args.y_var_name, hue=args.hue_var_name, data=df, errorbar=None)
    elif args.type_plot == 'box':
        ax = sns.boxplot(x=args.x_var_name, y=args.y_var_name, hue=args.hue_var_name, data=df)
    elif args.type_plot == 'violin':
        ax = sns.violinplot(x=args.x_var_name, y=args.y_var_name, hue=args.hue_var_name, data=df)
    elif args.type_plot == 'line':
        ax = sns.lineplot(x=args.x_var_name, y=args.y_var_name, marker=args.marker,
                          hue=args.hue_var_name, style=args.style_var_name,
                          markers=True, linewidth=args.line_width, data=df)
    elif args.type_plot == 'scatter':
        ax = sns.scatterplot(x=args.x_var_name, y=args.y_var_name, hue=args.hue_var_name,
                             style=args.style_var_name, size=args.size_var_name,
                             sizes=tuple(args.sizes), legend='brief', data=df)
    elif args.type_plot == 'histogram':
        ax = sns.histplot(x=args.x_var_name, bins=args.bins, data=df)
    else:
        raise NotImplementedError

    # Remove top, right border by default
    if hasattr(args, 'despine') and args.despine:
        sns.despine(top=True, right=True, left=False, bottom=False)

    # labels and title
    ax.set(xlabel=args.x_label, ylabel=args.y_label, title=args.title, ylim=args.y_lim)

    if args.log_scale_x:
        ax.set_xscale('log')
    if args.log_scale_y:
        ax.set_yscale('log')

    # ticks labels
    if args.xticks_labels:
        x_ticks = ax.get_xticks() if getattr(args, 'x_ticks', None) is None else args.x_ticks
        ax.set_xticks(x_ticks, labels=args.xticks_labels)

    # Rotate x-axis or y-axis ticks lables
    if (args.x_rotation != None):
        plt.xticks(rotation = args.x_rotation)
    if (args.y_rotation != None):
        plt.yticks(rotation = args.y_rotation)

    # Change location of legend
    if args.hue_var_name:
        sns.move_legend(ax, loc=args.loc_legend)

    # save plot
    output_file = os.path.join(args.results_dir, f'{args.output_file}.{args.save_format}')
    plt.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
    print('Save plot to directory ', output_file)

    plt.clf()

    return 0


def parse_args():
    parser = argparse.ArgumentParser(parents=[input_output_args(), filtering_args(), plot_args()])

    parser.set_defaults(
        output_file='box_acc_models', results_dir=os.path.join('results_all', 'plots'),
    )

    args= parser.parse_args()

    for k in vars(args):
        v = getattr(args, k)
        if not v or v == '' or v == 'none' or v == 'None':
            setattr(args, k, None)

    if args.color:
        args.palette = [args.color for _ in range(100)]

    args.plot_vars = [v for v in (args.x_var_name, args.y_var_name, args.hue_var_name) if isinstance(v, str)]
    if (any(['diff_is' in v for v in args.plot_vars]) or args.image_size == 384):
        args.filter_og_only = False

    return args


def merge_df_rename_vars(args):
    df_acc = process_acc_df(args)
    df_cost = load_cost_df(args)
    print(df_cost['model_name'].unique())

    df_stats_data = pd.read_csv(args.input_file_stats_data)
    df_stats_models = pd.read_csv(args.input_file_stats_models)

    df = pd.merge(df_acc, df_cost, on=['model_name', 'setting'])
    df = pd.merge(df, df_stats_data, on=['dataset_name'])
    df = pd.merge(df, df_stats_models, on=['model_name'])

    df = filter_df(args, df)
    print(df['dataset_name'].unique())

    df = rename_vars(df, args)

    return df


def main():
    args = parse_args()

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    df = merge_df_rename_vars(args)

    make_plot(args, df)

    return 0


if __name__ == '__main__':
    main()