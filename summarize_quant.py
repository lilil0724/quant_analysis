import os
import sys
import argparse

import numpy as np
import pandas as pd

from utils import rename_swin, highlight_top_k


# Quant eval runs come from the sibling TGDA project (eval_quant.py) and are
# logged to the WandB project nycu_pcs/TiT. Download the 4a0c scan + 4999 fp32
# baseline with the serials list baked into --main_serials (see parse_args), or:
#   python download_save_wandb_data.py --quant --serials 4000 4001 ... 4703 4999
DATASETS_REDUCED = ['aircraft', 'cub', 'soygene', 'soylocal']

# Native low-bit ViT_integration models evaluated WITHOUT HQQ (their own
# bundled quantization), per the eval_quant.py docstring.
QUANT_MODELS_NATIVE = ['BHViT', 'BinaryViT', 'GSB', 'SI_BiViT', '158b_ViT']

# Metrics to compute diffs for and to highlight in the per-dataset xlsx.
BASE_METRICS = ['acc', 'vram_gb', 'params_m', 'time_total_s']


def _fmt_nbits(nbits):
    """Format nbits compactly: integer-valued floats drop the .0, fractional
       floats keep their value (e.g. 1.58 -> '1.58', 4.0 -> '4')."""
    if nbits is None or (isinstance(nbits, float) and np.isnan(nbits)):
        return 'na'
    try:
        if float(nbits).is_integer():
            return str(int(nbits))
        return str(nbits)
    except (TypeError, ValueError):
        return str(nbits)


def make_quant_setting(hqq, hqq_nbits, hqq_group_size, hqq_quant_zero,
                        hqq_quant_scale, hqq_offload_meta, model_name):
    """Build a categorical key per row, analog of `add_setting` for acc runs."""
    if hqq is True or (isinstance(hqq, str) and hqq.lower() == 'true'):
        try:
            s = f'hqq_n{_fmt_nbits(hqq_nbits)}_g{int(hqq_group_size)}'
        except (TypeError, ValueError):
            s = 'hqq'
        # Distinguish non-default HQQ cfg flags so they don't collide with the
        # baseline (zeros/scales quantized, meta offloaded) configs.
        if hqq_quant_zero in (False, 'False'):
            s += '_nozero'
        if hqq_quant_scale in (False, 'False'):
            s += '_noscale'
        if hqq_offload_meta in (True, 'True'):
            s += '_offload'
        return s
    if model_name in QUANT_MODELS_NATIVE:
        return f'native_{model_name}'
    return 'fp32'


def _long_from_hqq(row):
    """Exploded HQQ-quantized row from an eval_quant run."""
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


def _long_from_fp32(row):
    """FP32 baseline row, only present when the run was launched with
       --hqq_compare_fp32 so that the original_* summary fields are populated."""
    if pd.isna(row.get('original_top1')):
        return None
    return {
        'dataset_name': row['dataset_name'],
        'model_name': row['model_name'],
        'seed': row['seed'],
        'serial': row['serial'],
        'image_size': row['image_size'],
        'hqq_nbits': np.nan,
        'hqq_group_size': np.nan,
        'hqq_quant_zero': np.nan,
        'hqq_quant_scale': np.nan,
        'hqq_offload_meta': np.nan,
        'hqq_compute_dtype': row['hqq_compute_dtype'],
        'quant_setting': 'fp32',
        'acc': row.get('original_top1'),
        'vram_gb': row.get('original_vram_gb'),
        'params_m': row.get('original_params_m'),
        'loss': np.nan,
        'time_total_s': row.get('time_total_s'),
        'kind': 'fp32',
    }


def _long_from_native(row):
    """Native low-bit model evaluated without HQQ (`--hqq` omitted on a
       ViT_integration quant model). The plain `top1/params_m/vram_gb` summary
       fields then carry the results."""
    if pd.isna(row.get('top1')):
        return None
    return {
        'dataset_name': row['dataset_name'],
        'model_name': row['model_name'],
        'seed': row['seed'],
        'serial': row['serial'],
        'image_size': row['image_size'],
        'hqq_nbits': np.nan,
        'hqq_group_size': np.nan,
        'hqq_quant_zero': np.nan,
        'hqq_quant_scale': np.nan,
        'hqq_offload_meta': np.nan,
        'hqq_compute_dtype': row['hqq_compute_dtype'],
        'quant_setting': f'native_{row["model_name"]}',
        'acc': row.get('top1'),
        'vram_gb': row.get('vram_gb'),
        'params_m': row.get('params_m'),
        'loss': row.get('loss'),
        'time_total_s': row.get('time_total_s'),
        'kind': 'native',
    }


def _long_from_plain_fp32(row):
    """Plain fp32 evaluation run (hqq omitted, model is NOT a ViT_integration
       native-quant model). e.g. the 4999 baseline serial. Mirrors
       `_long_from_native` but emits quant_setting='fp32' so it serves as the
       reference for `diff_fp32_*`."""
    if pd.isna(row.get('top1')):
        return None
    return {
        'dataset_name': row['dataset_name'],
        'model_name': row['model_name'],
        'seed': row['seed'],
        'serial': row['serial'],
        'image_size': row['image_size'],
        'hqq_nbits': np.nan,
        'hqq_group_size': np.nan,
        'hqq_quant_zero': np.nan,
        'hqq_quant_scale': np.nan,
        'hqq_offload_meta': np.nan,
        'hqq_compute_dtype': row['hqq_compute_dtype'],
        'quant_setting': 'fp32',
        'acc': row.get('top1'),
        'vram_gb': row.get('vram_gb'),
        'params_m': row.get('params_m'),
        'loss': row.get('loss'),
        'time_total_s': row.get('time_total_s'),
        'kind': 'fp32',
    }


def load_quant_df(args):
    """Read the downloaded quant CSV and explode each run into long-format
       rows (one per quant_setting). Mirrors `load_acc_df` in summarize_acc."""
    df = pd.read_csv(args.input_file_quant)

    if args.main_serials:
        df = df[df['serial'].isin(args.main_serials)]

    if args.subset_datasets:
        df = df[df['dataset_name'].isin(args.subset_datasets)]
    if args.subset_models:
        df = df[df['model_name'].isin(args.subset_models)]

    # Explode each run into the long format.
    rows = []
    for _, row in df.iterrows():
        is_hqq = row['hqq'] in (True, 'True', 'true', 1)
        if is_hqq:
            rows.append(_long_from_hqq(row))
            # If the run also logged the fp32 baseline (--hqq_compare_fp32),
            # add it as a separate row so it can serve as the diff reference.
            fp_row = _long_from_fp32(row)
            if fp_row is not None:
                rows.append(fp_row)
        elif row['model_name'] in QUANT_MODELS_NATIVE:
            nat_row = _long_from_native(row)
            if nat_row is not None:
                rows.append(nat_row)
        else:
            # Plain non-HQQ, non-native run: the 4999 fp32 baseline path.
            rows.append(_long_from_plain_fp32(row))

    long = pd.DataFrame([r for r in rows if r is not None])

    long['model_name'] = long['model_name'].apply(rename_swin)

    # Optional dataset stats merge (e.g. group/category columns).
    if args.input_file_stats_data and os.path.exists(args.input_file_stats_data):
        df_stats = pd.read_csv(args.input_file_stats_data)
        long = long.merge(df_stats, on='dataset_name', how='left')

    return long


def filter_df(args, df):
    if args.subset_datasets:
        df = df[df['dataset_name'].isin(args.subset_datasets)]
    if args.subset_models:
        df = df[df['model_name'].isin(args.subset_models)]
    if args.subset_quant_settings:
        df = df[df['quant_setting'].isin(args.subset_quant_settings)]

    # Stable categorical ordering for downstream xlsx/plot consumers.
    # Note: drop any residual Categorical dtype first ??otherwise re-constructing
    # a Categorical over a column that still carries a stale dtype/categories
    # trips a Pandas deprecation warning even when no values are dropped.
    df['quant_setting'] = df['quant_setting'].astype(str)
    cats = _ordered_quant_settings(df)
    if len(cats) > 0:
        df['quant_setting'] = pd.Categorical(df['quant_setting'], categories=cats, ordered=True)
        df = df.sort_values(by=['dataset_name', 'quant_setting', 'model_name'], ascending=True)
    else:
        df = df.sort_values(by=['dataset_name', 'quant_setting', 'model_name'], ascending=True)

    return df


def _ordered_quant_settings(df):
    """fp32 first, then native_<model> (sorted), then hqq_n{nbits}_g{gs}
       ordered by nbits desc then group_size asc."""
    settings = set(df['quant_setting'].dropna().unique())
    fp32 = ['fp32'] if 'fp32' in settings else []
    native = sorted([s for s in settings if s.startswith('native_') and s != 'fp32'])

    hqq_codes = []
    for s in settings:
        if s.startswith('hqq_'):
            try:
                # parse the n<nbits> and g<group_size> tokens of an
                # hqq_n{nb}_g{gs}[_<flags>] quant_setting key. nbits may be a
                # float (e.g. 1.58), so accept ints and floats alike.
                tokens = s.split('_')
                nb = gs = None
                for tok in tokens:
                    if tok.startswith('n'):
                        try:
                            nb = float(tok[1:])
                            if nb.is_integer():
                                nb = int(nb)
                        except ValueError:
                            pass
                    elif tok.startswith('g'):
                        try:
                            gs = int(tok[1:])
                        except ValueError:
                            pass
                if nb is not None:
                    hqq_codes.append((nb, gs if gs is not None else -1, s))
            except (ValueError, IndexError):
                continue
    hqq_codes.sort(key=lambda t: (-t[0], t[1]))
    hqq = [c[2] for c in hqq_codes]
    return fp32 + native + hqq


def aggregate_results(df, args):
    g = ['dataset_name', 'quant_setting', 'model_name']
    df_std = df.groupby(g, as_index=False).std(numeric_only=True)
    df_median = df.groupby(g, as_index=False).median(numeric_only=True)
    df = df.groupby(g, as_index=False).mean(numeric_only=True)
    df['acc_median'] = df_median['acc']
    df['acc_std'] = df_std['acc']

    for col in BASE_METRICS:
        df = diff_fp32(df, col)
        df = diff_nbits(df, col)
        df = diff_group_size(df, col)

    df = filter_df(args, df)
    return df


def diff_fp32(df, col_name):
    """% change vs the fp32 baseline (same dataset+model).
       fp32 rows themselves get NaN (they ARE the baseline)."""
    diff_name = f'diff_fp32_{col_name}'
    df[diff_name] = np.nan

    fp32_mask = df['quant_setting'] == 'fp32'
    ref_cols = ['dataset_name', 'model_name', col_name]
    ref = df.loc[fp32_mask, ref_cols].rename(columns={col_name: f'{col_name}_ref'})

    if ref.empty:
        return df

    merged = df.merge(ref, on=['dataset_name', 'model_name'], how='left')
    ref_val = merged[f'{col_name}_ref']
    row_val = merged[col_name]
    with np.errstate(divide='ignore', invalid='ignore'):
        df[diff_name] = np.where(
            fp32_mask | ref_val.isna() | row_val.isna() | (ref_val == 0),
            np.nan,
            100.0 * (row_val - ref_val) / ref_val,
        )
    return df


def _diff_within_hqq(df, col_name, ref_key, label):
    """% change of `col_name` vs the hqq row whose `ref_key` is the max within
       the same (dataset, model, other quant dimension). The row selected as
       ref gets NaN (it IS the baseline).
       ref_key is 'hqq_nbits' or 'hqq_group_size'."""
    diff_name = f'diff_{label}_{col_name}'
    df[diff_name] = np.nan

    hqq_mask = df['hqq_nbits'].notna()
    if not hqq_mask.any():
        return df

    other_key = 'hqq_group_size' if ref_key == 'hqq_nbits' else 'hqq_nbits'

    ref_vals = {}
    for _, row in df.loc[hqq_mask].iterrows():
        sub = df[(df['dataset_name'] == row['dataset_name']) &
                 (df['model_name'] == row['model_name']) &
                 (df[other_key] == row[other_key]) &
                 hqq_mask]
        # Need >= 2 distinct ref_key values to make a difference meaningful.
        if sub[ref_key].nunique() < 2:
            continue
        max_key = sub[ref_key].max()
        if row[ref_key] == max_key:
            continue
        r = sub[sub[ref_key] == max_key]
        if len(r) == 0 or pd.isna(r[col_name].iloc[0]) or r[col_name].iloc[0] == 0:
            continue
        ref_vals[(row.name,)] = r[col_name].iloc[0]

    for (idx,), ref_val in ref_vals.items():
        row_val = df.at[idx, col_name]
        if pd.notna(row_val):
            df.at[idx, diff_name] = 100.0 * (row_val - ref_val) / ref_val

    return df


def diff_nbits(df, col_name):
    """Reference = max(hqq_nbits) at the same (dataset, model, group_size).
       Only meaningful when >= 2 nbits levels exist for that group_size."""
    return _diff_within_hqq(df, col_name, 'hqq_nbits', 'nbits')


def diff_group_size(df, col_name):
    """Reference = max(hqq_group_size) at the same (dataset, model, nbits).
       Only meaningful when >= 2 group_size levels exist for that nbits."""
    return _diff_within_hqq(df, col_name, 'hqq_group_size', 'group_size')


def _metric_cols(df):
    cols = []
    for c in df.columns:
        if c.startswith('diff_'):
            cols.append(c)
        elif c in BASE_METRICS:
            cols.append(c)
        elif 'acc' in c:
            cols.append(c)
    return cols


def highlight_per_ds(df, args):
    metric_list = _metric_cols(df)
    if not metric_list:
        return 0

    for ds in df['dataset_name'].unique():
        df_ds = df[df['dataset_name'] == ds]
        df_ds.style.apply(
            highlight_top_k, largest=True, k=args.top_k, color='lightgreen',
            axis=0, subset=metric_list).apply(
            highlight_top_k, largest=False, k=args.top_k, color='lightcoral',
            axis=0, subset=metric_list).to_excel(
            f'{args.output_file}_{ds}.xlsx', header=True, index=False,
            engine='openpyxl')


def count_best_per_ds(df, args):
    metric_list = _metric_cols(df)
    if not metric_list:
        return 0

    df_best = pd.DataFrame()
    with open(f'{args.output_file}_counts.txt', 'w') as sys.stdout:
        for metric in metric_list:
            for ds in df['dataset_name'].unique():
                df_ds = df[df['dataset_name'] == ds]
                ascending = 'std' in metric or 'time' in metric
                df_ds_sorted = df_ds.sort_values(by=metric, ascending=ascending)
                df_ds_sorted = df_ds_sorted.iloc[:args.top_k]
                keep = ['dataset_name', 'quant_setting', 'model_name', metric]
                df_ds_sorted = df_ds_sorted[[c for c in keep if c in df_ds_sorted.columns]]
                df_best = pd.concat([df_best, df_ds_sorted], axis=0)

            print(metric)
            print(df_best.to_string())
            print(df_best[['quant_setting', 'model_name']].value_counts().to_string(), '\n')
            print(df_best.model_name.value_counts().to_string(), '\n')
            print(df_best.quant_setting.value_counts().to_string(), '\n\n')
            df_best = pd.DataFrame()


def summarize_results(args):
    df = load_quant_df(args)
    df = aggregate_results(df, args)
    df.to_csv(f'{args.output_file}.csv', header=True, index=False)
    highlight_per_ds(df, args)
    count_best_per_ds(df, args)
    return df


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--input_file_quant', type=str,
                        default=os.path.join('data', 'backbones_quant.csv'),
                        help='filename for input .csv file from wandb (TiT project)')
    parser.add_argument('--input_file_stats_data', type=str,
                        default=os.path.join('data', 'stats_datasets.csv'),
                        help='optional dataset stats csm to merge in')

    # filters
    # All TiT 4a0c serials: a = nbits code (0..7), c = group_size code (0..3),
    # plus 4999 = fp32 baseline. See AGENTS.md for the full mapping.
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

    # output
    parser.add_argument('--output_file', type=str, default='quant',
                        help='filename stem for output .csv/.xlsx files (no extension)')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'quant'),
                        help='the directory where results will be stored')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    args.output_file = os.path.join(args.results_dir, args.output_file)

    summarize_results(args)


if __name__ == '__main__':
    main()