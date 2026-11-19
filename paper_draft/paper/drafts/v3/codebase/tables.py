"""Write the LaTeX macros and tables from the build/ tables into tables/."""
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

from registry import BUILD, DATASET_LABELS, DATASETS, GROUP_LABELS, MODEL_LABELS, MODE_LABELS, PAPER, group_of

TABLES = PAPER / 'tables'


def fmt(value, dp):
    text = str(Decimal(str(round(float(value), 6))).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP))
    if text.startswith('-'):
        return '$-$' + text[1:]
    return text


def thousands(value):
    return f'{int(value):,}'.replace(',', '{,}')


def setting_label(model, mode):
    return f'{MODEL_LABELS[model]} & {MODE_LABELS[mode]}'


def write_macros():
    numbers = pd.read_csv(BUILD / 'numbers.csv')
    lines = []
    for row in numbers.itertuples():
        text = thousands(row.value) if row.dp == 0 and abs(row.value) >= 10000 else fmt(row.value, row.dp)
        lines.append(f'\\expandafter\\def\\csname V@{row.key}\\endcsname{{{text}}}')
    (TABLES / 'macros.tex').write_text('\n'.join(lines) + '\n')


def write_datasets():
    table = pd.read_csv(BUILD / 'datasets.csv', index_col=0)
    rows = []
    for dataset in DATASETS:
        row = table.loc[dataset]
        rows.append(f'{DATASET_LABELS[dataset]} & {GROUP_LABELS[group_of(dataset)]} & {thousands(row.n)} & '
                    f'{thousands(row.classes)} & {fmt(row.fp_accuracy, 1)} \\\\')
    body = '\n'.join(rows)
    (TABLES / 'datasets.tex').write_text(f"""\\begin{{table}}[t]
\\centering
\\caption{{Datasets, their group, test-set size and number of classes $C$. FP is the baseline top-1, the mean over the nine backbone and fine-tuning strategy settings.}}
\\label{{tab:datasets}}
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\small
\\begin{{tabular}}{{llrrr}}
\\toprule
Dataset & Group & Test $N$ & $C$ & FP (\\%) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def write_w3_summary():
    table = pd.read_csv(BUILD / 'w3_summary.csv')
    rows = []
    for row in table.itertuples():
        rows.append(f'{setting_label(row.model, row.mode)} & {fmt(row.fp_accuracy, 1)} & {fmt(row.accuracy, 1)} & '
                    f'{fmt(row.relative_loss, 1)} & {fmt(row.swap_rate, 1)} & {fmt(row.damage_rate, 1)} & '
                    f'{fmt(row.rescue_rate, 1)} \\\\')
    body = '\n'.join(rows)
    (TABLES / 'w3_summary.tex').write_text(f"""\\begin{{table}}[t]
\\centering
\\caption{{W3/G128 results over the {{\\V{{nDatasets}}}} datasets. FP and Q: baseline and quantized top-1. Accuracies, swap, damage and rescue rates are dataset means in \\%; the relative loss is the dataset median.}}
\\label{{tab:w3}}
\\setlength{{\\tabcolsep}}{{2.2pt}}
\\small
\\begin{{tabular}}{{llrrrrrr}}
\\toprule
Backbone & Strategy & FP & Q & Loss $\\downarrow$ & Swap & Damage & Rescue \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")
    rows = []
    for row in table.itertuples():
        rows.append(f'{setting_label(row.model, row.mode)} & {fmt(row.final_cka, 2)} & {fmt(row.bias_gain, 2)} & '
                    f'{fmt(row.shuffle_gain, 2)} & {fmt(1000 * row.delta_brier, 2)} \\\\')
    body = '\n'.join(rows)
    (TABLES / 'w3_diagnostics.tex').write_text(f"""\\begin{{table}}[h]
\\centering
\\caption{{W3/G128 diagnostics over the {{\\V{{nDatasets}}}} datasets. CKA: median final-layer CKA. Offset: mean top-1 change in points after the cross-fitted class-offset correction, beside the mean of five shuffled-class controls. $\\Delta$Brier ($\\times 10^{{3}}$): change in damage Brier score from adding competitor counts.}}
\\label{{tab:diagnostics}}
\\setlength{{\\tabcolsep}}{{3pt}}
\\small
\\begin{{tabular}}{{llrrrr}}
\\toprule
Backbone & Strategy & CKA $\\uparrow$ & Offset & Shuffled & $\\Delta$Brier $\\downarrow$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def write_margin():
    table = pd.read_csv(BUILD / 'margin_table.csv')
    rows = []
    for row in table.itertuples():
        rows.append(f'{setting_label(row.model, row.mode)} & {fmt(row.raw, 3)} & {fmt(row.matched, 3)} & '
                    f'{fmt(row.reduction, 1)} & {fmt(row.overlap, 2)} & {fmt(100 * row.ess_ratio, 1)} \\\\')
    body = '\n'.join(rows)
    (TABLES / 'margin.tex').write_text(f"""\\begin{{table}}[t]
\\centering
\\caption{{Retention gap (FG $-$ UFG) at W3/G128 before and after matching the baseline margin distributions, bin width 0.05. Red.: reduction of the gap. Overlap is the shared histogram mass; ESS is the lower of the two groups' matched-to-raw effective sample size.}}
\\label{{tab:margin}}
\\setlength{{\\tabcolsep}}{{1.6pt}}
\\small
\\begin{{tabular}}{{llrrrrr}}
\\toprule
Backbone & Strategy & Raw & Matched & Red. (\\%) & Overlap & ESS (\\%) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def write_gates():
    table = pd.read_csv(BUILD / 'difficulty_gates.csv')
    rows = []
    for row in table.itertuples():
        p = '$<$0.01' if row.p_margin < 0.01 else fmt(row.p_margin, 2)
        rows.append(f'{setting_label(row.model, row.mode)} & {fmt(row.w, 2)} & {fmt(row.rho_margin, 2)} & {p} & '
                    f'{fmt(row.rho_fp_accuracy, 2)} & {fmt(1000 * row.err_controls, 1)} & '
                    f'{fmt(1000 * row.err_margin, 1)} \\\\')
    body = '\n'.join(rows)
    (TABLES / 'gates.tex').write_text(f"""\\begin{{table}}[t]
\\centering
\\caption{{Difficulty score of the {{\\V{{gridDatasets}}}} datasets. $W$: agreement of the four primary conditions. $\\rho$: Spearman correlation of the score with the baseline margin $q_{{10}}$ (one-sided permutation $p$) and with FP top-1. Err: leave-one-dataset-out squared error ($\\times 10^{{3}}$) of the controls (Ctrl.) alone and with $q_{{10}}$ added.}}
\\label{{tab:gates}}
\\setlength{{\\tabcolsep}}{{1.8pt}}
\\small
\\begin{{tabular}}{{llrrrrrr}}
\\toprule
 & & & \\multicolumn{{2}}{{c}}{{Margin $q_{{10}}$}} & FP top-1 & \\multicolumn{{2}}{{c}}{{Err $\\downarrow$}} \\\\
\\cmidrule(lr){{4-5}}\\cmidrule(lr){{7-8}}
Backbone & Strategy & $W$ $\\uparrow$ & $\\rho$ & $p$ & $\\rho$ & Ctrl. & +$q_{{10}}$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def write_gates_short():
    table = pd.read_csv(BUILD / 'difficulty_gates.csv')
    rows = [f'{setting_label(row.model, row.mode)} & {fmt(row.w, 2)} & {fmt(row.rho_margin, 2)} & '
            f'{fmt(row.rho_fp_accuracy, 2)} \\\\' for row in table.itertuples()]
    body = '\n'.join(rows)
    (TABLES / 'gates_short.tex').write_text(f"""\\begin{{table}}[t]
\\centering
\\caption{{Agreement $W$ of the four primary conditions on the ranking of the {{\\V{{gridDatasets}}}} datasets, and Spearman $\\rho$ of the difficulty score with the baseline margin $q_{{10}}$ and with baseline top-1.}}
\\label{{tab:gatesShort}}
\\setlength{{\\tabcolsep}}{{3pt}}
\\small
\\begin{{tabular}}{{llrrr}}
\\toprule
Backbone & Strategy & $W$ $\\uparrow$ & $\\rho$ ($q_{{10}}$) & $\\rho$ (top-1) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def write_properties():
    table = pd.read_csv(BUILD / 'difficulty_gates.csv')
    columns = [('fp_normalized_margin_q10', '$q_{10}$'), ('fp_accuracy', 'FP'),
               ('num_classes', '$C$'), ('fp_separability', 'Sep.'), ('fp_h_score_final', 'H'),
               ('fp_norm_p99_p50', 'Norm'), ('final_cka', 'CKA')]
    rows = []
    for row in table.itertuples():
        cells = ' & '.join(fmt(getattr(row, f'rho_{c}'), 2) for c, _ in columns)
        rows.append(f'{setting_label(row.model, row.mode)} & {cells} \\\\')
    body = '\n'.join(rows)
    header = ' & '.join(label for _, label in columns)
    (TABLES / 'properties.tex').write_text(f"""\\begin{{table}}[h]
\\centering
\\caption{{Spearman correlation of each property with the difficulty score over the {{\\V{{gridDatasets}}}} datasets. FP: baseline top-1. Sep.: separability. H: H-score. Norm: feature norm ratio P99/P50. All are measured on the baseline except the final-layer CKA between baseline and W3/G128 features.}}
\\label{{tab:properties}}
\\setlength{{\\tabcolsep}}{{1.5pt}}
\\footnotesize
\\begin{{tabular}}{{llrrrrrrr}}
\\toprule
Backbone & Strategy & {header} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def write_robustness():
    table = pd.read_csv(BUILD / 'robustness.csv')
    rows = [f'{row.check} & {fmt(row.mean, 2)} & {fmt(row.min, 2)} & {int(row.positive)} \\\\'
            for row in table.itertuples()]
    body = '\n'.join(rows)
    (TABLES / 'robustness.tex').write_text(f"""\\begin{{table}}[t]
\\centering
\\caption{{Retention gap (FG $-$ UFG) at W3/G128 under alternative definitions: the mean and minimum over the nine settings, and the settings with a positive gap. Chance correction uses $(a_{{\\mathrm{{Q}}}}-1/C)/(a_{{\\mathrm{{FP}}}}-1/C)$. The worst leave-one-out row removes the dataset that lowers the mean gap most.}}
\\label{{tab:robust}}
\\setlength{{\\tabcolsep}}{{3pt}}
\\small
\\begin{{tabular}}{{lrrr}}
\\toprule
Check & Mean & Min & Positive (of 9) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def main():
    TABLES.mkdir(exist_ok=True)
    write_macros()
    write_datasets()
    write_w3_summary()
    write_margin()
    write_gates()
    write_gates_short()
    write_properties()
    write_robustness()


if __name__ == '__main__':
    main()
