"""Verify saved artifacts against their inputs, including independent SciPy rho."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from scipy.stats import spearmanr


def verify(root):
    config = json.loads((root / 'run_config.json').read_text(encoding='utf-8'))
    source = Path(config['input_file'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == config['input_sha256']
    script = Path(__file__).with_name('analyze.py')
    assert hashlib.sha256(script.read_bytes()).hexdigest() == config['script_sha256']
    cells = pd.read_csv(root / 'cells.csv')
    excluded = pd.read_csv(root / 'excluded_cells.csv')
    summary = pd.read_csv(source)
    assert len(cells) + len(excluded) == len(summary)
    assert cells['baseline_matched'].all()
    assert not cells.duplicated(['dataset_name', 'model_name', 'ckpt_kind',
                                 'hqq_nbits', 'hqq_group_size']).any()
    np.testing.assert_allclose(cells.accuracy_drop_points, cells.fp32_top1 - cells.hqq_top1)
    single = cells.n_seeds.eq(1)
    np.testing.assert_allclose(cells.loc[single, 'accuracy_ratio'],
                               cells.loc[single, 'hqq_top1'] / cells.loc[single, 'fp32_top1'])
    coverage = pd.read_csv(root / 'coverage.csv')
    assert (coverage.observed_conditions <= coverage.expected_conditions).all()
    assert (coverage.included == coverage.observed_conditions.eq(coverage.expected_conditions)).all()
    assert set(coverage.scope) == {'primary', 'without_2bit', 'full_grid'}

    scopes = {'primary': [2., 3., 4.], 'without_2bit': [3., 4.],
              'full_grid': sorted(cells.hqq_nbits.unique())}
    panels = {}
    for scope, bits in scopes.items():
        for context, frame in cells.groupby(['model_name', 'ckpt_kind']):
            ids = coverage[(coverage.scope == scope) &
                           (coverage.model_name == context[0]) &
                           (coverage.ckpt_kind == context[1]) & coverage.included].dataset_name
            frame = frame[frame.dataset_name.isin(ids) & frame.hqq_nbits.isin(bits)]
            panels[(scope, *context)] = frame.pivot(index='dataset_name',
                columns=['hqq_nbits', 'hqq_group_size'], values='accuracy_ratio')
    within = pd.read_csv(root / 'within_context_agreement.csv')
    for row in within.itertuples():
        panel = panels[(row.scope, row.model_name, row.ckpt_kind)]
        assert len(panel) == row.n_datasets
        left, right = panel[(row.bit_a, row.group_a)], panel[(row.bit_b, row.group_b)]
        expected = (spearmanr(left, right).statistic
                    if len(panel) >= 3 and left.nunique() > 1 and right.nunique() > 1 else np.nan)
        np.testing.assert_allclose(row.rho, expected, atol=1e-12, equal_nan=True)
    cross = pd.read_csv(root / 'cross_context_agreement.csv')
    cross_panels = {setting: frame.pivot(index='dataset_name', columns='context',
                                        values='accuracy_ratio').dropna()
                    for setting, frame in cells.groupby(['hqq_nbits', 'hqq_group_size'])}
    for row in cross.itertuples():
        panel = cross_panels[(row.hqq_nbits, row.hqq_group_size)]
        assert len(panel) == row.n_datasets
        left, right = panel[row.context_a], panel[row.context_b]
        expected = (spearmanr(left, right).statistic
                    if len(panel) >= 3 and left.nunique() > 1 and right.nunique() > 1 else np.nan)
        np.testing.assert_allclose(row.rho, expected, atol=1e-12, equal_nan=True)
    orders = pd.read_csv(root / 'pairwise_order_stability.csv')
    assert (orders.a_higher + orders.b_higher + orders.ties == orders.n_conditions).all()
    assert (orders.order_reverses == (orders.a_higher.gt(0) & orders.b_higher.gt(0))).all()

    legacy = source.parent
    completeness = pd.read_csv(legacy / 'completeness.csv')
    assert completeness.observed_hqq_conditions.sum() == len(summary)
    assert completeness.unmatched_conditions.sum() == (~summary.baseline_matched).sum()
    assert (legacy / 'configuration_quality.csv').is_file()
    workbooks = list(legacy.glob('quant_*.xlsx'))
    assert len(workbooks) == summary.dataset_name.nunique()
    for dataset in summary.dataset_name.unique():
        workbook = pd.read_excel(legacy / f'quant_{dataset}.xlsx')
        assert len(workbook) == summary.dataset_name.eq(dataset).sum()
        assert 'accuracy_ratio' in workbook and 'baseline_matched' in workbook
        assert not any('time' in name or 'loss' in name or 'top5' in name for name in workbook.columns)
    assert not list((legacy / 'corr').rglob('*.png'))
    for name in ['overview.png', 'factor_contribution_accuracy.png',
                 'quant_acc_by_bits_group_cal.png', 'accuracy_heatmap_cal.png']:
        assert (legacy / 'plots' / name).is_file(), name
    manifest = pd.read_csv(root / 'plots_manifest.csv')
    assert not manifest.file.duplicated().any()
    n_contexts = cells[['model_name', 'ckpt_kind']].drop_duplicates().shape[0]
    assert len(manifest) == 3 * n_contexts + 2
    for row in manifest.itertuples():
        assert (root / row.source_table).is_file()
        with Image.open(root / row.file) as image:
            assert image.width >= 800 and image.height >= 400
            assert max(ImageStat.Stat(image.convert('RGB')).var) > 100
    result = dict(status='passed', cells=len(cells), exclusions=len(excluded),
                  within_correlations_checked=len(within), cross_correlations_checked=len(cross),
                  pairwise_conservation_checks=len(orders), workbook_checks=len(workbooks),
                  figure_integrity_checks=len(manifest),
                  source_and_script_hashes_verified=True,
                  note='Figure integrity is automated; visual layout was inspected separately.')
    (root / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'results_all/dataset_robustness')
    print(json.dumps(verify(parser.parse_args().results_dir.resolve()), indent=2))
