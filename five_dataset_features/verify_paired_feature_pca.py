"""Verify the saved paired-feature PCA tables, plots, and input identities."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat

from .analysis_snapshot import sha256
from .paired_feature_pca import MODULE, QUANTIZED, binary_auc


def verify(output):
    config = json.loads((output / 'run_config.json').read_text(encoding='utf-8'))
    input_root = Path(config['input_root'])
    analysis_root = Path(config['analysis_root'])
    for name, expected in config['code_sha256'].items():
        assert sha256(MODULE / name) == expected, name
    for dataset, hashes in config['source_sha256'].items():
        for name, expected in hashes.items():
            path = (analysis_root / dataset / name if name in
                    ('analysis_snapshot.json', 'fp_pca.npz') else
                    input_root / dataset / name)
            assert sha256(path) == expected, str(path)
    for name, expected in config['output_sha256'].items():
        assert sha256(output / name) == expected, name

    summary = pd.read_csv(output / 'summary.csv')
    samples = pd.read_csv(output / 'samples.csv')
    correlations = pd.read_csv(output / 'cross_dataset_spearman.csv')
    assert len(summary) == len(config['datasets']) * len(QUANTIZED)
    assert not summary.duplicated(['dataset', 'condition']).any()
    assert set(summary.condition) == set(QUANTIZED)
    assert not samples.duplicated(['dataset', 'condition', 'sample_id']).any()
    checked = 0
    for dataset in config['datasets']:
        meta = json.loads((input_root / dataset / 'dataset.json').read_text(encoding='utf-8'))
        with np.load(analysis_root / dataset / 'fp_pca.npz', allow_pickle=False) as pca:
            explained = float(pca['explained'].sum())
        for condition in QUANTIZED:
            row = summary[(summary.dataset == dataset) &
                          (summary.condition == condition)].iloc[0]
            part = samples[(samples.dataset == dataset) &
                           (samples.condition == condition)]
            assert len(part) == row.n_samples == meta['n_samples']
            assert part.sample_id.tolist() == meta['sample_ids']
            np.testing.assert_array_equal(part.target.to_numpy(), meta['targets'])
            np.testing.assert_allclose(row.pca_fp_variance_2d, explained, atol=1e-12)
            fp_correct = part.fp_correct.to_numpy(dtype=bool)
            q_correct = part.q_correct.to_numpy(dtype=bool)
            damage = fp_correct & ~q_correct
            rescue = ~fp_correct & q_correct
            assert int(damage.sum()) == row.n_damage == row.damage_count
            assert int(rescue.sum()) == row.n_rescue == row.rescue_count
            assert int(fp_correct.sum()) == row.n_fp_correct
            np.testing.assert_allclose(row.fp_accuracy, fp_correct.mean(), atol=1e-12)
            np.testing.assert_allclose(row.accuracy, q_correct.mean(), atol=1e-12)
            np.testing.assert_allclose(row.relative_loss,
                                       1 - q_correct.mean() / fp_correct.mean(), atol=1e-12)
            np.testing.assert_allclose(row.accuracy_drop,
                                       fp_correct.mean() - q_correct.mean(), atol=1e-12)
            projected = np.hypot(part.q_pc1 - part.fp_pc1,
                                 part.q_pc2 - part.fp_pc2)
            np.testing.assert_allclose(part.projected_2d_drift, projected,
                                       rtol=1e-7, atol=1e-7)
            assert (part.projected_2d_drift <= part.full_drift + 1e-7).all()
            capture = np.square(part.projected_2d_drift).sum() / np.square(part.full_drift).sum()
            np.testing.assert_allclose(row.drift_energy_captured_2d, capture,
                                       rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(row.median_relative_drift,
                                       part.relative_full_drift.median(), atol=1e-12)
            np.testing.assert_allclose(row.median_projected_drift,
                                       part.projected_2d_drift.median(), atol=1e-12)
            np.testing.assert_allclose(row.drift_damage_auc,
                                       binary_auc(damage[fp_correct],
                                                  part.loc[fp_correct, 'relative_full_drift']),
                                       atol=1e-12)
            np.testing.assert_allclose(row.fp_margin_damage_auc,
                                       binary_auc(damage[fp_correct],
                                                  -part.loc[fp_correct, 'fp_normalized_margin']),
                                       atol=1e-12)
            checked += 1

    for row in correlations.itertuples():
        panel = summary[summary.condition == row.condition]
        expected = panel[row.predictor].rank().corr(panel.relative_loss.rank())
        np.testing.assert_allclose(row.spearman_rho, expected, atol=1e-12)
        assert row.n_datasets == len(panel)

    plots = sorted((output / 'plots').glob('*.png'))
    assert len(plots) == len(config['datasets']) + 1
    for path in plots:
        with Image.open(path) as image:
            assert image.width >= 800 and image.height >= 400
            assert max(ImageStat.Stat(image.convert('RGB')).var) > 100
    report = (output / 'REPORT.md').read_text(encoding='utf-8')
    for image in re.findall(r'!\[[^]]*\]\(([^)]+)\)', report):
        assert (output / image).is_file(), image
    result = dict(status='passed', dataset_condition_pairs_checked=checked,
                  sample_rows_checked=len(samples), figures_checked=len(plots),
                  source_and_output_hashes_verified=True,
                  note='Figure integrity is automated; visual layout was inspected separately.')
    (output / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=MODULE / 'analysis/paired_feature_pca')
    print(json.dumps(verify(parser.parse_args().output_dir.resolve()), indent=2))
