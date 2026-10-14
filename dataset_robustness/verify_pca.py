"""Verify saved PCA outputs against their source and linear-algebra invariants."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat

from dataset_robustness.pca_profiles import ROOT, complete_matrix, load_verified_source


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(output):
    config = json.loads((output / 'run_config.json').read_text(encoding='utf-8'))
    main_results = Path(config['main_results'])
    assert sha256(main_results / 'run_config.json') == config['main_run_config_sha256']
    assert sha256(ROOT / 'analyze.py') == config['main_analyzer_sha256']
    assert sha256(ROOT / 'pca_profiles.py') == config['pca_script_sha256']
    assert sha256(Path(config['source'])) == config['source_sha256']
    for name, expected in config['output_sha256'].items():
        assert sha256(output / name) == expected, name
    for name, expected in config['figure_sha256'].items():
        assert sha256(output / 'plots' / name) == expected, name
    assert sha256(output / 'report.md') == config['report_sha256']

    cells, excluded, _, _ = load_verified_source(main_results)
    assert len(cells) == config['n_source_cells']
    assert len(excluded) == config['n_source_exclusions']
    summary = pd.read_csv(output / 'pca_summary.csv')
    scores = pd.read_csv(output / 'scores.csv')
    loadings = pd.read_csv(output / 'loadings.csv')
    variance = pd.read_csv(output / 'variance.csv')
    coverage = pd.read_csv(output / 'coverage.csv')
    assert len(summary) == config['n_fits']
    assert summary[['model_name', 'ckpt_kind']].drop_duplicates().shape[0] == config['n_contexts']
    checked = 0
    for row in summary.itertuples():
        identity = ((summary.model_name == row.model_name) &
                    (summary.ckpt_kind == row.ckpt_kind) &
                    (summary.scope == row.scope) & (summary.metric == row.metric))
        assert identity.sum() == 1
        bits = (config['primary_bits'] if row.scope == 'primary' else
                [bit for bit in config['primary_bits'] if bit != 2])
        frame = cells[(cells.model_name == row.model_name) &
                      (cells.ckpt_kind == row.ckpt_kind)]
        wide, expected_coverage = complete_matrix(
            frame, bits, config['group_sizes'], row.metric)
        part = lambda table: table[(table.model_name == row.model_name) &
                                    (table.ckpt_kind == row.ckpt_kind) &
                                    (table.scope == row.scope) &
                                    (table.metric == row.metric)]
        s = part(scores).set_index('dataset_name').loc[wide.index]
        l = part(loadings).set_index(['hqq_nbits', 'hqq_group_size']).loc[wide.columns]
        v = part(variance).sort_values('component')
        c = part(coverage).set_index('dataset_name').loc[expected_coverage.dataset_name]
        assert len(wide) == row.n_datasets and wide.shape[1] == row.n_conditions
        np.testing.assert_array_equal(c.included.to_numpy(), expected_coverage.included.to_numpy())
        np.testing.assert_array_equal(c.observed_conditions.to_numpy(),
                                      expected_coverage.observed_conditions.to_numpy())
        assert len(v) == min(wide.shape)
        np.testing.assert_allclose(v.variance_fraction.sum(), 1, atol=1e-12)
        np.testing.assert_allclose(v.variance_fraction.iloc[:2].sum(),
                                   row.pc1_pc2_variance_fraction, atol=1e-12)
        score = s[['pc1', 'pc2']].to_numpy()
        loading = l[['pc1_loading', 'pc2_loading']].to_numpy()
        np.testing.assert_allclose(score.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(loading.T @ loading, np.eye(2), atol=1e-12)
        centered = wide.to_numpy() - wide.to_numpy().mean(axis=0)
        reconstruction = score @ loading.T
        residual_fraction = np.square(centered - reconstruction).sum() / np.square(centered).sum()
        np.testing.assert_allclose(residual_fraction, 1 - row.pc1_pc2_variance_fraction,
                                   atol=1e-11)
        eigenvalues = np.linalg.eigvalsh(np.cov(wide.to_numpy(), rowvar=False))[::-1]
        eigenvalues = np.maximum(eigenvalues, 0)
        np.testing.assert_allclose(v.variance_fraction.to_numpy(),
                                   eigenvalues[:len(v)] / eigenvalues.sum(), atol=1e-11)
        checked += 1

    manifest = pd.read_csv(output / 'plots_manifest.csv')
    assert not manifest.file.duplicated().any()
    for row in manifest.itertuples():
        assert (output / row.source_table).is_file()
        with Image.open(output / row.file) as image:
            assert image.width >= 800 and image.height >= 400
            assert max(ImageStat.Stat(image.convert('RGB')).var) > 100
    result = dict(status='passed', fits_checked=checked,
                  figures_checked=len(manifest), source_and_output_hashes_verified=True,
                  note='Figure integrity is automated; visual layout was inspected separately.')
    (output / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/pca_phase1')
    print(json.dumps(verify(parser.parse_args().output_dir.resolve()), indent=2))
