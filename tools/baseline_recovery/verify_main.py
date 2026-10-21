"""Verify promoted defaults against the immutable completed-recovery reference."""
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from legacy_hqq.recover_baselines import context, differences, DATASETS, MODELS, MODES, BITS, GROUPS

ROOT = Path(__file__).resolve().parents[2]
SELECTED = ROOT / 'data/baseline_recovery/selected_20261021'
REFERENCE = ROOT / 'data/baseline_recovery/reference_20261021'
BACKUP = ROOT / 'outputs/backups/baseline-promotion-20261021'
KEY = ['dataset_name', 'model_name', 'ckpt_kind', 'hqq_nbits', 'hqq_group_size']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_input(name):
    path = Path(name).resolve()
    relative = path.relative_to(ROOT.resolve())
    if relative.parts[:2] == ('outputs', 'worktrees'):
        raise ValueError(f'Active source still depends on a worktree: {name}')
    return path


def verify_raw_source(raw, curated, receipt):
    assert sha(raw) == sha(curated), 'Default raw CSV differs from corrected curated source'
    imported = json.loads(receipt.read_text(encoding='utf-8'))
    assert imported['sha256'] == sha(raw)
    assert Path(imported['source']).resolve() == curated.resolve()
    assert Path(imported['destination']).resolve() == raw.resolve()


def verify_downstream(root):
    from dataset_robustness.verify import verify as dataset_verify
    from dataset_robustness.verify_pca import verify as pca_verify
    from granularity_robustness.verify import verify as granularity_verify
    dataset_verify(root / 'dataset_robustness/results/main')
    pca_verify(root / 'dataset_robustness/results/pca_phase1')
    granularity_verify(root / 'granularity_robustness/results/main')


def verify_artifacts(root, artifacts):
    for name, expected in artifacts.items():
        assert sha(root / name) == expected, name


def main():
    manifest = json.loads((SELECTED / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'complete' and manifest['missing_conditions'] == 0
    for name, expected in manifest['inputs'].items():
        assert sha(local_input(name)) == expected, name
    for name, expected in manifest['artifacts'].items():
        assert sha(SELECTED / name) == expected, name
    reference = json.loads((REFERENCE / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in reference['files'].items():
        assert sha(REFERENCE / name) == expected, name
    records = [json.loads(line) for line in (SELECTED / 'selected_runs.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(records) == len({r['id'] for r in records}) == 4752
    groups = {}
    for run in records:
        assert run['state'] == 'finished'
        groups.setdefault(context(run), []).append(run)
    assert set(groups) == set(itertools.product(DATASETS, MODELS, MODES))
    for key, group in groups.items():
        base = [r for r in group if r['config']['hqq'] is False]
        quant = [r for r in group if r['config']['hqq'] is True]
        assert len(base) == 1 and len(quant) == 32, key
        assert {(float(r['config']['hqq_nbits']), int(r['config']['hqq_group_size'])) for r in quant} == set(itertools.product(BITS, GROUPS))
        assert all(not differences(r['config'], base[0]['config'])[0] for r in quant), key
    raw_path = ROOT / 'legacy_hqq/data/raw/backbones_quant.csv'
    verify_raw_source(raw_path, SELECTED / 'curated_15_datasets.csv', raw_path.with_name('source_manifest.json'))
    raw = pd.read_csv(raw_path, low_memory=False)
    assert len(raw) == 4455 and raw.dataset_name.nunique() == 15
    summary_path = ROOT / 'legacy_hqq/data/processed/main/quant_summary.csv'
    summary = pd.read_csv(summary_path).sort_values(KEY).reset_index(drop=True)
    expected = pd.read_csv(REFERENCE / 'quant_summary.csv').sort_values(KEY).reset_index(drop=True)
    assert len(summary) == len(expected) == 4320 and summary.baseline_matched.all()
    pd.testing.assert_frame_equal(summary[KEY], expected[KEY])
    fields = ['fp32_top1', 'hqq_top1', 'accuracy_ratio']
    np.testing.assert_allclose(summary[fields], expected[fields], rtol=0, atol=1e-12)
    original = pd.read_csv(BACKUP / 'files/legacy_hqq/data/processed/main/quant_summary.csv')
    comparison = summary.merge(original, on=KEY, suffixes=('_new', '_old'), validate='one_to_one')
    affected = comparison.dataset_name.isin(['dogs','flowers','moe','pets','vegfru']) & comparison.model_name.str.startswith('beitv2')
    assert affected.sum() == 480
    for field in fields:
        np.testing.assert_allclose(comparison.loc[~affected, field+'_new'], comparison.loc[~affected, field+'_old'], rtol=0, atol=1e-12)
    upstream = [ROOT / 'dataset_robustness/data/upstream/legacy/quant_summary.csv',
                ROOT / 'granularity_robustness/data/upstream/quant_summary.csv']
    assert all(sha(path) == sha(summary_path) for path in upstream)
    pairs_dir = ROOT / 'outputs/serial_20007/results/main'
    pairs = pd.read_csv(pairs_dir / 'acc_ratio_pairs.csv').sort_values(['dataset','model','mode']).reset_index(drop=True)
    expected_pairs = pd.read_csv(REFERENCE / 'w6a4_pairs.csv').sort_values(['dataset','model','mode']).reset_index(drop=True)
    assert len(pairs) == 144 and pairs.accepted.all()
    pd.testing.assert_frame_equal(pairs[['w6a4_run_id','baseline_run_id']], expected_pairs[['w6a4_run_id','baseline_run_id']])
    np.testing.assert_allclose(pairs[['w6a4_top1_pct','baseline_top1_pct','accuracy_ratio']],
                               expected_pairs[['w6a4_top1_pct','baseline_top1_pct','accuracy_ratio']], rtol=0, atol=1e-12)
    verify_downstream(ROOT)
    receipts = [ROOT / 'dataset_robustness/results/main/verification.json',
                ROOT / 'dataset_robustness/results/pca_phase1/verification.json',
                ROOT / 'granularity_robustness/results/main/verification.json',
                pairs_dir / 'verification.json']
    for receipt in receipts:
        result = json.loads(receipt.read_text(encoding='utf-8'))
        assert result['status'] in ('passed', 'complete'), receipt
        for name, expected_hash in result.get('inputs', {}).items():
            assert sha(local_input(name)) == expected_hash
        verify_artifacts(receipt.parent, result.get('artifacts', {}))
    for path in [ROOT / 'legacy_hqq/results/main/summary/run_config.json',
                 ROOT / 'dataset_robustness/results/main/run_config.json',
                 ROOT / 'dataset_robustness/results/pca_phase1/run_config.json',
                 ROOT / 'granularity_robustness/results/main/run_config.json']:
        config = json.loads(path.read_text(encoding='utf-8'))
        for name in ('input_file', 'source', 'main_results'):
            if name in config:
                assert local_input(config[name]).exists()
    coverage = pd.read_csv(SELECTED / 'context_coverage.csv')
    assert len(coverage) == 144 and coverage.complete.all()
    assert pd.read_csv(SELECTED / 'pair_audit.csv').accepted.all()
    inputs = [SELECTED / 'manifest.json', REFERENCE / 'manifest.json', summary_path, *upstream, *receipts]
    result = dict(status='passed', verified_at=datetime.now(timezone.utc).isoformat(),
                  full_grid_pairs=4608, baseline_contexts=144, legacy_cells=4320,
                  unchanged_legacy_cells=3840, corrected_legacy_cells=480, w6a4_ratios=144,
                  missing_conditions=0, unmatched_baselines=0,
                  completed_worktree_values_match=True, active_inputs_are_main_local=True,
                  inputs={str(path):sha(path) for path in inputs})
    out = ROOT / 'outputs/baseline_recovery/promotion_20261021'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'inputs'}, indent=2))


if __name__ == '__main__':
    main()
