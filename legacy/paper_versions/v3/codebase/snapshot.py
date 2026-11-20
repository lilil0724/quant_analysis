"""Copy the small tables the paper reads from the run bundles into inputs/; run where the bundles exist."""
import hashlib, json, shutil
from pathlib import Path
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]
RUN = REPO / 'paper_draft/data/run_all/20261115_211022_759540_w3_s20004'
DERIVED = REPO / 'paper_draft/data/derived/20261116_140853_554056_runall_s20004_audit'
MARGIN = RUN / 'margin/result/analysis/tables'
OUT = PAPER / 'inputs'

COPIES = {
    'grid.csv': REPO / 'quant_acc/20261117_legacy_hqq_s10000_10016_cal10016_v2/prepared/quant_summary.csv',
    'dataset_master.csv': DERIVED / 'dataset_master.csv',
    'margin_gaps.csv': MARGIN / 'gaps.csv',
    'margin_diagnostics.csv': MARGIN / 'diagnostics.csv',
    'margin_balance.csv': MARGIN / 'balance.csv',
    'margin_coverage.csv': MARGIN / 'coverage.csv',
    'bias_results.csv': RUN / 'bias/bias_results.csv',
    'winner_ranks.csv': RUN / 'bias/winner_ranks.csv',
    'competitor_comparisons.csv': RUN / 'bias/heldout_prediction_comparisons.csv',
    'competitor_scores.csv': RUN / 'bias/heldout_prediction_scores.csv',
    'matching_support.csv': DERIVED / 'matching_support_summary.csv',
    'tsne_cub_vit_b16_ft.csv': RUN / 'tsne/cub/vit_b16/ft/coordinates.csv',
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    manifest = {}
    for name, source in COPIES.items():
        shutil.copyfile(source, OUT / name)
        manifest[name] = {'source': str(source.relative_to(REPO)), 'sha256': sha256(source)}
    layers = sorted(RUN.glob('paired/*/*/*/layer_metrics.csv'))
    assert len(layers) == 144
    pd.concat([pd.read_csv(p) for p in layers]).to_csv(OUT / 'layer_metrics.csv', index=False)
    manifest['layer_metrics.csv'] = {'source': str((RUN / 'paired/*/*/*/layer_metrics.csv').relative_to(REPO)),
                                     'files': len(layers)}
    (OUT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')


if __name__ == '__main__':
    main()
