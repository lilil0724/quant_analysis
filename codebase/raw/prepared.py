"""Convert a raw-export run into the paper table formats it can supply."""

import json

import numpy as np

from .common import read_csv, write_csv


GEOMETRY_FIELDS = (
    'n_classes', 'feature_dim', 'source_dtype', 'n_intra_classes', 'n_inter_class_pairs',
    'mean_norm', 'rms_feature_norm', 'mean_energy_fraction', 'centered_min_norm',
    'centered_zero_norm_count', 'centered_status', 'raw_d_inter', 'raw_d_intra',
    'raw_distance_gap', 'raw_distance_ratio', 'raw_normalized_separation',
    'centered_d_inter', 'centered_d_intra', 'centered_distance_gap',
    'centered_distance_ratio', 'centered_normalized_separation',
)
BIAS_FIELDS = ('gain_pp', 'shuffle_mean_gain_pp', 'specificity_pp', 'corrected_acc_pct')


def write_prepared(out, pairs, tables, analyses):
    """Write fresh per-run inputs without editing the committed paper snapshot."""
    prepared = out / 'prepared'
    for source, filename in (('classifier_bias', 'bias_results.csv'),
                             ('competitor_ranks', 'winner_ranks.csv')):
        if source in tables:
            write_csv(prepared / filename, tables[source])
    if 'all_pairs' not in tables:
        return

    classes = {(pair['model'], pair['mode']): pair['num_classes'] for pair in pairs}
    centered = {(row['model'], row['mode']): row for row in tables.get('centered_class_distance', ())}
    bias = {(row['model_name'], row['checkpoint_mode']): row for row in tables.get('classifier_bias', ())}
    master = []
    layers = []
    for row in tables['all_pairs']:
        key = row['model'], row['mode']
        context = out / row['dataset'] / key[0] / key[1]
        layers.extend(read_csv(context / 'layer_metrics.csv'))
        with np.load(context / 'fp_pca.npz', allow_pickle=False) as pca:
            pc1, pc2 = map(float, pca['explained'])
        tsne = {}
        if 'tsne' in analyses:
            tsne = json.loads((context / 'tsne' / 'method.json').read_text(encoding='utf-8'))
        selected = tsne.get('selected_samples')
        loss = row['relative_loss']
        master.append(dict(
            **row, pc1_explained=pc1, pc2_explained=pc2, pca_2d_explained=pc1 + pc2,
            geometry_relative_loss_percent=None if loss is None else 100 * loss,
            **{f'geometry_{name}': centered.get(key, {}).get(name) for name in GEOMETRY_FIELDS},
            num_classes=classes[key],
            **{f'bias_{name}': bias.get(key, {}).get(name) for name in BIAS_FIELDS},
            tsne_selected_samples=selected, tsne_selected_classes=tsne.get('selected_classes'),
            tsne_selected_fraction=None if selected is None else selected / row['n'],
        ))
    write_csv(prepared / 'dataset_master.csv', master)
    write_csv(prepared / 'layer_metrics.csv', layers)
