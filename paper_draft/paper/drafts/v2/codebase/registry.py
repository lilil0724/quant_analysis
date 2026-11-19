"""Datasets, groups, models and modes the paper reports, in print order."""
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
INPUTS = PAPER / 'inputs'
BUILD = PAPER / 'build'

OTHER = ['aircraft', 'cars', 'cub', 'dogs', 'flowers', 'food', 'inat17', 'moe', 'nabirds', 'pets', 'vegfru']
SOY_COTTON = ['soyageing', 'soygene', 'soyglobal', 'soylocal', 'cotton']
DATASETS = OTHER + SOY_COTTON
GRID_DATASETS = DATASETS

MODELS = ['vit', 'swin', 'beit']
MODES = ['ft', 'fz', 'cal']
SETTINGS = [(m, k) for m in MODELS for k in MODES]
MODEL_KEYS = {
    'vit_b16': 'vit',
    'swin_base_patch4_window7_224_in22k': 'swin',
    'beitv2_base_patch16_224_in22k': 'beit',
    'beitv2_base_patch16_224': 'beit',
}
MODEL_LABELS = {'vit': 'ViT-B/16', 'swin': 'Swin-B', 'beit': 'BEiTv2-B'}
MODE_LABELS = {'ft': 'FT', 'fz': 'LP', 'cal': 'CAL'}
GROUP_LABELS = {'sc': 'UFG', 'other': 'FG'}
DATASET_LABELS = {
    'aircraft': 'Aircraft', 'cars': 'Cars', 'cub': 'CUB', 'dogs': 'Dogs', 'flowers': 'Flowers',
    'food': 'Food', 'inat17': 'iNat17', 'moe': 'Moe', 'nabirds': 'NABirds', 'pets': 'Pets',
    'vegfru': 'VegFru', 'soyageing': 'SoyAgeing', 'soygene': 'SoyGene', 'soyglobal': 'SoyGlobal',
    'soylocal': 'SoyLocal', 'cotton': 'Cotton',
}
BITS = [1.0, 1.58, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
GROUP_SIZES = [8, 32, 128, 512]
PRIMARY = [(3.0, 8), (3.0, 128), (4.0, 8), (4.0, 128)]
PERMUTATIONS = 10000
SEED = 0


def group_of(dataset):
    return 'sc' if dataset in SOY_COTTON else 'other'
