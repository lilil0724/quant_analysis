"""Shared checkpoint identity and evaluation pairing; independent of legacy HQQ."""
import re

DATASETS = ('aircraft', 'cars', 'cotton', 'cub', 'dogs', 'flowers', 'food', 'inat17',
            'moe', 'nabirds', 'pets', 'soyageing', 'soygene', 'soyglobal', 'soylocal', 'vegfru')
MODELS = ('vit_b16', 'swin_base_patch4_window7_224_in22k', 'beitv2_base_patch16_224')
MODES = ('ft', 'fz', 'cal')
EVAL_FIELDS = ('dataset_name', 'seed', 'fp16', 'image_size', 'classifier', 'test_resize_size',
               'test_multiple', 'test_square_resize_center_crop', 'test_resize_directly',
               'cal_single_crop', 'selector', 'shuffle_test', 'df_test', 'folder_test',
               'num_images_test', 'cal_ap_only', 'train_trainval', 'custom_mean_std')
REQUIRED = {'dataset_name', 'seed', 'fp16', 'image_size', 'classifier', 'test_resize_size',
            'test_multiple', 'test_square_resize_center_crop', 'num_images_test'}


def canonical(path):
    if isinstance(path, str) and path.startswith('../../data/'):
        return '/work/kyle0724/data/' + path[len('../../data/'):]
    return path


def context(run):
    c = run['config']
    path = canonical(c['ckpt_path'])
    match = re.search(r'/(?:ckpt/)?(cal|ft|fz)(?:_ckpts)?/', path)
    if not match:
        raise ValueError(f'Unknown checkpoint mode: {path}')
    model = c['model_name']
    if model == 'beitv2_base_patch16_224_in22k':
        model = 'beitv2_base_patch16_224'
    return c['dataset_name'], model, match[1]


def differences(left, right):
    mismatch, unrecorded = [], []
    for field in EVAL_FIELDS:
        if field not in left and field not in right and field not in REQUIRED:
            unrecorded.append(field)
        elif field not in left or field not in right or left[field] != right[field]:
            mismatch.append(field)
    for field in ('ckpt_path', 'dataset_root_path'):
        if field not in left or field not in right or canonical(left[field]) != canonical(right[field]):
            mismatch.append(field)
    if left.get('checkpoint_sha256') and right.get('checkpoint_sha256'):
        if left['checkpoint_sha256'] != right['checkpoint_sha256']:
            mismatch.append('checkpoint_sha256')
    return mismatch, unrecorded
