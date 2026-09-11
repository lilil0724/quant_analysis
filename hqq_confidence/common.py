import json
import os
import re
import subprocess
from datetime import datetime, timezone

import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PREPARED_DIR = os.path.join(BASE_DIR, 'results', 'prepared')
STATS_DIR = os.path.join(BASE_DIR, 'results', 'stats')
PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'plots')
HQQ_NBITS = (1, 1.58, 2, 3, 4, 8)
HQQ_CONFIGS = [(bits, group) for bits in HQQ_NBITS for group in (8, 128)]
EXPECTED_CONDITIONS = ['fp_unquantized'] + [
    f'hqq_n{bits:g}_g{group}' for bits, group in HQQ_CONFIGS
]
SWEEP_COLUMNS = [
    'project', 'dataset_name', 'model_name', 'checkpoint_sha256', 'serial', 'seed',
]


def write_run_config(directory, args):
    os.makedirs(directory, exist_ok=True)
    config = vars(args).copy() if hasattr(args, '__dict__') else dict(args)
    config['started_at'] = datetime.now(timezone.utc).isoformat()
    try:
        config['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip()
        config['git_dirty'] = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        config['git_commit'] = None
        config['git_dirty'] = None
    with open(os.path.join(directory, 'run_config.json'), 'w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2, default=str)


def slug(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '-', str(value)).strip('-')


def sweep_id(row):
    seed = row.get('seed', 'unknown')
    return slug(
        f'{row["project"]}_{row["dataset_name"]}_{row["model_name"]}_s{row["serial"]}'
        f'_seed{seed}_{str(row["checkpoint_sha256"])[:12]}'
    )


def as_bool(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ('true', '1', 'yes'):
        return True
    if text in ('false', '0', 'no'):
        return False
    return None


def condition_config(condition):
    if condition == 'fp_unquantized':
        return False, -1, -1
    match = re.fullmatch(r'hqq_n(1|1\.58|2|3|4|8)_g(8|128)', str(condition))
    if not match:
        raise ValueError(f'Unknown confidence condition: {condition}')
    return True, float(match.group(1)), int(match.group(2))
