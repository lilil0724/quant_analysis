"""Download run-associated prediction artifacts into a reproducible cache."""

import argparse
import json
import netrc
import os

import pandas as pd
import wandb

if __package__:
    from .common import DATA_DIR, EXPECTED_CONDITIONS, SWEEP_COLUMNS, as_bool, write_run_config
else:
    from common import DATA_DIR, EXPECTED_CONDITIONS, SWEEP_COLUMNS, as_bool, write_run_config


DEFAULT_CACHE = os.path.join(DATA_DIR, 'artifacts')
DEFAULT_MANIFEST = os.path.join(DATA_DIR, 'selected_runs.csv')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download finished TGDA HQQ confidence artifacts from W&B.')
    parser.add_argument('--project-name', default='nycu_pcs/TiT')
    parser.add_argument('--serials', nargs='+', type=int)
    parser.add_argument(
        '--run-ids-file',
        help='CSV with a run_id column, or a text file containing one run ID per line')
    parser.add_argument('--output-file', default=DEFAULT_MANIFEST)
    parser.add_argument('--cache-dir', default=DEFAULT_CACHE)
    parser.add_argument('--results-dir', default=DATA_DIR)
    return parser.parse_args()


def has_wandb_credentials():
    if os.environ.get('WANDB_API_KEY'):
        return True
    candidates = [None]
    if os.name == 'nt':
        candidates.insert(0, os.path.expanduser('~/_netrc'))
    for path in candidates:
        try:
            credentials = netrc.netrc(path).authenticators('api.wandb.ai')
            if credentials and credentials[2]:
                return True
        except (OSError, netrc.NetrcParseError):
            continue
    return False


def read_run_ids(path):
    if not path:
        return None
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        frame = None
    if frame is not None and 'run_id' in frame:
        return [str(value) for value in frame['run_id'].dropna()]
    with open(path, encoding='utf-8') as handle:
        return [line.strip() for line in handle if line.strip()]


def _run_value(run, name, default=None):
    value = getattr(run, name, default)
    return value if value is not None else default


def discover_runs(api, project_name, serials=None, run_ids=None):
    if run_ids:
        runs = [api.run(f'{project_name}/{run_id}') for run_id in run_ids]
    else:
        runs = list(api.runs(path=project_name, filters={'state': 'finished'}))
    selected = []
    for run in runs:
        if _run_value(run, 'state') != 'finished':
            continue
        if _run_value(run, 'job_type') != 'hqq-confidence-eval':
            continue
        if serials is not None and run.config.get('serial') not in serials:
            continue
        if run.config.get('condition') not in EXPECTED_CONDITIONS:
            continue
        selected.append(run)
    return selected


def _prediction_artifact(run):
    artifacts = [artifact for artifact in run.logged_artifacts()
                 if getattr(artifact, 'type', None) == 'prediction-logits']
    if len(artifacts) != 1:
        raise ValueError(
            f'Run {run.id} must log exactly one prediction-logits artifact; '
            f'found {len(artifacts)}')
    return artifacts[0]


def _find_npz(directory):
    matches = []
    for root, _, files in os.walk(directory):
        matches.extend(os.path.join(root, name) for name in files
                       if name == 'predictions.npz')
    if len(matches) != 1:
        raise ValueError(
            f'Artifact cache {directory} must contain one predictions.npz; '
            f'found {len(matches)}')
    return os.path.abspath(matches[0])


def _manifest_row(project_name, run, artifact, npz_path):
    config = run.config
    condition = config['condition']
    hqq = condition != 'fp_unquantized'
    accuracy_key = ('hqq_top1'
                    if hqq and run.summary.get('hqq_top1') is not None else 'top1')
    return {
        'project': project_name, 'run_id': run.id,
        'run_name': _run_value(run, 'name', ''),
        'group': _run_value(run, 'group', ''),
        'job_type': _run_value(run, 'job_type', ''),
        'state': _run_value(run, 'state', ''),
        'created_at': str(_run_value(run, 'created_at', '')),
        'serial': config.get('serial'), 'dataset_name': config.get('dataset_name'),
        'model_name': config.get('model_name'), 'ckpt_path': config.get('ckpt_path'),
        'checkpoint_sha256': config.get('checkpoint_sha256'),
        'condition': condition, 'seed': config.get('seed'),
        'debugging': as_bool(config.get('debugging')),
        'amp_fp16': as_bool(config.get('fp16', config.get('amp_fp16'))),
        'compute_dtype': config.get('hqq_compute_dtype', config.get('compute_dtype')),
        'git_commit': config.get('git_commit'),
        'artifact_name': getattr(artifact, 'name', None),
        'artifact_version': getattr(artifact, 'version', None),
        'artifact_digest': getattr(artifact, 'digest', None),
        'npz_path': npz_path, 'summary_accuracy_key': accuracy_key,
        'summary_accuracy': run.summary.get(accuracy_key),
        'n_samples': run.summary.get('n_samples'),
        'num_classes': run.summary.get('num_classes'),
        'quantized_linear_count': run.summary.get('quantized_linear_count'),
        'config_json': json.dumps(dict(config), sort_keys=True, default=str),
        'summary_json': json.dumps(dict(run.summary), sort_keys=True, default=str),
    }


def download(api, project_name, output_file, cache_dir, serials=None, run_ids=None):
    runs = discover_runs(api, project_name, serials=serials, run_ids=run_ids)
    identities = []
    for run in runs:
        config = run.config
        identities.append(tuple(config.get(column) if column != 'project' else project_name
                                for column in SWEEP_COLUMNS) + (config.get('condition'),))
    if len(identities) != len(set(identities)):
        raise ValueError(
            'Ambiguous duplicate runs found. Supply --run-ids-file with one explicit '
            'run per sweep condition.')

    rows = []
    for run in runs:
        artifact = _prediction_artifact(run)
        version = getattr(artifact, 'version', 'unknown') or 'unknown'
        destination = os.path.join(cache_dir, str(run.id), str(version))
        os.makedirs(destination, exist_ok=True)
        artifact.download(root=destination)
        rows.append(_manifest_row(
            project_name, run, artifact, _find_npz(destination)))
    manifest = pd.DataFrame(rows)
    if len(manifest):
        manifest = manifest.sort_values(
            SWEEP_COLUMNS[1:] + ['condition'], kind='stable').reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    manifest.to_csv(output_file, index=False)
    return manifest


def main():
    args = parse_args()
    args.output_file = os.path.abspath(args.output_file)
    args.cache_dir = os.path.abspath(args.cache_dir)
    args.results_dir = os.path.abspath(args.results_dir)
    if not has_wandb_credentials():
        raise RuntimeError(
            'W&B authentication is required. Set WANDB_API_KEY or run "wandb login".')
    write_run_config(args.results_dir, args)
    manifest = download(
        wandb.Api(), args.project_name, args.output_file, args.cache_dir,
        serials=args.serials, run_ids=read_run_ids(args.run_ids_file))
    print(f'Downloaded {len(manifest)} run artifacts')
    print(f'Wrote {args.output_file}')


if __name__ == '__main__':
    main()
