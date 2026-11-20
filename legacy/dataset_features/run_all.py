"""Coordinate the existing paired W3/G128 analyses; no new metrics or plots."""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .common import DATASETS

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('vit_b16', 'swin_base_patch4_window7_224_in22k', 'beitv2_base_patch16_224')
ANALYSES = ('paired', 'centered', 'bias', 'margin', 'tsne')
TZ = ZoneInfo('Asia/Taipei')


def csv_names(value):
    return value.split(',')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--serial', type=int, required=True, help='Source serial, or cohort label when using a source map')
    p.add_argument('--input-root', type=Path, default=ROOT/'../../../data/w3_g128_analysis')
    p.add_argument('--source-map', type=Path)
    p.add_argument('--output-root', type=Path, help='Fresh result directory; existing directories are never reused')
    p.add_argument('--run-id', help='Default: Taiwan timestamp plus source serial')
    p.add_argument('--datasets', type=csv_names, default=list(DATASETS))
    p.add_argument('--models', type=csv_names, default=list(MODELS))
    p.add_argument('--modes', type=csv_names, default=['ft', 'fz', 'cal'])
    p.add_argument('--analyses', type=csv_names, default=list(ANALYSES))
    p.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))
    p.add_argument('--dry-run', action='store_true', help='Print commands only; no data scan or output files')
    p.add_argument('--preflight-only', action='store_true', help='Validate selected raw inputs without writing analysis output')
    return p


def jobs(args):
    """Keep each original CLI's defaults and its CSV/space-list convention."""
    root, out = str(args.input_root), args.output_root
    source = ['--source-map', str(args.source_map)] if args.source_map else []
    serial = ['--serial', str(args.serial)]
    words = ['--datasets', *args.datasets, '--models', *args.models, '--modes', *args.modes]
    csv = ['--datasets', ','.join(args.datasets), '--models', ','.join(args.models), '--modes', ','.join(args.modes)]
    def command(module, *parts):
        return [sys.executable, '-B', '-u', '-m', module, *map(str, parts)]
    paired = ['--input-root', root, *serial, *words, *source, '--output-root', str(out/'paired')]
    tsne = ['--input-root', root, *serial, *words, *source, '--output-root', str(out/'tsne')]
    centered = ['--input-root', root, *serial, *words, *source, '--output-root', str(out/'centered')]
    bias = ['--input-root', root, *serial, *csv, *source, '--threads', str(args.threads), '--output-dir', str(out/'bias')]
    margin_source = ['--source-root', root, *serial, *csv, *source]
    margin_workspace = ['--workspace', str(out/'margin'), '--input-id', 'source']
    all_jobs = {
        'paired': dict(preflight=[command('dataset_features.paired', 'preflight', *paired)],
                       commands=[command('dataset_features.paired', 'all', *paired)]),
        'tsne': dict(preflight=[command('dataset_features.tsne_paired', 'preflight', *tsne)],
                     commands=[command('dataset_features.tsne_paired', 'run', *tsne)]),
        # The centered CLI has no preflight command; call its original validator below.
        'centered': dict(preflight=[[]], commands=[command('dataset_features.centered_class_distance', *centered)]),
        'bias': dict(preflight=[command('dataset_confidence.server_bias_competitors', *bias, '--preflight-only')],
                     commands=[command('dataset_confidence.server_bias_competitors', *bias)]),
        'margin': dict(preflight=[command('dataset_confidence.analyze_margin_matching', 'preflight-server', *margin_source)],
                       commands=[command('dataset_confidence.analyze_margin_matching', 'import-server', *margin_source, *margin_workspace),
                                 command('dataset_confidence.analyze_margin_matching', 'run', *margin_workspace, *serial, '--run-id', 'analysis')]),
    }
    return [dict(name=name, **all_jobs[name]) for name in args.analyses]


def now():
    return datetime.now(TZ).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_receipt(path, value):
    partial = path.with_suffix('.json.tmp')
    partial.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    partial.replace(path)


def validate(args):
    if args.serial <= 0 or args.threads <= 0:
        raise ValueError('serial and threads must be positive')
    for key, allowed in [('datasets', DATASETS), ('models', MODELS), ('modes', ('ft','fz','cal')), ('analyses', ANALYSES)]:
        values = getattr(args, key)
        if not values or len(set(values)) != len(values) or not set(values) <= set(allowed):
            raise ValueError(f'Invalid or duplicate {key}: {values}')
    if args.dry_run and args.preflight_only:
        raise ValueError('Choose --dry-run or --preflight-only')
    args.run_id = args.run_id or datetime.now(TZ).strftime('%Y%m%d_%H%M%S_%f') + f'_w3_s{args.serial}'
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', args.run_id):
        raise ValueError('Use a safe run-id containing letters, digits, dot, underscore or hyphen')
    args.input_root = args.input_root.expanduser().resolve()
    if args.input_root.name == str(args.serial):
        args.input_root = args.input_root.parent
    args.output_root = (args.output_root or ROOT/'dataset_features/result'/args.run_id).expanduser().resolve()
    if args.source_map:
        args.source_map = args.source_map.expanduser().resolve()
    if (args.input_root == args.output_root or args.input_root.is_relative_to(args.output_root)
            or args.output_root.is_relative_to(args.input_root)):
        raise ValueError('Choose an output-root outside the raw input tree')


def preflight_centered(args):
    from . import centered_class_distance as centered, paired_inputs
    for dataset, model, mode in itertools.product(args.datasets, args.models, args.modes):
        folder, serial = paired_inputs.select_source(args.input_root/str(args.serial)/dataset/model/mode,
                                                     args.serial, dataset, model, mode, args.source_map)
        centered.validate_pair(folder, serial, dataset, model, mode)
    print('Centered-distance source validation passed.', flush=True)


def snapshot_code(out):
    records = {}
    paths = [p for folder in ('dataset_features','dataset_confidence') for p in (ROOT/folder).glob('*.py')]
    paths.append(ROOT/'dataset_features/run_all_nano4.sh')
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        dest = out/'provenance/source'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        records[relative.as_posix()] = sha256(dest)
    return records


def environment():
    versions = {}
    for package in ('numpy','pandas','matplotlib','scipy','scikit-learn'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(python=sys.version, platform=platform.platform(), packages=versions)


def git_identity():
    try:
        values = []
        for arguments in (['rev-parse','HEAD'], ['status','--short']):
            process = subprocess.Popen(['git','-C',str(ROOT),*arguments], stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL, text=True)
            output, _ = process.communicate()
            if process.returncode:
                return dict(commit=None, dirty=None)
            values.append(output.strip())
        return dict(commit=values[0], dirty=bool(values[1]), status=values[1].splitlines())
    except OSError:
        return dict(commit=None, dirty=None)


def completion_records(job, out):
    name = job['name']
    path = {'paired': out/'paired/analysis_manifest.json', 'centered': out/'centered/run_config.json',
            'bias': out/'bias/completion.json', 'margin': out/'margin/result/analysis/run.json'}.get(name)
    paths = list((out/'tsne').rglob('method.json')) if name == 'tsne' else [path]
    if not paths:
        raise RuntimeError(f'{name} returned without completion receipts')
    for p in paths:
        if not p.is_file() or json.loads(p.read_text(encoding='utf-8')).get('status') != 'complete':
            raise RuntimeError(f'{name} did not finish completely: {p}')
    return {p.relative_to(out).as_posix(): sha256(p) for p in paths}


def run(args):
    validate(args)
    planned = jobs(args)
    for job in planned:
        print(f'[{job["name"]}]', flush=True)
        for command in job['preflight'] + job['commands']:
            print(shlex.join(command) if command else 'read-only centered_class_distance.validate_pair for selected contexts', flush=True)
    if args.dry_run:
        print('Dry-run only; no inputs scanned, outputs created or jobs submitted.', flush=True)
        return
    from . import paired_inputs
    source_map = paired_inputs.source_map_identity(args.source_map)
    for d, m, mode in itertools.product(args.datasets, args.models, args.modes):
        folder, _ = paired_inputs.select_source(args.input_root/str(args.serial)/d/m/mode, args.serial, d, m, mode, args.source_map)
        if folder.resolve().is_relative_to(args.output_root) or args.output_root.is_relative_to(folder.resolve()):
            raise ValueError(f'Output overlaps selected source: {folder}')
    if not args.preflight_only and args.output_root.exists():
        raise FileExistsError(f'Preserving existing output: {args.output_root}; choose a new run-id/output-root')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key] = str(args.threads)
    receipt = dict(run_id=args.run_id, status='running', started_at=now(), timezone='Asia/Taipei',
                   command=[sys.executable, '-B', '-m', __spec__.name, *sys.argv[1:]],
                   parameters=vars(args).copy(), source_map=source_map, stages=[],
                   environment=environment(), outputs={})
    if not args.preflight_only:
        args.output_root.mkdir(parents=True, exist_ok=False)
        (args.output_root/'logs').mkdir()
        write_receipt(args.output_root/'run.json', receipt)
    def save():
        if not args.preflight_only:
            write_receipt(args.output_root/'run.json', receipt)
    def execute(job, phase, index, command):
        label = f'{phase}_{job["name"]}_{index}'
        record = dict(stage=job['name'], phase=phase, command=command or 'centered.validate_pair', started_at=now(), status='running')
        receipt['stages'].append(record)
        save()
        try:
            if args.source_map and sha256(args.source_map) != source_map['sha256']:
                raise RuntimeError('Source map changed during this run')
            if not command:
                preflight_centered(args)
                code = 0
            elif args.preflight_only:
                code = subprocess.run(command, cwd=ROOT, env=env).returncode
            else:
                log = args.output_root/'logs'/(label+'.log')
                record['log'] = str(log)
                with log.open('x', encoding='utf-8') as stream:
                    code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
            record.update(status='complete' if code == 0 else 'failed', exit_code=code, finished_at=now())
            if code:
                raise RuntimeError(f'{label} failed with exit code {code}; see {record.get("log", "Slurm log")}')
        except BaseException as error:
            record.update(status='failed', error=f'{type(error).__name__}: {error}', finished_at=now())
            raise
        finally:
            save()
    try:
        if not args.preflight_only:
            receipt['code'] = snapshot_code(args.output_root)
            receipt['git'] = git_identity()
            if args.source_map:
                shutil.copy2(args.source_map, args.output_root/'provenance/source_map.json')
                if sha256(args.output_root/'provenance/source_map.json') != source_map['sha256']:
                    raise RuntimeError('Source map changed while saving its provenance copy')
        if 'margin' in args.analyses:
            from dataset_confidence.analyze_margin_matching import focus
            focus(argparse.Namespace(grouping='soy-cotton'), args.datasets)
        for job in planned:
            for i, command in enumerate(job['preflight']):
                execute(job, 'preflight', i, command)
        if args.preflight_only:
            print('All selected source preflights passed; no analysis outputs written.', flush=True)
            return
        for job in planned:
            for i, command in enumerate(job['commands']):
                execute(job, 'analysis', i, command)
            receipt['outputs'].update(completion_records(job, args.output_root))
            save()
        if any(sha256(ROOT/name) != value for name, value in receipt['code'].items()):
            raise RuntimeError('Analysis code changed during this run; select a new run after checking the source snapshot')
        receipt['status'] = 'complete'
        print(f'Complete: {args.output_root}', flush=True)
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        receipt['finished_at'] = now()
        save()


def main():
    try:
        run(parser().parse_args())
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(f'ERROR: {error}') from error


if __name__ == '__main__':
    import signal
    def terminate(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminate)
    main()
