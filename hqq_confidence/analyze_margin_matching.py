"""Import immutable logits snapshots and run baseline-margin standardization.

Server input: SERIAL/DATASET/MODEL/MODE/{manifest.json,samples.json,fp/,w3_g128/}.
Confidence input: selected_runs.csv plus explicitly rooted artifacts/RUN/VERSION/.
No W&B, new inference, or checkpoint loading is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import itertools
import json
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import margin_matching as core
from . import server_bias_competitors as server
from .common import EXPECTED_CONDITIONS
from .prepare_pairs import load_predictions, provenance_status, _validate_metadata

BASE = Path(__file__).resolve().parent
REPO = BASE.parent
TZ = timezone(timedelta(hours=8), 'Asia/Taipei')
SOY = ('soyageing', 'soygene', 'soyglobal', 'soylocal')
FOCUS = (*SOY, 'cotton')
EVAL_KEYS = ('classifier', 'classifier_aux', 'num_classes', 'image_size', 'resize_size',
             'test_resize_size', 'test_resize_directly', 'test_square_resize_center_crop',
             'test_multiple', 'cal_single_crop', 'eval_split', 'df_test', 'dataset_root_path',
             'test_only_bs1', 'fp16', 'encoder_norm', 'vit_features_type', 'seed')


def now():
    return datetime.now(TZ).isoformat()


def identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value):
        raise argparse.ArgumentTypeError('Use a safe input/run ID, at most 128 characters')
    return value


def write_json(path, value):
    """Publish receipts atomically; serialization/I/O failures never publish complete."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.receipt-', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(server.clean(value), handle, indent=2, allow_nan=False, default=str)
            handle.write('\n')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command', required=True)
    for name in ('preflight-server', 'import-server', 'import-confidence', 'run'):
        sub = commands.add_parser(name)
        sub.add_argument('--workspace', type=Path, default=BASE,
                         help='Owning analysis root containing data/, result/, reports/')
        sub.add_argument('--input-id', required=name != 'preflight-server', type=identifier)
        if name != 'run':
            sub.add_argument('--source-root', type=Path, required=True)
            sub.add_argument('--serial', type=int, required=True)
        if name in ('preflight-server', 'import-server'):
            sub.add_argument('--datasets', type=server.csv_selection, default=list(server.DATASETS))
            sub.add_argument('--models', type=server.csv_selection, default=list(server.MODELS))
            sub.add_argument('--modes', type=server.csv_selection, default=list(server.MODES))
            sub.add_argument('--chunk-size', type=int, default=256)
        if name == 'run':
            sub.add_argument('--serial', type=int, help='Optional assertion against imported serial')
            sub.add_argument('--run-id', type=identifier,
                             default=datetime.now(TZ).strftime('%Y%m%d_%H%M%S_%f_margin'))
            sub.add_argument('--grouping', choices=('soy-cotton', 'legacy-soy'), default='soy-cotton',
                             help='legacy-soy reproduces the historical soy4/other10 snapshot')
            sub.add_argument('--chunk-size', type=int, default=256)
    return result


def validate_server(args):
    if args.serial <= 0 or args.chunk_size <= 0:
        raise ValueError('serial and chunk-size must be positive')
    root = args.source_root.expanduser().resolve()
    if root.name == str(args.serial):
        root = root.parent
    # Reuse the already deployed pairing/hash/precision protocol, without running
    # the unrelated bias/competitor statistical models or writing their outputs.
    options = argparse.Namespace(input_root=root, serial=args.serial, output_dir=None,
        datasets=args.datasets, models=args.models, modes=args.modes,
        chunk_size=args.chunk_size, threads=1, seed=100004)
    pairs, _ = server.preflight(options)
    return root, pairs


def source_record(source, relative, expected=None):
    source = Path(source).resolve()
    digest = server.sha256_file(source)
    if expected is not None and digest != expected:
        raise ValueError(f'Source hash mismatch: {source}')
    return dict(source=str(source), path=relative, size_bytes=source.stat().st_size, sha256=digest)


def publish_input(args, kind, records, identities, selection):
    target = args.workspace.expanduser().resolve() / 'data' / args.input_id
    # Never place a snapshot over, inside, or around its upstream source tree.
    upstream = args.source_root.expanduser().resolve()
    if target == upstream or target in upstream.parents or upstream in target.parents:
        raise ValueError('Input snapshot must be separate from upstream source root')
    target.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema=1, status='running', input_id=args.input_id, source_format=kind,
        serial=args.serial, acquired_at=now(), timezone='Asia/Taipei', source_root=str(upstream),
        selection=selection, identities=identities, files=records,
        relocation='Original manifests are preserved verbatim; explicit source-to-snapshot paths are recorded here.')
    try:
        write_json(target/'source_manifest.json', receipt)
        for record in records:
            source = Path(record['source'])
            destination = server.safe_file(target, record['path'])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if destination.stat().st_size != record['size_bytes'] or server.sha256_file(destination) != record['sha256']:
                raise ValueError(f'Imported hash mismatch: {source}')
        # Upstream mutation during copying invalidates the whole import.
        for record in records:
            if server.sha256_file(record['source']) != record['sha256']:
                raise ValueError(f'Source changed during import: {record["source"]}')
        receipt.update(status='complete', finished_at=now())
        write_json(target/'source_manifest.json', receipt)
    except BaseException as error:
        receipt.update(status='failed', finished_at=now(), error=f'{type(error).__name__}: {error}')
        write_json(target/'source_manifest.json', receipt)
        raise
    print(f'IMPORTED {kind}: {target}', flush=True)


def import_server(args):
    target = args.workspace.expanduser().resolve()/'data'/args.input_id
    if target.exists():
        raise FileExistsError(f'Input exists: {target}; choose a new input ID')
    root, pairs = validate_server(args)
    records, identities = [], []
    for pair in pairs:
        relative = pair['folder'].relative_to(root).as_posix()
        identities.append(dict(dataset=pair['dataset'], model=pair['model'], mode=pair['mode'], folder=relative))
        for name, digest in pair['hashes'].items():
            records.append(source_record(pair['folder']/name, relative+'/'+name, digest))
    publish_input(args, 'server-w3-g128', records, identities,
                  dict(datasets=args.datasets, models=args.models, modes=args.modes))


def confidence_rows(manifest, serial):
    frame = pd.read_csv(manifest, keep_default_na=False)
    required = {'dataset_name','model_name','serial','condition','seed','ckpt_path',
                'checkpoint_sha256','config_json','run_id','artifact_version','sha256',
                'n_samples','num_classes','summary_accuracy'}
    if not required <= set(frame) or frame.empty or set(frame.serial) != {serial}:
        raise ValueError('Invalid confidence manifest identity/columns/serial')
    if frame.duplicated(['dataset_name','model_name','condition']).any():
        raise ValueError('Duplicate confidence dataset/model/condition')
    for (_, _), group in frame.groupby(['dataset_name','model_name']):
        if set(group.condition) != set(EXPECTED_CONDITIONS):
            raise ValueError('Incomplete 13-condition confidence sweep')
        if provenance_status(group):
            raise ValueError(f'Confidence provenance mismatch: {provenance_status(group)}')
        for key in ('seed', 'ckpt_path', 'checkpoint_sha256'):
            if group[key].nunique() != 1 or not str(group.iloc[0][key]):
                raise ValueError(f'Confidence pairing mismatch: {key}')
        signatures = set()
        for row in group.to_dict('records'):
            config = json.loads(row['config_json'])
            for column, key in [('ckpt_path','ckpt_path'),('seed','seed'),('dataset_name','dataset_name'),
                                ('model_name','model_name'),('serial','serial')]:
                if config.get(key) != row[column]:
                    raise ValueError(f'Config/manifest identity mismatch: {key}')
            if not re.fullmatch('[0-9a-fA-F]{64}', row['checkpoint_sha256']):
                raise ValueError('Missing checkpoint bytes identity')
            signatures.add(server.digest_json({k:config.get(k) for k in EVAL_KEYS}))
        if len(signatures) != 1:
            raise ValueError('Confidence evaluation/classifier config mismatch')
    return frame


def import_confidence(args):
    target = args.workspace.expanduser().resolve()/'data'/args.input_id
    if target.exists():
        raise FileExistsError(f'Input exists: {target}; choose a new input ID')
    root = args.source_root.expanduser().resolve()
    frame = confidence_rows(root/'selected_runs.csv', args.serial)
    records = [source_record(root/'selected_runs.csv', 'selected_runs.csv')]
    identities = []
    for row in frame.to_dict('records'):
        # Explicit RUN/VERSION rooting is required after relocation. Never search
        # by basename or mutate the archived absolute npz_path in the old manifest.
        relative = f'artifacts/{identifier(row["run_id"])}/{identifier(row["artifact_version"])}/predictions.npz'
        records.append(source_record(server.safe_file(root, relative), relative, row['sha256']))
        identities.append(dict(dataset=row['dataset_name'], model=row['model_name'], condition=row['condition'], path=relative))
    publish_input(args, 'confidence-npz', records, identities,
                  dict(datasets=sorted(frame.dataset_name.unique()), models=sorted(frame.model_name.unique()), modes=['ft']))


def read_input(args):
    root = args.workspace.expanduser().resolve()/'data'/args.input_id
    manifest = root/'source_manifest.json'
    meta = json.loads(manifest.read_text(encoding='utf-8'))
    if meta.get('status') != 'complete' or meta.get('schema') != 1 or meta.get('input_id') != args.input_id:
        raise ValueError('Input snapshot is incomplete or incompatible')
    seen = set()
    for record in meta['files']:
        path = server.safe_file(root, record['path'])
        if record['path'] in seen or path.stat().st_size != record['size_bytes'] or server.sha256_file(path) != record['sha256']:
            raise ValueError(f'Input checksum/hash mismatch: {path}')
        seen.add(record['path'])
    return root, meta, server.sha256_file(manifest)


def focus(args, datasets):
    chosen = set(SOY if args.grouping == 'legacy-soy' else FOCUS)
    label = 'soy' if args.grouping == 'legacy-soy' else 'soy_cotton'
    if not chosen.intersection(datasets) or not set(datasets).difference(chosen):
        raise ValueError('Selected setting must contain both dataset groups')
    return chosen, label


def load_server(root, meta, args):
    for model, mode in itertools.product(meta['selection']['models'], meta['selection']['modes']):
        chunks, outcomes, coverage = [], [], []
        identities = [p for p in meta['identities'] if p['model'] == model and p['mode'] == mode]
        if set(p['dataset'] for p in identities) != set(meta['selection']['datasets']) or len(identities) != len(meta['selection']['datasets']):
            raise ValueError(f'Incomplete snapshot matrix: {model}/{mode}')
        chosen, label = focus(args, [p['dataset'] for p in identities])
        signatures = set()
        for identity in identities:
            folder = server.safe_file(root, identity['folder'])
            p = server.check_pair(folder, meta['serial'], identity['dataset'], model, mode, args.chunk_size, keep_predictions=True)
            signatures.add(server.digest_json(dict(amp_fp16=p['meta']['amp_fp16'],
                quantization=p['meta']['quantization'],code=p['meta']['code_sha256'],hqq=p['meta']['hqq_source_sha256'])))
            z = np.load(folder/'fp/logits.npy', mmap_mode='r', allow_pickle=False)
            try:
                margins = np.concatenate([core.normalized_margin(z[i:i+args.chunk_size]) for i in range(0,len(z),args.chunk_size)])
            finally:
                server.close_maps(z)
            dataset = p['dataset']
            chunks.append(pd.DataFrame(dict(dataset=dataset,group=label if dataset in chosen else 'other',
                sample_id=p['ids'],target=p['target'],baseline_correct=p['fp_pred']==p['target'],margin=margins)))
            outcomes.append((p['quant_pred']==p['target'])[:,None])
            coverage.extend(server.coverage_rows([p],meta['serial']))
            print(f'LOADED {dataset}/{model}/{mode}',flush=True)
        if len(signatures) != 1:
            raise ValueError(f'Mixed server precision/quantization/export protocol: {model}/{mode}')
        yield model, mode, pd.concat(chunks,ignore_index=True), np.vstack(outcomes), ['hqq_n3_g128'], coverage


def load_confidence(root, meta, args):
    frame = confidence_rows(root/'selected_runs.csv',meta['serial'])
    for model, model_rows in frame.groupby('model_name',sort=True):
        chunks, outcomes, coverage = [], [], []
        chosen, label = focus(args,model_rows.dataset_name.unique())
        for dataset, group in model_rows.groupby('dataset_name',sort=True):
            by_condition = {r['condition']:r for r in group.to_dict('records')}
            baseline = None
            columns = []
            for condition in EXPECTED_CONDITIONS:
                row = by_condition[condition]
                path = server.safe_file(root,f'artifacts/{row["run_id"]}/{row["artifact_version"]}/predictions.npz')
                data = load_predictions(path)
                _validate_metadata(data,row)
                declared_shape = (int(row['n_samples']), int(row['num_classes']))
                if data['_shape'] != declared_shape or json.loads(row['config_json']).get('num_classes') != data['_shape'][1]:
                    raise ValueError(f'Confidence logit shape/class dimension mismatch: {dataset}/{condition}')
                if not np.isclose(100*np.mean(data['correct']),float(row['summary_accuracy']),rtol=0,atol=1e-4):
                    raise ValueError(f'Accuracy mismatch: {dataset}/{condition}')
                if baseline is None:
                    baseline = data
                    with np.load(path,allow_pickle=False) as payload:
                        z = payload['logits']
                        margins = np.concatenate([core.normalized_margin(z[i:i+args.chunk_size]) for i in range(0,len(z),args.chunk_size)])
                    del z
                else:
                    if data['_shape'] != baseline['_shape']:
                        raise ValueError(f'Confidence paired class dimension mismatch: {dataset}/{condition}')
                    ids = pd.Index(data['sample_id'])
                    order = ids.get_indexer(baseline['sample_id'])
                    if len(ids)!=len(order) or (order<0).any() or not np.array_equal(data['target'][order],baseline['target']):
                        raise ValueError(f'Confidence sample/target pairing mismatch: {dataset}/{condition}')
                    columns.append(data['correct'][order])
            chunks.append(pd.DataFrame(dict(dataset=dataset,group=label if dataset in chosen else 'other',
                sample_id=baseline['sample_id'],target=baseline['target'],baseline_correct=baseline['correct'],margin=margins)))
            outcomes.append(np.column_stack(columns))
            row = by_condition['fp_unquantized']
            coverage.append(dict(serial=meta['serial'],dataset_name=dataset,model_name=model,checkpoint_mode='ft',
                n_samples=len(baseline['target']),checkpoint=row['ckpt_path'],checkpoint_sha256=row['checkpoint_sha256'],
                inference_seed=row['seed'],amp_fp16=row['amp_fp16'],hqq_compute_dtype=row['compute_dtype'],
                classifier=json.loads(row['config_json']).get('classifier'),source_format='confidence-npz'))
            print(f'LOADED {dataset}/{model}/ft: N={len(baseline["target"])}',flush=True)
        yield model,'ft',pd.concat(chunks,ignore_index=True),np.vstack(outcomes),EXPECTED_CONDITIONS[1:],coverage


def source_identity(output):
    names = ('analyze_margin_matching.py','margin_matching.py','server_bias_competitors.py',
             'bias_competitor_core.py','common.py','prepare_pairs.py','__init__.py',
             'run_margin_matching_nano4.sh','requirements_margin_matching.txt')
    hashes = {}
    for name in names:
        path = BASE/name
        if not path.is_file():
            continue
        copy = output/'provenance/source/hqq_confidence'/name
        copy.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,copy)
        hashes[f'hqq_confidence/{name}'] = server.sha256_file(path)
        if server.sha256_file(copy)!=hashes[f'hqq_confidence/{name}']:
            raise ValueError('Code snapshot hash mismatch')
    try:
        commit = subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(['git','-C',str(REPO),'status','--short'],text=True,stderr=subprocess.DEVNULL).splitlines()
    except (OSError,subprocess.CalledProcessError):
        commit, status = None, None
    return dict(commit=commit,dirty=bool(status) if status is not None else None,status=status,files=hashes)


def plot_setting(samples,tables,fits,output,model,mode,serial):
    primary=tables['group_metrics'].query('bin_width == 0.05')
    sizes=sorted(primary.group_size.unique())
    fig,axes=plt.subplots(1,2+len(sizes),figsize=(5*(2+len(sizes)),4.6),layout='constrained')
    fit=fits[.05]
    names=sorted(samples.group.unique())
    colors={name:('#287d9e' if name=='other' else '#be4560') for name in names}
    for ax,weighting in zip(axes[:2],('raw','matched')):
        for name in names:
            mask=samples.group.eq(name).to_numpy()
            # Display the actual 0.05 matching bins with complete support.
            if fit:
                hist=fit['original_hist'][name] if weighting=='raw' else fit['matched_hist'][name]
                ax.stairs(hist,fit['edges'],label=name,color=colors[name],linestyle='-' if name=='other' else '--')
        ax.set(title=weighting+' baseline margin',xlabel='(top1 - top2) / class-logit std',ylabel='Group mass per 0.05 bin')
        if fit: ax.legend()
        else: ax.text(.5,.5,'No common margin-bin support',ha='center',transform=ax.transAxes)
    if fit:
        supported=np.flatnonzero(fit['target']>0)
        axes[1].set_xlim(0,fit['edges'][supported[-1]+1]*1.03)
    for ax,size in zip(axes[2:],sizes):
        t=primary[primary.group_size.eq(size)].sort_values('bits')
        conditions=list(dict.fromkeys(t.condition))
        for name in names:
            for weighting,style in (('raw','--'),('matched','-')):
                v=t[t.group.eq(name)&t.weighting.eq(weighting)].set_index('condition').reindex(conditions)
                ax.plot(range(len(conditions)),v.accuracy_ratio,style,color=colors[name],marker='o',label=f'{name}: {weighting}')
        ax.set(title=f'Accuracy retention, G{size}',ylabel='weighted Acc_Q / weighted Acc_FP',xlabel='Weight bits',
               xticks=range(len(conditions)),xticklabels=[c.split('_n')[1].split('_g')[0] for c in conditions])
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.suptitle(f'Serial {serial} | {model} | {mode} | baseline-only overlap matching')
    path=output/'figures'/f'{model}_{mode}.png'
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160)
    plt.close(fig)
    return path


def number(value):
    return f'{value:.5f}' if np.isfinite(value) else 'N/A'


def report_text(receipt,tables=None,figures=()):
    lines=['# Baseline normalized-margin matching','',f'Status: **{receipt["status"]}**; run ID: `{receipt["run_id"]}`.',
        f'Start: {receipt["started_at"]}; finish: {receipt.get("finished_at","pending")}; timezone: Asia/Taipei.',
        f'Elapsed: {receipt.get("elapsed_seconds",0):.2f} s.',
        f'Input: `{receipt["input_path"]}`; source manifest SHA-256: `{receipt.get("input_manifest_sha256","unavailable")}`.',
        f'Source serial: {receipt.get("serial","unknown")}; upstream: `{receipt.get("source_root","unknown")}`.',
        'Full file identity, sizes and SHA-256 are in provenance/source_manifest.json and run.json.',
        f'Group mapping: `{json.dumps(receipt.get("groups",{}),ensure_ascii=False)}`.', '',
        '## Method','',
        'Each model/checkpoint mode is analyzed separately. Original weights give each dataset equal mass within its group.',
        'Baseline normalized margin = (highest - second-highest logit) / population std of all class logits (ddof=0).',
        'Primary width 0.05; sensitivity widths 0.025 and 0.10. Bins are half-open [k*w,(k+1)*w).',
        'q[b] = min(p_focus[b],p_other[b]) / sum(min); sample weight = 1/(D_group*N_dataset) * q[b]/p_group[b].',
        'Identical weights apply to FP and Q correctness. Accuracy ratio = weighted Acc_Q / weighted Acc_FP.',
        'Accuracies, drops, damage/rescue masses and ratios are fractions. Gap is Other minus focus; multiply by 100 for ratio percentage points.',
        'damage_rate/rescue_rate in these tables are unconditional weighted sample masses; Q = FP - damage + rescue.',
        'Zero baseline accuracy and absent common support produce N/A. Missing input cells/checksum/pairing mismatches fail the run.', '']
    if tables is not None:
        t=tables['group_metrics'].query('bin_width == 0.05')
        lines += ['## Primary results','', '| Model | Mode | Condition | Group | Weighting | FP | Q | Ratio | Status |',
                  '|---|---|---|---|---|---:|---:|---:|---|']
        for r in t.itertuples():
            lines.append(f'| {r.model_name} | {r.checkpoint_mode} | {r.condition} | {r.group} | {r.weighting} | {number(r.baseline_accuracy)} | {number(r.quant_accuracy)} | {number(r.accuracy_ratio)} | {r.status} |')
        lines += ['',f'Validation: `{json.dumps(receipt.get("validation",{}))}`.','',
                  'Balance, ESS, support, weighted KS and dataset-composition changes are in tables/diagnostics.csv, balance.csv and dataset_weights.csv.']
    lines += ['', '## Limits','',
        'This is descriptive standardization of observed test samples, not a causal decomposition or a significance test.',
        'Bin proportions match, not individual continuous margins. Matching changes dataset masses and effective sample sizes.',
        'FP correctness, class count, checkpoint training, preprocessing and quantization perturbations can remain different.',
        'Conclusions cover only common support. Cotton is part of the requested comparison group, not a soybean dataset.',
        'Baseline precision is recorded per setting in coverage.csv; AMP-FP16 is not pure FP32. CAL retains exported evaluation aggregation.',
        'Checkpoint/test-CSV hashes are exporter declarations; original weights/images and inference are not re-executed.',
        'Models and modes reuse datasets and are not independent replications. No new inference or W&B download.', '']
    if receipt.get('error'):
        lines += ['## Failure','',receipt['error'],'']
    for path in figures:
        lines += [f'![{path.stem}](figures/{path.name})','']
    lines += ['## Reproduce','', '```bash',receipt['command'],'```','',
              f'Git: `{receipt.get("code",{}).get("commit")}`; dirty: `{receipt.get("code",{}).get("dirty")}`.',
              'Exact source copies: result/<run_id>/provenance/source/. Use a new run ID for reruns.',
              f'Full output: `{receipt["output_path"]}`.','']
    return '\n'.join(lines)


def run_analysis(args,argv):
    if args.chunk_size<=0:
        raise ValueError('chunk-size must be positive')
    workspace=args.workspace.expanduser().resolve()
    output,report=workspace/'result'/args.run_id,workspace/'reports'/args.run_id
    if output.exists() or report.exists():
        raise FileExistsError(f'Run/report exists: {args.run_id}; choose a new ID')
    output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    receipt=dict(schema=1,run_id=args.run_id,status='running',started_at=now(),timezone='Asia/Taipei',
        command=shlex.join(['conda','run','-n','opencode_env','python','-m','hqq_confidence.analyze_margin_matching',*argv]),
        parameters=vars(args),input_id=args.input_id,input_path=str(workspace/'data'/args.input_id),
        output_path=str(output),report_path=str(report/'REPORT.md'),
        environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,
                         matplotlib=matplotlib.__version__,scipy=server.scipy.__version__,sklearn=server.sklearn.__version__))
    tables=None
    figures=[]
    owns_report=False
    try:
        write_json(output/'run.json',receipt)
        report.mkdir(parents=True,exist_ok=False)
        owns_report=True
        receipt['code']=source_identity(output)
        root,meta,manifest_hash=read_input(args)
        if args.serial is not None and args.serial != meta['serial']:
            raise ValueError('Requested serial differs from imported snapshot')
        receipt.update(input_manifest_sha256=manifest_hash,serial=meta['serial'],source_root=meta['source_root'],
            inputs=meta['files'],source_format=meta['source_format'],selection=meta['selection'])
        chosen,label=focus(args,meta['selection']['datasets'])
        receipt['groups']={label:sorted(chosen.intersection(meta['selection']['datasets'])),
                           'other':sorted(set(meta['selection']['datasets'])-chosen)}
        shutil.copy2(root/'source_manifest.json',output/'provenance/source_manifest.json')
        loader={'server-w3-g128':load_server,'confidence-npz':load_confidence}.get(meta['source_format'])
        if loader is None: raise ValueError('Unsupported input format')
        collected={}
        coverage=[]
        settings=0
        for model,mode,samples,outcomes,conditions,setting_coverage in loader(root,meta,args):
            current,fits,weights=core.compute_tables(samples,outcomes,conditions)
            for name,table in current.items():
                table.insert(0,'checkpoint_mode',mode)
                table.insert(0,'model_name',model)
                collected.setdefault(name,[]).append(table)
            save=output/'prepared'/model/mode
            save.mkdir(parents=True)
            np.savez_compressed(save/'sample_weights.npz',**{k:v.astype(str) if v.dtype.kind=='O' else v for k,v in weights.items()})
            figures.append(plot_setting(samples,current,fits,output,model,mode,meta['serial']))
            coverage.extend(setting_coverage)
            settings+=1
        if not settings: raise ValueError('Empty selected matrix')
        tables={name:pd.concat(parts,ignore_index=True) for name,parts in collected.items()}
        (output/'tables').mkdir()
        for name,table in tables.items(): table.to_csv(output/'tables'/f'{name}.csv',index=False)
        pd.DataFrame(coverage).to_csv(output/'tables/coverage.csv',index=False)
        # Rehash every imported used file after computation, before publishing.
        if read_input(args)[2]!=manifest_hash:
            raise ValueError('Input manifest changed during analysis')
        current_code={name:server.sha256_file(REPO/name) for name in receipt['code']['files']}
        if current_code!=receipt['code']['files']:
            raise ValueError('Source code changed during analysis')
        for path in figures:
            destination=report/'figures'/path.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(path,destination)
            if server.sha256_file(path)!=server.sha256_file(destination):
                raise ValueError('Report figure hash mismatch')
        balance=tables['balance']
        statuses=tables['group_metrics'].status
        receipt['validation']=dict(settings=settings,source_files=len(meta['files']),
            max_histogram_balance_error=float(balance.balance_error.abs().max()) if len(balance) else None,
            undefined_estimates=int(statuses.ne('ok').sum()),
            source_hashes_rechecked=True,formula_checks=True,report_figure_hashes_equal=True)
        receipt.update(status='complete',finished_at=now(),elapsed_seconds=time.monotonic()-start)
        (report/'REPORT.md').write_text(report_text(receipt,tables,figures),encoding='utf-8')
        receipt['outputs']={p.relative_to(output).as_posix():server.sha256_file(p)
                            for p in sorted(output.rglob('*')) if p.is_file() and p.name!='run.json'}
        receipt['report_files']={p.relative_to(report).as_posix():server.sha256_file(p)
                                 for p in sorted(report.rglob('*')) if p.is_file()}
        write_json(output/'run.json',receipt)
    except BaseException as error:
        receipt.update(status='failed',finished_at=now(),elapsed_seconds=time.monotonic()-start,
                       error=f'{type(error).__name__}: {error}')
        # A report-only I/O error must not suppress failure publication.
        receipt.pop('report_files',None)
        write_json(output/'run.json',receipt)
        try:
            if owns_report:
                existing=[p for p in figures if (report/'figures'/p.name).is_file()]
                (report/'REPORT.md').write_text(report_text(receipt,figures=existing),encoding='utf-8')
        except OSError as report_error:
            receipt['report_error']=f'{type(report_error).__name__}: {report_error}'
            write_json(output/'run.json',receipt)
        raise
    print(f'COMPLETE: {report/"REPORT.md"}',flush=True)
    return receipt


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    args=parser().parse_args(argv)
    if args.command=='preflight-server':
        _,pairs=validate_server(args)
        focus(argparse.Namespace(grouping='soy-cotton'),args.datasets)
        print(f'PREFLIGHT COMPLETE: {len(pairs)} source pairs; no outputs written',flush=True)
    elif args.command=='import-server':
        focus(argparse.Namespace(grouping='soy-cotton'),args.datasets)
        import_server(args)
    elif args.command=='import-confidence':
        import_confidence(args)
    else:
        return run_analysis(args,argv)


if __name__=='__main__':
    main()
