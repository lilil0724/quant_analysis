"""Run within-DAFB diagnostics on matched TGDA W3/G128 exports."""
import argparse
import itertools
import os
import re
import shlex
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import server_bias_competitors as bias
from .margins import assign_margin_deciles, binary_auroc
from . import centered_class_distance as centered, paired, paired_inputs, tsne_paired
from .common import ANALYSES, sha256, write_csv
from .prepared import write_prepared
from .provenance import RAW_ROOT, REPO_ROOT, environment, git_identity, raw_source_files, snapshot_source, write_receipt

TAIPEI = timezone(timedelta(hours=8), 'Asia/Taipei')


def stamp(path):
    info = path.stat()
    return info.st_size, info.st_mtime_ns


def preflight(args, model, mode, out, reports):
    folder, serial = paired_inputs.select_source(
        args.input_root / str(args.serial) / 'dafb' / model / mode,
        args.serial, 'dafb', model, mode, args.source_map)
    folder = folder.resolve()
    if any(folder.is_relative_to(path) or path.is_relative_to(folder) for path in (out, reports)):
        raise ValueError(f'Output or report overlaps DAFB export: {folder}')
    print(f'VALIDATING {folder}', flush=True)
    stages = set(args.analyses)
    features = ('all' if 'paired' in stages else
                ('final',) if stages & {'tsne', 'centered'} else ())
    margins = bool(stages & {'paired', 'margin', 'bias'})
    pair = paired_inputs.read_pair(folder, serial, 'dafb', model, mode,
                                   features=features, fp_only=bool(features) and not stages & {'paired', 'tsne'},
                                   predictions=margins, margins=margins)
    if features and 'final' not in pair['meta']['layers']:
        raise ValueError(f'{folder}: final features are required for PCA and t-SNE')
    if 'bias' in stages and len(pair['targets']) < bias.core.FOLDS:
        raise ValueError(f'{folder}: at least {bias.core.FOLDS} samples are required for bias cross-fitting')
    # Keep the TGDA stored winner for ties, as verified against each logit row.
    return dict(pair, folder=folder, dataset='dafb', model=model, mode=mode,
                stamps={name: stamp(folder / name) for name in pair['hashes']})


def margin_tables(pair, destination):
    fp = pair['predictions']['fp']
    quant = pair['predictions']['w3_g128']
    margins = np.array([float(row['normalized_margin']) for row in fp])
    deciles = assign_margin_deciles(margins)
    samples = []
    for f, q, decile in zip(fp, quant, deciles):
        f_correct, q_correct = bool(int(f['correct'])), bool(int(q['correct']))
        samples.append(dict(
            sample_id=f['sample_id'], target=int(f['target']), fp_prediction=int(f['prediction']),
            w3_prediction=int(q['prediction']), fp_correct=int(f_correct), w3_correct=int(q_correct),
            swap=int(f['prediction'] != q['prediction']), damage=int(f_correct and not q_correct),
            rescue=int(not f_correct and q_correct), fp_margin_decile=int(decile),
            fp_normalized_margin=float(f['normalized_margin']),
            w3_normalized_margin=float(q['normalized_margin']),
            fp_true_class_margin=float(f['true_class_margin']),
            w3_true_class_margin=float(q['true_class_margin']),
        ))
    write_csv(destination / 'sample_margins.csv', samples)
    identity = dict(model=pair['model'], mode=pair['mode'])
    bins = []
    for decile in sorted(set(deciles)):
        rows = [row for row in samples if row['fp_margin_decile'] == decile]
        n_fp_correct = sum(row['fp_correct'] for row in rows)
        n_damage = sum(row['damage'] for row in rows)
        n_rescue = sum(row['rescue'] for row in rows)
        n_fp_wrong = len(rows) - n_fp_correct
        bins.append(dict(**identity, fp_margin_decile=int(decile), n=len(rows),
                         fp_correct_n=n_fp_correct, fp_wrong_n=n_fp_wrong,
                         damage_n=n_damage, rescue_n=n_rescue,
                         swap_rate=sum(row['swap'] for row in rows) / len(rows),
                         damage_given_fp_correct=n_damage / n_fp_correct if n_fp_correct else None,
                         rescue_given_fp_wrong=n_rescue / n_fp_wrong if n_fp_wrong else None))
    scores = []
    for event, eligible in (
        ('swap', np.ones(len(fp), dtype=bool)),
        ('damage', np.array([row['fp_correct'] for row in samples], dtype=bool)),
        ('rescue', np.array([not row['fp_correct'] for row in samples], dtype=bool)),
    ):
        observed = np.array([row[event] for row in samples], dtype=bool)[eligible]
        auc, status = (binary_auroc(-margins[eligible], observed) if eligible.any()
                       else (None, 'undefined_empty_cohort'))
        scores.append(paired.clean(dict(**identity, event=event, n=len(observed),
                                        events=int(observed.sum()), auroc=auc, status=status)))
    return bins, scores


def analyze(pair, out, args):
    model, mode, folder = pair['model'], pair['mode'], pair['folder']
    destination = out / 'dafb' / model / mode
    tables = {}
    stages = set(args.analyses)
    if 'paired' in stages:
        tables['all_pairs'] = [paired.analyze_pair(pair, out, args.serial)]
    if 'margin' in stages:
        tables['margin_deciles'], tables['margin_auroc'] = margin_tables(pair, destination)
    targets = np.asarray(pair['targets'])
    if 'centered' in stages:
        final = np.load(paired_inputs.pair_file(folder, 'fp', 'final.npy'), mmap_mode='r', allow_pickle=False)
        distances, mean = centered.analyze_features(final, targets)
        del final
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / 'fp_global_mean.npy', mean)
        tables['centered_class_distance'] = [dict(model=model, mode=mode, **distances)]
    if 'tsne' in stages:
        source_pair = tsne_paired._source_pair(folder, pair['meta']['serial'], 'dafb', model, mode)
        selected = tsne_paired.sample_indices(targets, args.tsne_max_samples, seed=100)
        tsne_paired._render_pair(source_pair, destination / 'tsne', selected, args.tsne_max_samples,
                                 30.0, 750, 100, False, 'samples', None)
    if 'bias' in stages:
        settings = SimpleNamespace(serial=args.serial, seed=100, chunk_size=256, save_sample_results=True)
        bias_row, ranks, folds, _ = bias.analyze_pair(pair, out / 'bias', settings)
        tables.update(classifier_bias=[paired.clean(bias_row)], competitor_ranks=[paired.clean(ranks)],
                      bias_folds=[paired.clean(row) for row in folds])
    for name, expected in pair['hashes'].items():
        if sha256(folder / name) != expected:
            raise ValueError(f'Source changed during analysis: {folder / name}')
    return tables


def write_report(report, args, run, tables):
    lines = [
        '# DAFB W3/G128 analysis', '',
        f'- Started at {run["started_at"]} (Asia/Taipei).',
        f'- Inputs are {len(run["inputs"])} TGDA schema-2 pairs. The run.json records their resolved paths and checksums.',
        f'- Selected analyses are {", ".join(args.analyses)}.',
        '- The baseline is the matched unquantized AMP export. The comparison uses Linear weight-only W3/G128.',
        '- Layer geometry uses all paired samples. PCA uses the FP final-layer basis. Joint t-SNE uses sampled matched rows.',
        '- Damage is FP-correct to W3-wrong. Rescue is FP-wrong to W3-correct. Deciles use the FP top-1/top-2 gap divided by the logit standard deviation.',
        '- Fold-held-out bias subtraction and competitor ranks are descriptive checks within DAFB.',
        '- CAL features come from the raw-image backbone call. CAL logits may aggregate crop/flip views.',
    ]
    if 'all_pairs' in tables:
        lines.extend(['', '| Backbone | Mode | FP top-1 | W3 top-1 | Damage | Rescue | Final CKA |',
                      '|---|---|---:|---:|---:|---:|---:|'])
    for row in tables.get('all_pairs', ()):
        value = row['final_cka']
        cka = f'{value:.4f}' if value is not None else 'N/A'
        lines.append(f'| {row["model"]} | {row["mode"]} | {row["fp_accuracy"]:.2%} | '
                     f'{row["accuracy"]:.2%} | {row["damage_count"]} | {row["rescue_count"]} | {cka} |')
    lines.extend(['', f'Run artifacts and input checksums are in {run["output"]}.',
                  '', 'Reproduce from the repository root with a fresh run ID.', '', '~~~bash',
                  run['command'], '~~~', ''])
    report.write_text('\n'.join(lines), encoding='utf-8')


def main(argv=None):
    started = datetime.now(TAIPEI)
    parser = argparse.ArgumentParser(description='Run the paired DAFB analyses for all nine W3/G128 exports.')
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--serial', type=int, default=20004)
    parser.add_argument('--models', nargs='+', choices=paired.MODELS, default=paired.MODELS)
    parser.add_argument('--modes', nargs='+', choices=paired.MODES, default=paired.MODES)
    parser.add_argument('--analyses', nargs='+', choices=ANALYSES, default=ANALYSES)
    parser.add_argument('--source-map', type=Path)
    parser.add_argument('--result-root', type=Path, default=RAW_ROOT/'result')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--report-root', type=Path, default=RAW_ROOT/'reports')
    parser.add_argument('--run-id', default=started.strftime('%Y%m%d_%H%M%S_%f_dafb'))
    parser.add_argument('--tsne-max-samples', type=int, default=2000)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    if args.serial <= 0 or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', args.run_id) or args.tsne_max_samples < 2:
        parser.error('Use a positive serial, a simple run ID and a t-SNE sample limit of at least two.')
    args.input_root = args.input_root.resolve()
    if args.input_root.name == str(args.serial):
        args.input_root = args.input_root.parent
    out = (args.output_root or args.result_root / args.run_id).resolve()
    reports = (args.report_root / args.run_id).resolve()
    source = (args.input_root / str(args.serial) / 'dafb').resolve()
    selection = list(itertools.product(dict.fromkeys(args.models), dict.fromkeys(args.modes)))
    for left, right in ((out, reports), (out, source), (reports, source)):
        if left.is_relative_to(right) or right.is_relative_to(left):
            parser.error('The run, report and source directories must be separate.')
    if not args.preflight_only and (out.exists() or reports.exists()):
        parser.error(f'Run or report directory already exists: {out} or {reports}')
    if args.preflight_only:
        for model, mode in selection:
            preflight(args, model, mode, out, reports)
        print(f'PREFLIGHT OK: {len(selection)} DAFB pairs', flush=True)
        return

    out.mkdir(parents=True)
    report = reports / 'REPORT.md'
    options = ['--input-root', str(args.input_root), '--serial', str(args.serial),
               '--models', *args.models, '--modes', *args.modes, '--tsne-max-samples', str(args.tsne_max_samples),
               '--analyses', *args.analyses, '--result-root', str(args.result_root.resolve()),
               '--report-root', str(reports.parent)]
    if args.source_map:
        options.extend(['--source-map', str(args.source_map.resolve())])
    run = dict(
        status='running', started_at=started.isoformat(), timezone='Asia/Taipei', output=str(out),
        command=shlex.join([sys.executable, '-B', '-m', 'codebase.raw.dafb', *options]),
        run_id=args.run_id, slurm_job=os.getenv('SLURM_JOB_ID'), submit_dir=os.getenv('SLURM_SUBMIT_DIR'),
        parameters=vars(args), environment=environment(), git=git_identity(), inputs=[], completed_pairs=[],
    )
    receipt = out / 'run.json'
    write_receipt(receipt, run)
    try:
        run['code_sha256'] = snapshot_source(out, raw_source_files())
        pairs = []
        for model, mode in selection:
            pair = preflight(args, model, mode, out, reports)
            pairs.append(pair)
            run['inputs'].append(dict(folder=str(pair['folder']), model=model, mode=mode, ties=pair['ties'],
                                      metadata=pair['meta'], sizes={n: v[0] for n, v in pair['stamps'].items()}))
            write_receipt(receipt, run)
        print(f'PREFLIGHT OK: {len(pairs)} DAFB pairs', flush=True)
        reports.mkdir(parents=True)
        tables = {}
        for pair in pairs:
            for key, rows in analyze(pair, out, args).items():
                tables.setdefault(key, []).extend(rows)
            run['completed_pairs'].append(f'{pair["model"]}/{pair["mode"]}')
            write_receipt(receipt, run)
        for key, rows in tables.items():
            write_csv(out / f'{key}.csv', rows)
        write_prepared(out, pairs, tables, args.analyses)
        write_report(report, args, run, tables)
        if any(sha256(REPO_ROOT / name) != value for name, value in run['code_sha256'].items()):
            raise RuntimeError('Analysis code changed during this run')
        run['outputs'] = {path.relative_to(out).as_posix(): sha256(path)
                          for path in sorted(out.rglob('*')) if path.is_file() and path != receipt}
        run['report'] = {'path': str(report), 'sha256': sha256(report)}
        run['status'] = 'complete'
    except BaseException as error:
        run.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        finished = datetime.now(TAIPEI)
        run['finished_at'] = finished.isoformat()
        run['elapsed_seconds'] = (finished - started).total_seconds()
        write_receipt(receipt, run)
    print(f'COMPLETE: {out}', flush=True)


if __name__ == '__main__':
    main()
