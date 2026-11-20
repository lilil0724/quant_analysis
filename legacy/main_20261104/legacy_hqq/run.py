"""Run the legacy accuracy-ratio pipeline from the repository root."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(__file__).resolve().parent
STAGES = ('acquire', 'prepare', 'analyze', 'plot')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def invoke(module: str, *args: object) -> None:
    command = [sys.executable, '-m', f'legacy_hqq.{module}', *(str(arg) for arg in args)]
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(command, cwd=BASE.parent, check=True)


def acquire(raw: Path, source: Path | None, refresh: bool, serials: list[int] | None) -> None:
    if source is not None:
        if not source.is_file():
            raise FileNotFoundError(source)
        if raw.exists() and sha256(raw) != sha256(source) and not refresh:
            raise FileExistsError(f'{raw} already has different contents; use --refresh to replace it')
        raw.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != raw.resolve():
            shutil.copy2(source, raw)
        manifest = {
            'source': str(source.resolve()),
            'destination': str(raw),
            'sha256': sha256(raw),
            'imported_at': datetime.now(timezone.utc).isoformat(),
        }
        (raw.parent / 'source_manifest.json').write_text(
            json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'Imported {raw} ({manifest["sha256"]})')
        return
    if raw.is_file() and not refresh:
        print(f'Using existing {raw} ({sha256(raw)})')
        return
    arguments: list[object] = ['--output-file', raw, '--results-dir', raw.parent]
    if serials:
        arguments.extend(['--serials', *serials])
    invoke('download_save_wandb_data', *arguments)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=(*STAGES, 'all'), default='all')
    parser.add_argument('--run-name', default='main')
    parser.add_argument('--source-file', type=Path,
                        help='existing raw CSV to import into this analysis')
    parser.add_argument('--refresh', action='store_true',
                        help='replace an existing raw CSV with --source-file or a new download')
    parser.add_argument('--serials', nargs='+', type=int,
                        help='optional explicit HQQ and baseline serials')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name):
        parser.error('--run-name must contain only letters, digits, dot, underscore, or hyphen')

    raw = BASE / 'data' / 'raw' / 'backbones_quant.csv'
    processed = BASE / 'data' / 'processed' / args.run_name
    results = BASE / 'results' / args.run_name
    stages = STAGES if args.stage == 'all' else (args.stage,)
    for stage in stages:
        if stage == 'acquire':
            acquire(raw, args.source_file, args.refresh, args.serials)
        elif stage == 'prepare':
            if not raw.is_file():
                raise FileNotFoundError(f'Missing {raw}; run --stage acquire first')
            arguments: list[object] = [
                '--input-file', raw,
                '--output-file', processed / 'quant_summary.csv',
                '--results-dir', results / 'summary',
            ]
            if args.serials:
                arguments.extend(['--serials', *args.serials])
            invoke('summarize_quant', *arguments)
        elif stage == 'analyze':
            summary = processed / 'quant_summary.csv'
            if not summary.is_file():
                raise FileNotFoundError(f'Missing {summary}; run --stage prepare first')
            arguments = [
                '--input-file', summary,
                '--output-file', results / 'corr' / 'factor_ablation_accuracy.csv',
                '--results-dir', results / 'corr',
                '--prepared-output', processed / 'prepared_accuracy_cells.csv',
            ]
            if args.serials:
                arguments.extend(['--serials', *args.serials])
            invoke('corr_quant', *arguments)
        else:
            prepared = processed / 'prepared_accuracy_cells.csv'
            ablation = results / 'corr' / 'factor_ablation_accuracy.csv'
            if not prepared.is_file() or not ablation.is_file():
                raise FileNotFoundError(
                    f'Missing {prepared} or {ablation}; run --stage analyze first')
            invoke('plot_quant_acc_bit_group',
                   '--input-file', prepared,
                   '--ablation-file', ablation,
                   '--corr-dir', results / 'corr',
                   '--results-dir', results / 'plots',
                   '--output-file', results / 'plots' / 'quant_accuracy_overview.png')


if __name__ == '__main__':
    main()
