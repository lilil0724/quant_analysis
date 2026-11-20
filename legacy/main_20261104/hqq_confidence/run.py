"""Run the matched-sample HQQ confidence pipeline by stage."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parent
STAGES = ('acquire', 'prepare', 'analyze', 'plot')


def invoke(module: str, *args: object) -> None:
    command = [sys.executable, '-m', f'hqq_confidence.{module}',
               *(str(value) for value in args)]
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(command, cwd=BASE.parent, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=(*STAGES, 'all'), default='all')
    parser.add_argument('--run-name', default='main')
    parser.add_argument('--refresh', action='store_true',
                        help='download again even when the selected-runs manifest exists')
    parser.add_argument('--serials', nargs='+', type=int,
                        help='explicit serials used only when downloading')
    parser.add_argument('--allow-unknown-provenance', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name):
        parser.error('--run-name must contain only letters, digits, dot, underscore, or hyphen')

    data = BASE / 'data' if args.run_name == 'main' else BASE / 'data' / args.run_name
    results = BASE / 'results' if args.run_name == 'main' else BASE / 'results' / args.run_name
    manifest = data / 'selected_runs.csv'
    prepared = BASE / 'data' / 'processed' / args.run_name
    stats = results / 'stats'
    plots = results / 'plots'
    stages = STAGES if args.stage == 'all' else (args.stage,)
    for stage in stages:
        if stage == 'acquire':
            if manifest.is_file() and not args.refresh:
                print(f'Using existing {manifest}')
                continue
            arguments: list[object] = [
                '--output-file', manifest,
                '--cache-dir', data / 'artifacts',
                '--results-dir', data,
            ]
            if args.serials:
                arguments.extend(['--serials', *args.serials])
            invoke('download_artifacts', *arguments)
        elif stage == 'prepare':
            if not manifest.is_file():
                raise FileNotFoundError(f'Missing {manifest}; run --stage acquire first')
            arguments = ['--manifest-file', manifest, '--output-dir', prepared]
            if args.allow_unknown_provenance:
                arguments.append('--allow-unknown-provenance')
            invoke('prepare_pairs', *arguments)
        elif stage == 'analyze':
            if not (prepared / 'condition_summary.csv').is_file():
                raise FileNotFoundError(f'Missing {prepared}; run --stage prepare first')
            invoke('analyze_fragility', '--input-dir', prepared, '--results-dir', stats)
        else:
            if not (stats / 'condition_statistics.csv').is_file():
                raise FileNotFoundError(f'Missing {stats}; run --stage analyze first')
            invoke('plot_fragility', '--prepared-dir', prepared,
                   '--stats-dir', stats, '--results-dir', plots)


if __name__ == '__main__':
    main()
