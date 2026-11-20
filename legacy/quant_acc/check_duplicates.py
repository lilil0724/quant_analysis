"""Compare the top-1 of duplicate runs (same setting) in a rerun_serial_grid run; write duplicates.csv."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .recover_baselines import context

TOLERANCE = 0.1  # top-1 points; FP16 inference noise on one checkpoint and test set


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path, help='quant_acc/result/<run_id>')
    args = parser.parse_args()
    run = json.loads((args.run_dir / 'run.json').read_text(encoding='utf-8'))
    records = {}
    for line in (Path(run['input_dir']) / 'runs.jsonl').read_text(encoding='utf-8').splitlines():
        record = json.loads(line)
        records[record['id']] = record
    audit = pd.read_csv(args.run_dir / 'audit/pair_audit.csv')
    excluded = pd.read_csv(args.run_dir / 'audit/excluded_runs.csv')
    rows = []
    for key, part in audit[audit.reason.isin(['selected', 'older_compatible_condition'])].groupby(
            ['dataset', 'model', 'mode', 'nbits', 'group_size']):
        if len(part) > 1:
            kept = part[part.accepted].quant_run_id.iloc[0]
            for other in part[~part.accepted].quant_run_id:
                rows.append(dict(zip(['dataset', 'model', 'mode', 'nbits', 'group_size'], key), kind='hqq',
                                 kept_run=kept, other_run=other,
                                 kept_top1=records[kept]['summary']['hqq_top1'],
                                 other_top1=records[other]['summary']['hqq_top1']))
    coverage = pd.read_csv(args.run_dir / 'audit/context_coverage.csv').set_index(['dataset', 'model', 'mode'])
    for other in excluded[excluded.reason == 'older_baseline'].run_id:
        key = context(records[other])
        kept = coverage.loc[key, 'baseline_run_id']
        rows.append(dict(zip(['dataset', 'model', 'mode'], key), kind='baseline', kept_run=kept, other_run=other,
                         kept_top1=records[kept]['summary']['top1'], other_top1=records[other]['summary']['top1']))
    table = pd.DataFrame(rows)
    table['difference'] = (table.kept_top1 - table.other_top1).abs()
    table['conflict'] = ~(table.difference <= TOLERANCE)
    table.to_csv(args.run_dir / 'audit/duplicates.csv', index=False)
    print(f'{len(table)} duplicate pairs ({(table.kind == "hqq").sum()} HQQ, {(table.kind == "baseline").sum()} baseline); '
          f'max |difference| {table.difference.max():.3f} points; conflicts above {TOLERANCE}: {int(table.conflict.sum())}')


if __name__ == '__main__':
    main()
