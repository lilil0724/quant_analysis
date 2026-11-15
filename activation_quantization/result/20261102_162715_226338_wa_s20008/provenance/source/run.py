"""Run or read-only verify either supported activation quantization analysis."""
import argparse
import json
from pathlib import Path

from .storage import BASE, validate_id, load_manifest

SUPPORTED = (20007, 20008)


def resolve_serial(input_id, requested_serial=None):
    manifest = load_manifest(BASE / 'data' / validate_id(input_id))
    serial = manifest.get('serial')
    if serial not in SUPPORTED:
        raise ValueError(f'Unsupported activation serial: {serial}; expected 20007 or 20008')
    if requested_serial is not None and requested_serial != serial:
        raise ValueError(f'Requested serial {requested_serial} differs from input serial {serial}')
    return serial


def engine(serial):
    if serial == 20007:
        from . import fixed
        return fixed
    if serial == 20008:
        from . import grid
        return grid
    raise ValueError(f'Unsupported activation serial: {serial}')


def verify_run(run_id, requested_serial=None):
    validate_id(run_id)
    receipt_path = BASE / 'result' / run_id / 'run.json'
    if not receipt_path.is_file():
        raise FileNotFoundError(f'Full run receipt is unavailable: {receipt_path}; historical report-only runs require their original data/result backup')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    serial = resolve_serial(receipt['input_id'], requested_serial)
    if 'serial' in receipt and receipt['serial'] != serial:
        raise ValueError('Run and input serials differ')
    return engine(serial).verify_run(run_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--input-id', help='Verified input snapshot; serial is inferred from its manifest')
    action.add_argument('--verify-run', help='Verify an existing completed run without writing files')
    parser.add_argument('--serial', type=int, choices=SUPPORTED, help='Optional check against the input serial')
    parser.add_argument('--run-id', help='New run ID; generated in Asia/Taipei if omitted')
    args = parser.parse_args()
    if args.verify_run:
        if args.run_id:
            parser.error('--run-id cannot be used with --verify-run')
        print(json.dumps(verify_run(args.verify_run, args.serial), indent=2))
    else:
        serial = resolve_serial(args.input_id, args.serial)
        engine(serial).execute(args.input_id, args.run_id)


if __name__ == '__main__':
    main()
