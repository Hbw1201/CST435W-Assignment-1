"""Verify the published evidence without a GPU, dataset download or network connection."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify(root=ROOT):
    root = root.resolve()
    manifest = root/'results/SHA256SUMS'
    checked = 0
    for line in manifest.read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ',1)
        file = (root/name).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError('Missing or unsafe archive member: '+name)
        if hashlib.sha256(file.read_bytes()).hexdigest()!=expected:
            raise ValueError('SHA-256 mismatch: '+name)
        checked += 1
    for file in (root/'results').rglob('*.json'):
        record = json.loads(file.read_text(encoding='utf-8'))
        for field in ('sources','source_sha256'):
            hashes = record.get(field)
            if isinstance(hashes,dict):
                for name,expected in hashes.items():
                    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
                        raise ValueError('Measured source differs: '+name)
        for name,expected in record.get('csv_sha256',{}).items():
            if hashlib.sha256((file.parent/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('Measured CSV differs: '+str(file.parent/name))
    with (root/'results/main_repeat/runs.csv').open(newline='',encoding='utf-8') as stream:
        runs = list(csv.DictReader(stream))
    cells = Counter((r['session'],r['n_test'],r['mode']) for r in runs)
    if len(runs)!=135 or len(cells)!=27 or set(cells.values())!={5}:
        raise ValueError('Incomplete three-session experiment')
    if any(int(r['mismatch_vs_cpu']) for r in runs):
        raise ValueError('Prediction disagreement')
    boundary = json.loads((root/'results/boundary_validation.json').read_text())
    clean = json.loads((root/'results/clean_environment.json').read_text())
    if boundary['checks_passed']!=50 or clean['status']!='passed' or clean['boundary']['checks_passed']!=18:
        raise ValueError('Archived validation is incomplete')
    result = dict(verified_files=checked,main_requests=135,process_sessions=3,
                  archived_cpu_gpu_checks=50,archived_clean_cpu_checks=18,
                  network_required=False,gpu_required=False)
    print(json.dumps(result))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    verify(args.root)
