#!/usr/bin/env python3
"""Bounded KNMI lc1 acquisition of the approved formal weather years.

Scope is exactly the ``recommended`` stage of
configs/weather_formal_acquisition_proposal_v0.json (2002-2011 and 2015, plus
the 2001-12 boundary files), approved by the user on 2026-09-30. Stage
``sample`` fetches January 2002 and January 2015 for both datasets; stage
``remaining`` refuses to run until the sample audit has passed. Every file must
match the listed size; the manifest records its SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from download_weather_v3_development import public_key, request  # noqa: E402

PLAN = ROOT / 'configs/weather_formal_acquisition_proposal_v0.json'
SAMPLE_MONTHS = ('200201', '201501')
YEARS = set(range(2002, 2012)) | {2015}
PAUSE_SECONDS = 6.0  # the anonymous KNMI key is shared and rate-limited
# The first sample audit (results/weather_formal_sample_audit.json) failed on one
# 2015 record; the gate is the re-audit under repair rule v0.
SAMPLE_AUDIT = ROOT / 'results/weather_formal_sample_audit_repair_v0.json'


def approved_rows(plan):
    stage = plan['stages']['recommended']
    rows = stage['files'] + stage['boundary_files']
    if (len(rows) != 266 or stage['file_count'] != 266 or stage['total_bytes'] != 34_871_543
            or sum(r['size'] for r in rows) != stage['total_bytes'] or set(stage['years']) != YEARS):
        raise RuntimeError('approved formal download scope changed')
    for row in rows:
        month = row['filename'].rsplit('_', 1)[-1][:6]
        if ('/' in row['filename'] or '..' in row['filename'] or row['size'] <= 0
                or row['dataset'] not in ('cesar_surface_meteo_lc1_t10', 'cesar_surface_radiation_lc1_t10')
                or not (int(month[:4]) in YEARS or month == '200112')):
            raise RuntimeError('file outside the approved scope: ' + row['filename'])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('sample', 'remaining'), required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    rows = approved_rows(json.loads(PLAN.read_text()))
    sample = [r for r in rows if r['filename'].endswith(tuple(f'_{m}.nc' for m in SAMPLE_MONTHS))]
    assert len(sample) == 4
    manifest_path = args.cache / 'manifest-formal-v0.json'
    args.cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'scope': PLAN.name + ' stage recommended', 'files': {}, 'bytes_downloaded': 0}
    if args.stage == 'remaining':
        if not SAMPLE_AUDIT.exists() or json.loads(SAMPLE_AUDIT.read_text()).get('gate') != 'pass':
            raise RuntimeError('sample audit has not passed')
    todo = sample if args.stage == 'sample' else rows
    cap = sum(r['size'] for r in rows)
    token = None
    for row in todo:
        name = row['filename']
        target = args.cache / row['dataset'] / name
        if name in manifest['files']:
            if (not target.is_file() or target.stat().st_size != row['size']
                    or hashlib.sha256(target.read_bytes()).hexdigest() != manifest['files'][name]['sha256']):
                raise RuntimeError('cached file fails manifest: ' + name)
            continue
        if manifest['bytes_downloaded'] + row['size'] > cap:
            raise RuntimeError('approved byte cap exceeded')
        token = token or public_key()
        endpoint = (f"https://api.dataplatform.knmi.nl/open-data/v1/datasets/{row['dataset']}"
                    f"/versions/v1.0/files/{urllib.parse.quote(name)}/url")
        for attempt in range(8):
            try:
                url = json.loads(request(endpoint, {'Authorization': token})).get('temporaryDownloadUrl')
                break
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == 7:
                    raise
                print(json.dumps({'rate_limited': name, 'wait_seconds': 300}), flush=True)
                time.sleep(300)
        if not isinstance(url, str) or not url.startswith('https://'):
            raise RuntimeError('invalid temporary download URL')
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix('.part')
        digest, received = hashlib.sha256(), 0
        try:
            with urllib.request.urlopen(url, timeout=45) as response, partial.open('wb') as output:
                declared = response.headers.get('Content-Length')
                if declared is not None and int(declared) != row['size']:
                    raise RuntimeError('remote size changed: ' + name)
                while chunk := response.read(min(65536, row['size'] - received + 1)):
                    received += len(chunk)
                    if received > row['size']:
                        raise RuntimeError('file larger than listed: ' + name)
                    output.write(chunk)
                    digest.update(chunk)
            if received != row['size']:
                raise RuntimeError('truncated file: ' + name)
            partial.replace(target)
        finally:
            if partial.exists():
                partial.unlink()
        manifest['files'][name] = {'dataset': row['dataset'], 'bytes': received, 'sha256': digest.hexdigest(),
                                   'source': 'KNMI Open Data API'}
        manifest['bytes_downloaded'] += received
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps({'stage': args.stage, 'file': name, 'bytes_downloaded': manifest['bytes_downloaded']}), flush=True)
        time.sleep(PAUSE_SECONDS)
    if args.stage == 'remaining' and set(manifest['files']) != {r['filename'] for r in rows}:
        raise RuntimeError('formal file set incomplete')
    print(json.dumps({'status': 'stage_complete', 'stage': args.stage, 'verified_files': len(manifest['files']),
                      'bytes_downloaded': manifest['bytes_downloaded']}))


if __name__ == '__main__':
    main()
