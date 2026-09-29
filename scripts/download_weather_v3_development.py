#!/usr/bin/env python3
"""Bounded, two-stage KNMI lc1 acquisition for approved v3 development years.

This script must not be run before the user approves the exact development
scope in configs/weather_v3_expanded_acquisition_proposal.json. It never
fetches the sealed 2015/2025 holdout years.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path('/private/tmp/slowlab-v22-cabauw-gapfilled')
PLAN = ROOT / 'configs/weather_v3_expanded_acquisition_proposal.json'
MANIFEST = CACHE / 'manifest-v3-development.json'
OLD_MANIFEST = CACHE / 'manifest.json'


def request(url, headers=None, limit=1_000_000):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}),
                                        timeout=30) as response:
                body = response.read(limit + 1)
                if len(body) > limit:
                    raise RuntimeError('metadata response exceeds limit')
                return body
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 5:
                raise
            retry = error.headers.get('Retry-After')
            delay = min(90, max(20, float(retry))) if retry and retry.isdigit() else min(90, 20 * (attempt + 1))
            time.sleep(delay)
    raise AssertionError('unreachable')


def public_key():
    page = request('https://developer.dataplatform.knmi.nl/open-data-api').decode('utf-8')
    matches = re.findall(r'eyJ[A-Za-z0-9_=-]{60,}', page)
    if not matches:
        raise RuntimeError('public KNMI anonymous access token unavailable')
    return matches[0]


def selected_rows(plan, stage):
    development = plan['stages']['development']
    rows = development['files']
    if (development['file_count'] != 96 or development['net_new_bytes_cap'] != 13_027_903
            or len(rows) != 96 or sum(row['size'] for row in rows) != development['catalog_bytes']):
        raise RuntimeError('development download scope changed')
    for row in rows:
        if ('/' in row['filename'] or '..' in row['filename']
                or not row['dataset'].startswith('cesar_surface_')
                or row['size'] <= 0):
            raise RuntimeError('invalid catalog file identity')
        year = int(row['filename'].rsplit('_', 1)[-1][:4])
        if year not in (2012, 2013, 2014, 2016):
            raise RuntimeError('sealed or undeclared year in development scope')
    if stage == 'sample':
        return [row for row in rows if row['filename'].endswith((
            '_201201.nc', '_201301.nc', '_201401.nc', '_201601.nc'))]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('sample', 'remaining'), required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    rows = selected_rows(plan, args.stage)
    full_rows = plan['stages']['development']['files']
    allowed = {row['filename']: row for row in full_rows}
    sample = {row['filename'] for row in selected_rows(plan, 'sample')}
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {
        'scope': 'v3 development weather only; no 2015/2025 holdout', 'files': {},
        'net_new_bytes_downloaded': 0}
    old = json.loads(OLD_MANIFEST.read_text())
    if args.stage == 'remaining':
        audit = ROOT / 'results/weather_v3_development_sample_audit.json'
        if not audit.exists() or json.loads(audit.read_text()).get('gate') != 'pass':
            raise RuntimeError('sample quality audit has not passed')
        if not sample.issubset(manifest['files']):
            raise RuntimeError('all sample files must be verified first')
    cap = plan['stages']['development']['net_new_bytes_cap']
    token = None
    for row in rows:
        filename = row['filename']
        target = CACHE / row['dataset'] / filename
        if filename in manifest['files']:
            entry = manifest['files'][filename]
            if (not target.is_file() or target.stat().st_size != row['size']
                    or hashlib.sha256(target.read_bytes()).hexdigest() != entry['sha256']):
                raise RuntimeError('existing v3 cache file fails manifest: ' + filename)
            continue
        if filename in old['files']:
            entry = old['files'][filename]
            if (not target.is_file() or target.stat().st_size != row['size']
                    or hashlib.sha256(target.read_bytes()).hexdigest() != entry['sha256']):
                raise RuntimeError('inherited 2016 boundary file fails old manifest')
            manifest['files'][filename] = {'dataset': row['dataset'], 'bytes': row['size'],
                                           'sha256': entry['sha256'], 'source': 'verified existing cache'}
            MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
            continue
        if manifest['net_new_bytes_downloaded'] + row['size'] > cap:
            raise RuntimeError('approved development byte cap exceeded')
        if token is None:
            token = public_key()
        endpoint = (f"https://api.dataplatform.knmi.nl/open-data/v1/datasets/{row['dataset']}"
                    f"/versions/v1.0/files/{urllib.parse.quote(filename)}/url")
        metadata = json.loads(request(endpoint, {'Authorization': token}))
        url = metadata.get('temporaryDownloadUrl')
        if not isinstance(url, str) or not url.startswith('https://'):
            raise RuntimeError('invalid temporary download URL')
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix('.v3part')
        received = 0
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(url, timeout=45) as response, partial.open('wb') as output:
                declared = response.headers.get('Content-Length')
                if declared is not None and int(declared) != row['size']:
                    raise RuntimeError('remote catalog size changed: ' + filename)
                while True:
                    chunk = response.read(min(65536, row['size'] - received + 1))
                    if not chunk:
                        break
                    received += len(chunk)
                    if (received > row['size']
                            or manifest['net_new_bytes_downloaded'] + received > cap):
                        raise RuntimeError('file or total byte cap exceeded')
                    output.write(chunk)
                    digest.update(chunk)
            if received != row['size']:
                raise RuntimeError('truncated file: ' + filename)
            partial.replace(target)
        finally:
            if partial.exists():
                partial.unlink()
        manifest['files'][filename] = {'dataset': row['dataset'], 'bytes': received,
                                       'sha256': digest.hexdigest(), 'source': 'KNMI Open Data API'}
        manifest['net_new_bytes_downloaded'] += received
        MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps({'stage': args.stage, 'file': filename,
                          'net_new_bytes_downloaded': manifest['net_new_bytes_downloaded']}), flush=True)
        time.sleep(2.1)
    if args.stage == 'remaining' and set(manifest['files']) != set(allowed):
        raise RuntimeError('development file set incomplete')
    print(json.dumps({'status': 'stage_complete', 'stage': args.stage,
                      'verified_files': len(manifest['files']),
                      'net_new_bytes_downloaded': manifest['net_new_bytes_downloaded']}))


if __name__ == '__main__':
    main()
