#!/usr/bin/env python3
"""Assemble the repaired formal dataset from the original run and the repair outputs (decision 2026-10-05).

    assemble_repaired_run.py core      --formal F --repair R --out A
    assemble_repaired_run.py e3-reuse  --formal F --out A        (after: e3_readers.py prepare and gp on A)
    assemble_repaired_run.py manifest  --formal F --out A

core: A/campaigns.json is the formal batch pointing at A/campaigns; records are the original ones, except
every main-baseline record (replaced by the repair rerun) and every repaired LLM record (original record
with its Full branch replaced and a 'repair' block; the Endpoint branch is the original). A/evaluations
holds the original evaluation records with the repaired ones replacing those of the same name. A/phase6
holds the original picks, reference searches and sensitivity records, the repaired sensitivity records
replacing those of the same name. A/ledger.jsonl is the original ledger followed by the repair ledger.
e3-reuse: after the E3 packets and GP Reader were rebuilt in A/e3 from the assembled records, copy from the
original E3 run every LLM Reader record whose packet is byte-identical, and every Reader evaluation whose
packet site and recommended policy are identical; what remains is run on the server.
manifest: sha256 of every assembled file with its provenance (original or repair).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('core', 'e3-reuse', 'manifest'))
    ap.add_argument('--formal', type=Path, required=True)
    ap.add_argument('--repair', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    F, A = args.formal.resolve(), args.out.resolve()
    prov = {}
    if args.stage == 'core':
        R = args.repair.resolve()
        batch = json.loads((F / 'campaigns.json').read_text())
        repaired = set(json.loads((R / 'llm_jobs.json').read_text()))
        (A / 'campaigns').mkdir(parents=True, exist_ok=True)
        for job in batch['jobs']:
            dst = A / 'campaigns' / f"{job['id']}.json"
            if job['method'] == 'bo':
                shutil.copyfile(R / 'baseline' / 'campaigns' / f"{job['id']}.json", dst)
                prov[str(dst.relative_to(A))] = 'repair'
            elif job['id'] in repaired:
                rec = json.loads((F / 'campaigns' / f"{job['id']}.json").read_text())
                new = json.loads((R / 'llm' / job['id'] / 'record.json').read_text())
                if new.get('status') != 'completed':
                    raise SystemExit(f"repair of {job['id']} did not complete")
                if new['initial_recommendation'] != rec['initial_recommendation']:
                    raise SystemExit(f"{job['id']}: replayed initial recommendation differs")
                rec['branches']['full'] = new['branches']['full']
                rec['repair'] = {**new['repair'], 'repair_outbound_audit': new['outbound_audit'],
                                 'repair_provider_call_records': new['provider_call_records']}
                dst.write_text(json.dumps(rec) + '\n')
                prov[str(dst.relative_to(A))] = 'original+repair'
            else:
                shutil.copyfile(F / 'campaigns' / f"{job['id']}.json", dst)
                prov[str(dst.relative_to(A))] = 'original'
        (A / 'campaigns.json').write_text(json.dumps({**batch, 'out_dir': str(A / 'campaigns')}, indent=1))
        for sub, src in (('evaluations', F / 'evaluations'), ('phase6/sensitivity', F / 'phase6' / 'sensitivity')):
            (A / sub).mkdir(parents=True, exist_ok=True)
            for f in sorted(src.glob('*.json')):
                shutil.copyfile(f, A / sub / f.name)
                prov[f'{sub}/{f.name}'] = 'original'
        for f in sorted((R / 'evaluations').glob('*.json')):
            shutil.copyfile(f, A / 'evaluations' / f.name)
            prov[f'evaluations/{f.name}'] = 'repair'
        for f in sorted((R / 'sensitivity').glob('*.json')):
            shutil.copyfile(f, A / 'phase6' / 'sensitivity' / f.name)
            prov[f'phase6/sensitivity/{f.name}'] = 'repair'
        shutil.copyfile(F / 'phase6' / 'picks.json', A / 'phase6' / 'picks.json')
        shutil.copytree(F / 'phase6' / 'reference', A / 'phase6' / 'reference', dirs_exist_ok=True)
        ledger = (F / 'ledger.jsonl').read_text() + ((R / 'ledger.jsonl').read_text() if (R / 'ledger.jsonl').exists() else '')
        (A / 'ledger.jsonl').write_text(ledger)
        (A / 'provenance.json').write_text(json.dumps(prov, indent=1))
        print(json.dumps({'campaign_records': len(batch['jobs']), 'repaired_llm': len(repaired),
                          'repaired_files': sum(v != 'original' for v in prov.values())}))
        return
    if args.stage == 'e3-reuse':
        old, new = F / 'e3', A / 'e3'
        old_gp = json.loads((old / 'gp_reader.json').read_text())
        new_gp = json.loads((new / 'gp_reader.json').read_text())
        (new / 'llm').mkdir(exist_ok=True)
        (new / 'evaluations').mkdir(exist_ok=True)
        same, changed = [], []
        for p in sorted((new / 'packets').glob('p*.json')):
            q = old / 'packets' / p.name
            pid = p.stem
            if q.exists() and json.loads(q.read_text())['packet'] == json.loads(p.read_text())['packet']:
                same.append(pid)
                for f in (old / 'llm').glob(f'*_{pid}.json'):
                    shutil.copyfile(f, new / 'llm' / f.name)
                for f in (old / 'llm').glob(f'*_{pid}.audit.jsonl'):
                    shutil.copyfile(f, new / 'llm' / f.name)
                readers = {'gp-reader': old_gp[pid]['policy'] == new_gp[pid]['policy']}
                for f in (old / 'llm').glob(f'*_{pid}.json'):
                    readers[json.loads(f.read_text())['reader']] = True
                for f in (old / 'evaluations').glob(f'{pid}_*_y*.json'):
                    reader = f.name[len(pid) + 1:].rsplit('_y', 1)[0]
                    if readers.get(reader):
                        shutil.copyfile(f, new / 'evaluations' / f.name)
            else:
                changed.append(pid)
        (new / 'reuse.json').write_text(json.dumps({'packets_identical': same, 'packets_changed': changed}, indent=1))
        print(json.dumps({'identical': len(same), 'changed': len(changed)}))
        return
    entries = {str(p.relative_to(A)): sha(p) for p in sorted(A.rglob('*')) if p.is_file() and p.name != 'MANIFEST.json'}
    prov = json.loads((A / 'provenance.json').read_text()) if (A / 'provenance.json').exists() else {}
    (A / 'MANIFEST.json').write_text(json.dumps({'files': entries, 'provenance': prov}, indent=1))
    print(json.dumps({'files': len(entries)}))


if __name__ == '__main__':
    main()
