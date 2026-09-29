#!/usr/bin/env python3
"""Audit the zero-canopy singularity algebraically, without an ODE run."""
import argparse,hashlib,json,math
from pathlib import Path


def nodes(obj):
    result={}
    for key,value in obj.items():
        if isinstance(value,dict):
            if 'type' in value: result[key]=value
            result.update(nodes(value))
    return result


def audit(source, contract):
    definitions=source/'greenlight/models/katzin_2021/definition'
    merged={}
    for name,expected in contract['model']['files'].items():
        data=(definitions/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected:
            raise ValueError('source hash mismatch')
        if name!=contract['model']['entrypoint']:
            merged.update(nodes(json.loads(data)))
    assert merged['lai']['definition']=='sla * cLeaf'
    assert merged['capCan']['definition']=='capLeaf * lai'
    assert '1/capCan' in merged['tCan']['definition']
    sla=float(merged['sla']['definition']);cap_leaf=float(merged['capLeaf']['definition'])
    lai=sla*0.;capacity=cap_leaf*lai
    try:
        inverse=1./capacity
    except ZeroDivisionError:
        undefined=True
    else:
        undefined=not math.isfinite(inverse)
    return {'scope':'pinned-equation algebra audit, no solver run','status':'requires_explicit_crop_absent_mode',
            'empty_cLeaf':0.,'lai':lai,'capCan':capacity,'inverse_capacity_undefined':undefined,
            'temperature_rhs':merged['tCan']['definition'],
            'claim_limit':'Naive clearing is singular; this does not prove all crop-absent formulations fail.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    c=json.loads((root/'configs/v22/task_contract_v0.json').read_text())
    result=audit(args.source,c)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
