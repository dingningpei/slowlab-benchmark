"""Read-only audit of approved KNMI Cabauw NetCDF cache (not a weather generator)."""
from __future__ import annotations
import calendar,hashlib,json,re
from pathlib import Path
import h5py
import numpy as np
from scipy.io import netcdf_file
ROOT=Path('/Users/dingningpei/Desktop/slowlab-v2-observation');CACHE=Path('/private/tmp/slowlab-v22-cabauw-weather')
plan=json.loads((ROOT/'configs/v22/weather_download_proposal.json').read_text())
manifest=json.loads((CACHE/'manifest.json').read_text());assert len(manifest['files'])==98
assert manifest['total_bytes_downloaded']==sum(f['size'] for f in plan['files'])<=22_000_000
by_month={};report={'status':'audit_in_progress','scope':'all approved files, read-only; no weather generator or formal split','total_bytes':manifest['total_bytes_downloaded'],'months':{},'years':{}}

def load(path,keys):
    magic=path.read_bytes()[:4]
    if magic==b'\x89HDF':
        with h5py.File(path) as f:
            return {k:np.asarray(f[k][:],dtype=np.float64) for k in keys}, {k:float(np.asarray(f[k].attrs['_FillValue']).item()) for k in keys if k not in ('time','time_bnds','valid_dates')},'hdf5'
    if magic==b'CDF\x01':
        with netcdf_file(path,mmap=False) as f:
            return {k:np.asarray(f.variables[k][:],dtype=np.float64) for k in keys}, {k:float(f.variables[k]._FillValue) for k in keys if k not in ('time','time_bnds','valid_dates')},'netcdf3'
    raise RuntimeError('unrecognized NetCDF container')

def longest(bad):
    if not bad.any():return 0
    d=np.diff(np.concatenate(([0],bad.astype(np.int8),[0])))
    return int((np.where(d==-1)[0]-np.where(d==1)[0]).max())
for year in range(2017,2021):
    annual={k:0 for k in ('rows','joint_invalid','rh_gt100','rh_lt0','swd_negative','lwd_negative')}
    channel_invalid={k:0 for k in ('TA002','RH002','TD002','F010','SWD','LWD')}
    max_gap=0
    for month in range(1,13):
        ym=f'{year}{month:02d}';arrays={};fills={};formats={};hashes={}
        for role,dataset,vars in [('meteo','cesar_surface_meteo_lb1_t10',('TA002','RH002','TD002','F010','SWD')),('radiation','cesar_surface_radiation_lb1_t10',('SWD','LWD'))]:
            f=next((f for f in plan['files'] if f['dataset']==dataset and f['filename'].endswith(f'_{ym}.nc')),None)
            if f is None:raise RuntimeError('missing planned month '+ym)
            path=CACHE/dataset/f['filename'];assert path.stat().st_size==f['size'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest==manifest['files'][f['filename']]['sha256']
            data,fill,formats[role]=load(path,('time','time_bnds','valid_dates',*vars));arrays[role]=data;fills[role]=fill;hashes[role]=digest
        met,rad=arrays['meteo'],arrays['radiation'];n=calendar.monthrange(year,month)[1]*144
        assert all(len(a['time'])==n for a in arrays.values()),('monthly row count',ym)
        for role,a in arrays.items():
            b=a['time_bnds'];assert b.shape==(n,2)
            assert abs(b[0,0])<1e-6 and abs(b[-1,1]-n/6)<1e-5,('month bounds',ym,role)
            assert np.all(np.diff(b[:,0])>0) and np.max(abs((b[:,1]-b[:,0])*3600-600))<1
            assert np.max(abs(b[1:,0]-b[:-1,1]))*3600<1
            assert len(a['valid_dates'])==calendar.monthrange(year,month)[1]
        assert np.max(abs(met['time_bnds']-rad['time_bnds']))*3600<1,('paired time misalignment',ym)
        bad=np.zeros(n,dtype=bool);invalid={}
        for key in channel_invalid:
            role='radiation' if key in ('SWD','LWD') else 'meteo'
            x=arrays[role][key];mask=(~np.isfinite(x))|(x==fills[role][key]);bad|=mask
            invalid[key]=int(mask.sum());channel_invalid[key]+=invalid[key]
        rh=met['RH002'];swd=rad['SWD'];lwd=rad['LWD'];entry={'rows':n,'invalid':invalid,'joint_invalid':int(bad.sum()),'longest_joint_invalid_intervals':longest(bad),'rh_above_100':int(((rh>100)&(~bad)).sum()),'rh_below_0':int(((rh<0)&(~bad)).sum()),'swd_negative':int(((swd<0)&(~bad)).sum()),'lwd_negative':int(((lwd<0)&(~bad)).sum()),'source_sha256':hashes,'formats':formats,'valid_dates_zero':{role:int((a['valid_dates']==0).sum()) for role,a in arrays.items()},'meteo_radiation_swd_max_abs_difference':float(np.nanmax(abs(met['SWD']-rad['SWD'])))}
        report['months'][ym]=entry
        annual['rows']+=n;annual['joint_invalid']+=entry['joint_invalid'];annual['rh_gt100']+=entry['rh_above_100'];annual['rh_lt0']+=entry['rh_below_0'];annual['swd_negative']+=entry['swd_negative'];annual['lwd_negative']+=entry['lwd_negative'];max_gap=max(max_gap,entry['longest_joint_invalid_intervals'])
    annual['individual_invalid']=channel_invalid;annual['longest_month_internal_joint_invalid_intervals']=max_gap
    annual['joint_invalid_fraction']=annual['joint_invalid']/annual['rows'];report['years'][str(year)]=annual
    print(json.dumps({'year':year,**annual}),flush=True)
report['status']='audited';out=ROOT/'results/v22/weather_quality_audit.json';out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'audited','year_count':len(report['years']),'total_bytes':report['total_bytes']}))
