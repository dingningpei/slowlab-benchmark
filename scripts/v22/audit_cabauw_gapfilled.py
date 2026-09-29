"""Read-only source/quality audit of approved KNMI Cabauw lc1 cache against lb1."""
from __future__ import annotations
import calendar,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import h5py,numpy as np
from scipy.io import netcdf_file
ROOT=Path('/Users/dingningpei/Desktop/slowlab-v2-observation')
CACHE=Path('/private/tmp/slowlab-v22-cabauw-gapfilled')
RAW=Path('/private/tmp/slowlab-v22-cabauw-weather')
PLAN=json.loads((ROOT/'configs/v22/weather_gapfilled_plan.json').read_text())
MANIFEST=json.loads((CACHE/'manifest.json').read_text())
assert len(MANIFEST['files'])==98 and MANIFEST['total_bytes_downloaded']==PLAN['total_bytes']==18_413_090
MET='cesar_surface_meteo_lc1_t10';RAD='cesar_surface_radiation_lc1_t10'
MET_RAW='cesar_surface_meteo_lb1_t10';RAD_RAW='cesar_surface_radiation_lb1_t10'
CHANNELS={'meteo':('TA002','RH002','TD002','F010','SWD'),'radiation':('SWD','LWD')}
SOURCE={'TA002':'ITA002','RH002':'IQ002','TD002':'IQ002','F010':'IF010','SWD':'ISWD','LWD':'ILWD'}
REPORT={'status':'audit_in_progress','scope':'2016-12 boundary and 2017-2020 paired 10-minute lc1/lb1; no formal weather split or simulator use','total_bytes':MANIFEST['total_bytes_downloaded'],'months':{},'years':{}}

def load(path,keys):
    with path.open('rb') as stream:magic=stream.read(4)
    if magic==b'\x89HDF':
        with h5py.File(path) as f:
            data={k:np.asarray(f[k][:],dtype=np.float64) for k in keys if k in f}
            fills={k:float(np.asarray(f[k].attrs['_FillValue']).item()) for k in data if '_FillValue' in f[k].attrs}
            units={k:f[k].attrs.get('units',b'').decode(errors='replace') if isinstance(f[k].attrs.get('units',b''),bytes) else str(f[k].attrs.get('units','')) for k in data}
        return data,fills,units,'hdf5'
    if magic==b'CDF\x01':
        with netcdf_file(path,mmap=False) as f:
            data={k:np.asarray(f.variables[k][:],dtype=np.float64) for k in keys if k in f.variables}
            fills={k:float(f.variables[k]._FillValue) for k in data if hasattr(f.variables[k],'_FillValue')}
            units={k:(getattr(f.variables[k],'units',b'').decode(errors='replace') if isinstance(getattr(f.variables[k],'units',b''),bytes) else str(getattr(f.variables[k],'units',''))) for k in data}
        return data,fills,units,'netcdf3'
    raise ValueError(f'unrecognized NetCDF: {path.name}')

def longest(bad):
    if not bad.any():return 0
    d=np.diff(np.concatenate(([0],bad.astype(np.int8),[0])))
    return int(np.max(np.where(d==-1)[0]-np.where(d==1)[0]))

def counts(a,fill):
    return {str(int(v)):int(n) for v,n in zip(*np.unique(a[(np.isfinite(a))&(a!=fill)].astype(int),return_counts=True))}

by_key={(f['dataset'],f['filename'][-9:-3]):f for f in PLAN['files']}
for year in range(2017,2021):
    year_bad=[];tot=Counter();src_count=defaultdict(Counter);months=[]
    for month in range(1,13):
        ym=f'{year}{month:02d}';n=calendar.monthrange(year,month)[1]*144
        data={};fills={};units={};formats={};sha={};zero_dates={};paired={}
        for role,dataset,raw_dataset in (('meteo',MET,MET_RAW),('radiation',RAD,RAD_RAW)):
            f=by_key[(dataset,ym)];path=CACHE/dataset/f['filename']
            assert path.stat().st_size==f['size']
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest==MANIFEST['files'][f['filename']]['sha256']
            keys={'time','time_bnds','valid_dates',*CHANNELS[role],*(SOURCE[k] for k in CHANNELS[role])}
            a,fill,u,fmt=load(path,keys);data[role]=a;fills[role]=fill;units[role]=u;formats[role]=fmt;sha[role]=digest
            assert len(a['time'])==n and a['time_bnds'].shape==(n,2)
            b=a['time_bnds']
            assert abs(b[0,0])<1e-6 and abs(b[-1,1]-n/6)<1e-5
            assert np.max(abs((b[:,1]-b[:,0])*3600-600))<1
            assert np.max(abs((b[1:,0]-b[:-1,1])*3600))<1
            assert len(a['valid_dates'])==calendar.monthrange(year,month)[1]
            zero_dates[role]=int((a['valid_dates']==0).sum())
            raw_path=RAW/raw_dataset/f['filename'].replace(dataset,raw_dataset)
            r,rf,ru,rfmt=load(raw_path,CHANNELS[role])
            paired[role]=(r,rf)
        assert np.max(abs(data['meteo']['time_bnds']-data['radiation']['time_bnds']))*3600<1
        invalid={};source={};comparison={};bad=np.zeros(n,dtype=bool)
        for role in CHANNELS:
            for channel in CHANNELS[role]:
                a=data[role][channel];fill=fills[role].get(channel,np.nan)
                miss=(~np.isfinite(a))|(a==fill);bad|=miss
                key=role+'/'+channel;invalid[key]=int(miss.sum())
                tot[key+'_missing']+=invalid[key]
                idx=SOURCE[channel]
                source_arr=data[role].get(idx)
                if source_arr is not None:
                    source[key]=counts(source_arr,fills[role].get(idx,np.nan))
                    for code,num in source[key].items():src_count[key][code]+=num
                else:source[key]=None
                raw,raw_fills=paired[role];rv=raw[channel];raw_miss=(~np.isfinite(rv))|(rv==raw_fills.get(channel,np.nan))
                common=(~raw_miss)&(~miss)
                comparison[key]={'lb1_missing':int(raw_miss.sum()),'lc1_valid_where_lb1_missing':int((raw_miss&(~miss)).sum()),'both_valid':int(common.sum()),'different_gt_0_01_on_both_valid':int((abs(a[common]-rv[common])>0.01).sum()),'max_abs_difference_on_both_valid':float(np.max(abs(a[common]-rv[common]))) if common.any() else None}
                tot[key+'_lb1_missing']+=comparison[key]['lb1_missing'];tot[key+'_lc1_valid_where_lb1_missing']+=comparison[key]['lc1_valid_where_lb1_missing']
        ta=data['meteo']['TA002'];td=data['meteo']['TD002'];rh=data['meteo']['RH002'];swd=data['radiation']['SWD'];lwd=data['radiation']['LWD']
        physical={'ta_min':float(ta.min()),'ta_max':float(ta.max()),'td_min':float(td.min()),'td_max':float(td.max()),'rh_above_100':int((rh>100).sum()),'rh_below_0':int((rh<0).sum()),'swd_negative':int((swd<0).sum()),'swd_min':float(swd.min()),'lwd_negative':int((lwd<0).sum()),'lwd_min':float(lwd.min())}
        entry={'rows':n,'joint_invalid':int(bad.sum()),'longest_joint_invalid_intervals':longest(bad),'channel_invalid':invalid,'source_index_counts':source,'lb1_comparison':comparison,'physical':physical,'units':{role:{k:units[role].get(k) for k in CHANNELS[role]} for role in CHANNELS},'formats':formats,'sha256':sha,'valid_dates_zero':zero_dates}
        REPORT['months'][ym]=entry;year_bad.append(bad);months.append(ym)
        tot['rows']+=n;tot['joint_invalid']+=entry['joint_invalid']
        for k in ('rh_above_100','rh_below_0','swd_negative','lwd_negative'):tot[k]+=physical[k]
    annual_bad=np.concatenate(year_bad)
    REPORT['years'][str(year)]={'rows':tot['rows'],'joint_invalid':tot['joint_invalid'],'joint_invalid_fraction':tot['joint_invalid']/tot['rows'],'longest_joint_invalid_intervals':longest(annual_bad),'channels':{key:{'lc1_missing':tot[key+'_missing'],'lb1_missing':tot[key+'_lb1_missing'],'lc1_valid_where_lb1_missing':tot[key+'_lc1_valid_where_lb1_missing'],'source_index_counts':dict(src_count[key])} for key in ('meteo/TA002','meteo/RH002','meteo/TD002','meteo/F010','meteo/SWD','radiation/SWD','radiation/LWD')},'physical':{k:tot[k] for k in ('rh_above_100','rh_below_0','swd_negative','lwd_negative')}}
    print(json.dumps({'year':year,**REPORT['years'][str(year)]}),flush=True)
# Audit boundary pair as files and time grid, without taking it into a yearly evaluation denominator.
boundary={}
for role,dataset in (('meteo',MET),('radiation',RAD)):
    f=by_key[(dataset,'201612')];path=CACHE/dataset/f['filename'];raw=path.read_bytes()
    assert len(raw)==f['size'] and hashlib.sha256(raw).hexdigest()==MANIFEST['files'][f['filename']]['sha256']
    a,fill,units,fmt=load(path,('time','time_bnds','valid_dates',*CHANNELS[role]))
    assert len(a['time'])==31*144 and abs(a['time_bnds'][-1,1]-744)<1e-5
    boundary[role]={'sha256':hashlib.sha256(raw).hexdigest(),'format':fmt,'missing':{k:int(((~np.isfinite(a[k]))|(a[k]==fill.get(k,np.nan))).sum()) for k in CHANNELS[role]}}
REPORT['boundary_201612']=boundary
REPORT['status']='audited'
path=ROOT/'results/v22/weather_gapfilled_quality_audit.json';path.write_text(json.dumps(REPORT,indent=2)+'\n')
print(json.dumps({'status':'audited','years':list(REPORT['years']),'total_bytes':REPORT['total_bytes'],'boundary':boundary}))
