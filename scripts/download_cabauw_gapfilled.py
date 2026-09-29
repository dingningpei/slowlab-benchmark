"""Bounded KNMI Cabauw acquisition. File bytes only; resume verified cache."""
from __future__ import annotations
import argparse,hashlib,json,os,re,time,urllib.error,urllib.parse,urllib.request
from pathlib import Path

ROOT=Path('/Users/dingningpei/Desktop/slowlab-v2-observation')
CACHE=Path('/private/tmp/slowlab-v22-cabauw-gapfilled')
PLAN=ROOT/'configs/weather_gapfilled_plan.json'
CAP=18_413_090

def request(url,headers=None,limit=1_000_000):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers or {}),timeout=30) as response:
                body=response.read(limit+1)
                if len(body)>limit:raise RuntimeError('metadata response exceeds cap')
                return body
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt==5:raise
            retry=error.headers.get('Retry-After')
            delay=min(90,max(20,float(retry))) if retry and retry.isdigit() else min(90,20*(attempt+1))
            print(json.dumps({'status':'rate_limited','retry_seconds':delay,'attempt':attempt+1}),flush=True)
            time.sleep(delay)

def public_key():
    page=request('https://developer.dataplatform.knmi.nl/open-data-api').decode('utf-8')
    matches=re.findall(r'eyJ[A-Za-z0-9_=-]{60,}',page)
    if not matches:raise RuntimeError('public KNMI anonymous access token unavailable')
    return matches[0]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['sample','remaining'],required=True);args=parser.parse_args()
    plan=json.loads(PLAN.read_text());allowed={(f['dataset'],f['filename']):f for f in plan['files']}
    sample={(f['dataset'],f['filename']) for f in plan['sample_files']}
    assert len(allowed)==98 and sum(f['size'] for f in allowed.values())==18_413_090
    assert all('/' not in fn and '..' not in fn for _,fn in allowed)
    expected=set(sample if args.stage=='sample' else allowed.keys()-sample)
    CACHE.mkdir(parents=True,exist_ok=True)
    manifest_path=CACHE/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'scope':'KNMI Cabauw v2.2 lc1 gapfilled candidate weather files, source cache only','files':{},'total_bytes_downloaded':0}
    # The initial 2017-01 sample and audit are mandatory gates for remaining stage.
    if args.stage=='remaining':
        if not all(allowed[key]['filename'] in manifest['files'] for key in sample):
            raise RuntimeError('sample download gate not complete')
        sample_audit=ROOT/'results/weather_gapfilled_sample_audit.json'
        if not sample_audit.exists() or json.loads(sample_audit.read_text()).get('gate')!='pass':
            raise RuntimeError('sample audit gate not complete')
    key=public_key();new_bytes=0
    for dataset,filename in sorted(expected):
        f=allowed[(dataset,filename)];path=CACHE/dataset/filename
        if filename in manifest['files']:
            if not path.is_file() or path.stat().st_size!=f['size'] or hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['files'][filename]['sha256']:
                raise RuntimeError('existing cache does not match manifest: '+filename)
            continue
        if manifest['total_bytes_downloaded']+f['size']>CAP:raise RuntimeError('approved 18,413,090-byte cap exceeded')
        endpoint=f'https://api.dataplatform.knmi.nl/open-data/v1/datasets/{dataset}/versions/v1.0/files/{urllib.parse.quote(filename)}/url'
        info=json.loads(request(endpoint,{'Authorization':key}))
        url=info.get('temporaryDownloadUrl')
        if not isinstance(url,str) or not url.startswith('https://'):raise RuntimeError('invalid KNMI temporary URL')
        path.parent.mkdir(parents=True,exist_ok=True);partial=path.with_suffix('.part')
        digest=hashlib.sha256();received=0
        try:
            with urllib.request.urlopen(url,timeout=45) as response, partial.open('wb') as out:
                declared=response.headers.get('Content-Length')
                if declared is not None and int(declared)!=f['size']:raise RuntimeError('remote size changed: '+filename)
                while True:
                    chunk=response.read(min(65536,f['size']-received+1))
                    if not chunk:break
                    received+=len(chunk)
                    if received>f['size'] or manifest['total_bytes_downloaded']+received>CAP:raise RuntimeError('file/total byte cap exceeded')
                    out.write(chunk);digest.update(chunk)
            if received!=f['size']:raise RuntimeError('truncated file: '+filename)
            partial.replace(path)
        finally:
            if partial.exists():partial.unlink()
        new_bytes+=received
        manifest['files'][filename]={'dataset':dataset,'bytes':received,'sha256':digest.hexdigest(),'source':'KNMI Open Data API'}
        manifest['total_bytes_downloaded']+=received
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        time.sleep(2.1)
        print(json.dumps({'stage':args.stage,'file':filename,'bytes':received,'completed_files':len(manifest['files']),'total_bytes_downloaded':manifest['total_bytes_downloaded']}),flush=True)
    print(json.dumps({'status':'stage_complete','stage':args.stage,'files':len(manifest['files']),'total_bytes_downloaded':manifest['total_bytes_downloaded']}))

if __name__=='__main__':main()
