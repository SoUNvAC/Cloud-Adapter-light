"""Resume software-only installation with preserved prefix and bounded range reads.

Never touches models, imagery, original conda, or historical results. Caller
must first confirm the previous installer exited; its lock is archived.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
import ssl
from install_phase111_runtime import host_free
from phase111_resource_gate import run

BASE=Path('/home/scv/shared/phase111_ocm_runtime')


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gate-directory',type=Path,required=True)
    args=ap.parse_args();gate=args.gate_directory
    oldlock=BASE/'INSTALL_LOCK';oldpid=int(oldlock.read_text())
    try:os.kill(oldpid,0)
    except ProcessLookupError:pass
    else:
        status=Path(f'/proc/{oldpid}/stat').read_text().split()[2]
        assert status=='Z','Previous installer still alive; do not duplicate'
    oldlock.rename(BASE/f'INSTALL_LOCK_stopped_{oldpid}')
    with oldlock.open('x') as f:f.write(str(os.getpid()))
    runtime=BASE/'runtime_cuda128';temp=BASE/'runtime_install_tmp';python=runtime/'bin/python'
    res=json.loads((gate/'PIP_RESOLUTION.json').read_text());price=json.loads((gate/'complete_wheel_pricing/COMPLETE_ENVIRONMENT_PRICE.json').read_text())
    items={x['metadata']['name'].lower().replace('_','-'):x for x in res['install']}
    costs={x['name']:x for x in price['packages']}
    ledger=BASE/'INSTALL_LEDGER.jsonl';done={x['name']:x for x in map(json.loads,ledger.read_text().splitlines())}
    env=dict(os.environ,TMPDIR=str(temp),PYTHONDONTWRITEBYTECODE='1',PIP_NO_CACHE_DIR='1')
    for key in ['HF_HOME','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR']:env[key]=str(BASE/'controlled_caches'/key.lower())
    check_code='import importlib.metadata as m,json; print(json.dumps({d.metadata["Name"].lower().replace("_","-"):d.version for d in m.distributions()}))'
    actual=json.loads(subprocess.check_output([str(python),'-c',check_code],env=env,text=True))
    for name,row in done.items():assert actual[name]==row['version']==items[name]['metadata']['version']
    for name in price['installation_order']:
        if name in done:continue
        item=items[name];cost=costs[name];url=item['download_info']['url'];size=cost['wheel_declared_length']
        wheel=temp/urllib.parse.unquote(urllib.parse.urlsplit(url).path.rsplit('/',1)[-1])
        prefix=wheel.stat().st_size if wheel.exists() else 0
        assert prefix<=size and host_free()-2*(size-prefix)-cost['uncompressed_all_entries_bytes']>=4256000000
        def fetch_interval(spec):
            i,start,end=spec;path=temp/(name+f'.range_{i}')
            for attempt in range(3):
                have=path.stat().st_size if path.exists() else 0
                assert have<=end-start+1
                if have==end-start+1:return path
                offset=start+have
                req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-{end}'})
                try:
                    with urllib.request.urlopen(req,timeout=60) as r,path.open('ab') as f:
                        assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {offset}-{end}/')
                        count=have
                        while True:
                            b=r.read(1024**2)
                            if not b:break
                            count+=len(b);assert count<=end-start+1;f.write(b)
                    assert count==end-start+1
                    return path
                except (urllib.error.URLError,TimeoutError,ConnectionError,ssl.SSLError) as exc:
                    with (temp/(name+f'.range_{i}.network.jsonl')).open('a') as log:
                        log.write(json.dumps({'attempt':attempt+1,'offset':offset,'error':str(exc)})+'\n')
                    if attempt==2:raise
                    time.sleep(attempt+1)
        if prefix<size:
            manifest_path=temp/(name+'.range_manifest.json')
            if manifest_path.exists():
                saved=json.loads(manifest_path.read_text());assert saved['url']==url and saved['size']==size
                intervals=saved['intervals']
            else:
                remaining=size-prefix;workers=4 if remaining>=4*1024**2 else 1
                chunk=max(1,(remaining+workers-1)//workers)
                intervals=[(i,start,min(size-1,start+chunk-1)) for i,start in enumerate(range(prefix,size,chunk))]
                manifest_path.write_text(json.dumps({'url':url,'size':size,'initial_prefix':prefix,'intervals':intervals})+'\n')
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(fetch_interval,s) for s in intervals]
                while not all(f.done() for f in futures):
                    assert host_free()>=4256000000
                    time.sleep(10)
                parts=[f.result() for f in futures]
            for part,(_,start,end) in zip(parts,intervals):
                present=wheel.stat().st_size if wheel.exists() else 0
                assert present>=start
                overlap=min(present-start,end-start+1)
                if overlap:
                    with wheel.open('rb') as existing,part.open('rb') as segment:
                        existing.seek(start);remaining=overlap
                        while remaining:
                            n=min(4*1024**2,remaining);assert existing.read(n)==segment.read(n);remaining-=n
                with wheel.open('ab') as f:
                    with part.open('rb') as source:
                        source.seek(overlap)
                        for b in iter(lambda:source.read(4*1024**2),b''):f.write(b)
        assert wheel.stat().st_size==size
        expected=item['download_info']['archive_info']['hashes']['sha256']
        assert digest(wheel)==expected,'Integrity mismatch; never bypass or retry as success'
        for part in temp.glob(name+'.range_[0-9]'):part.unlink()
        command=[sys.executable,'-m','pip','--isolated','--disable-pip-version-check','--python',str(python),
                 'install','--no-deps','--no-cache-dir','--no-compile',str(wheel)]
        if name in actual:
            assert actual[name]==item['metadata']['version']
            command.append('--force-reinstall')
        result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=240)
        (temp/(name+'.install.log')).write_text(result.stdout+result.stderr)
        assert result.returncode==0,(name,result.stderr)
        wheel.unlink()
        row={'name':name,'version':item['metadata']['version'],'downloaded_bytes':size-prefix,'preserved_prefix_bytes':prefix,
             'sha256':expected,'installed_to':str(runtime),'parallel_range_connections':4}
        with ledger.open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    check=subprocess.run([sys.executable,'-m','pip','--isolated','--disable-pip-version-check','--python',str(python),'check'],env=env,capture_output=True,text=True)
    (BASE/'PIP_CHECK.txt').write_text(check.stdout+check.stderr);assert check.returncode==0
    code='import torch,torchvision,fastai,timm,segmentation_models_pytorch as smp,rasterio,json; print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),torchvision=torchvision.__version__,fastai=fastai.__version__,timm=timm.__version__,smp=smp.__version__,rasterio=rasterio.__version__)))'
    result=subprocess.run([str(python),'-c',code],env=env,capture_output=True,text=True,timeout=120)
    (BASE/'RUNTIME_IMPORT_LOG.txt').write_text(result.stdout+result.stderr);assert result.returncode==0
    accepted=json.loads(result.stdout.strip());assert accepted['torch']=='2.11.0+cu128' and accepted['cuda_available']
    accepted.update(dependency_check='passed',host_free_after_bytes=host_free(),model_calls=0)
    (BASE/'RUNTIME_IMPORT_ACCEPTANCE.json').write_text(json.dumps(accepted,indent=2)+'\n');oldlock.unlink()
    print(json.dumps(accepted),flush=True)


if __name__=='__main__':main()
