"""Acquire only locked author package sources, V1 ensemble and selected B8A."""
import base64
import hashlib
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request
import urllib.error
import time
from install_phase111_runtime import host_free


REPO=Path(__file__).resolve().parents[2]
BASE=Path('/home/scv/shared/phase111_ocm_runtime')


def save(name,obj):
    (BASE/name).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')


def main():
    assert (BASE/'RUNTIME_IMPORT_ACCEPTANCE.json').exists()
    lock=json.loads((REPO/'src/research_results/phase111_20261014/PRE_EXECUTION_LOCK.json').read_text())
    target=BASE/'assets'
    target.mkdir(exist_ok=False)
    with (BASE/'ASSET_LOCK').open('x') as f: f.write(str(os.getpid()))
    budget={'cap_bytes':4000000000,'reserved_bytes':0,'actual_received_bytes':0}
    records=[]
    def fetch(url,path,size,sha=None,md5=None,git_sha=None):
        assert budget['reserved_bytes']+size<=budget['cap_bytes']
        assert host_free()-size>=4256000000
        budget['reserved_bytes']+=size;save('ASSET_BUDGET.json',budget)
        path.parent.mkdir(parents=True,exist_ok=True)
        h=hashlib.sha256();m=hashlib.md5();data=bytearray() if git_sha else None
        n=0
        for attempt in range(3):
            if attempt:
                assert budget['reserved_bytes']+size<=budget['cap_bytes']
                budget['reserved_bytes']+=size;save('ASSET_BUDGET.json',budget)
            try:
                response=urllib.request.urlopen(url,timeout=60)
                break
            except urllib.error.HTTPError:
                raise
            except urllib.error.URLError as exc:
                with (BASE/'ASSET_CONNECTION_RETRIES.jsonl').open('a') as retry_log:
                    retry_log.write(json.dumps({'url':url,'attempt':attempt+1,'error':str(exc)})+'\n')
                if attempt==2:raise
                time.sleep(attempt+1)
        with response,path.open('xb') as out:
            while True:
                chunk=response.read(1024*1024)
                if not chunk: break
                n+=len(chunk);budget['actual_received_bytes']+=len(chunk)
                assert n<=size
                out.write(chunk);h.update(chunk);m.update(chunk)
                if data is not None: data.extend(chunk)
        assert n==size
        if sha: assert h.hexdigest()==sha
        if md5: assert base64.b64encode(m.digest()).decode()==md5
        if git_sha: assert hashlib.sha1(b'blob '+str(n).encode()+b'\0'+data).hexdigest()==git_sha
        records.append({'path':str(path),'url':url,'bytes':n,'sha256':h.hexdigest(),
                        'length_pass':True,'declared_sha_pass':sha is not None,
                        'md5_pass':md5 is not None,'git_blob_pass':git_sha is not None})
        save('ASSET_VERIFICATION.json',{'files':records,'status':'in_progress'})
        save('ASSET_BUDGET.json',budget)
        print(json.dumps(records[-1]),flush=True)
    treepath=REPO/'src/research_results/phase109_20261013/sources/omni_tree.json'
    # The full commit tree is supplied as a separate SHA-bound small input by the caller.
    if not treepath.exists(): treepath=BASE/'AUTHOR_COMMIT_TREE.json'
    tree=json.loads(treepath.read_text())
    code_commit=lock['code_commit']
    assert tree['sha']==code_commit and not tree['truncated']
    for row in tree['tree']:
        if row['type']=='blob' and row['path'].startswith('omnicloudmask/') and row['path'].endswith(('.py','.csv')):
            fetch(f'https://raw.githubusercontent.com/DPIRD-DMA/OmniCloudMask/{code_commit}/'+row['path'],
                  target/'vendor'/row['path'],row['size'],git_sha=row['sha'])
    for w in lock['weights']:
        fetch('https://huggingface.co/NickWright/OmniCloudMask/resolve/'+lock['hf_model_revision']+'/'+w['hf_filename'],
              target/'weights'/w['hf_filename'],w['declared_length_bytes'],sha=w['declared_sha256'])
    b=lock['selected_b8a']
    fetch(b['future_media_url'],target/'B8A.jp2',b['bytes'],md5=b['md5_base64'])
    save('ASSET_VERIFICATION.json',{'files':records,'status':'all_files_verified',
                                 'selected_product_id':lock['selected_product_id'],
                                 'code_commit':code_commit,'model_calls':0})
    (BASE/'ASSET_LOCK').unlink()


if __name__=='__main__': main()
