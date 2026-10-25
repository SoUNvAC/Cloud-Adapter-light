"""At most three small primary-source documents; no datasets, weights, or repository clone."""
import base64
from datetime import datetime,timezone
import urllib.request
from phase309_common import *
def fetch():
    out=ROOT/'evidence/web';out.mkdir();receipts=[]
    used=load(ROOT/'ENVIRONMENT_LOCK.json')['download_wheel_bytes']
    urls=[('zenodo_record4172871','https://zenodo.org/api/records/4172871'),('author_IRIS_repository','https://api.github.com/repos/ESA-PhiLab/iris'),('author_IRIS_README','https://api.github.com/repos/ESA-PhiLab/iris/readme')]
    for name,url in urls:
        item=dict(name=name,url=url,accessed_UTC=datetime.now(timezone.utc).isoformat(),source_role='exact dataset version metadata' if name.startswith('zenodo') else 'author-linked tool repository, current state not dataset-version provenance')
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Phase309 bounded provenance audit','Accept':'application/json'})
            with urllib.request.urlopen(req,timeout=20) as response:
                b=response.read(min(2000000,20000000-used)+1);assert len(b)<=2000000 and used+len(b)<=20000000
                assert response.url.startswith(('https://zenodo.org/','https://api.github.com/'))
                item.update(status=response.status,final_url=response.url,bytes=len(b));used+=len(b)
            p=out/(name+'.json');p.write_bytes(b);item.update(path=p.relative_to(ROOT).as_posix(),sha256=sha(p))
            if name=='author_IRIS_README':
                data=__import__('json').loads(b);text=base64.b64decode(data['content']).decode('utf-8');q=out/'author_IRIS_README.md';q.write_text(text,encoding='utf-8');item['decoded_path']=q.relative_to(ROOT).as_posix();item['revision_blob_sha']=data['sha']
            if name=='zenodo_record4172871':assert __import__('json').loads(b)['id']==4172871
        except Exception as e:item.update(failure=repr(e),retained=True,no_retry=True)
        receipts.append(item)
    write(ROOT/'NETWORK_EVIDENCE.json',dict(documents=receipts,documents_attempted=len(urls),wheel_bytes=load(ROOT/'ENVIRONMENT_LOCK.json')['download_wheel_bytes'],document_bytes=used-load(ROOT/'ENVIRONMENT_LOCK.json')['download_wheel_bytes'],total_download_bytes=used,limit_bytes=20000000,limit_documents=10,no_whole_repo_or_images=True))
    return receipts
