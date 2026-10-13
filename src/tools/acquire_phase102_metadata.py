"""Bounded public metadata acquisition. Never download imagery/mask archives."""
import hashlib,json,time,urllib.request,urllib.error,threading
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/phase102/phase102_20261013/acquired'
OUT.mkdir(parents=True,exist_ok=True)
LEDGER=OUT/'access_ledger.json'
events=json.loads(LEDGER.read_text(encoding='utf-8')) if LEDGER.exists() else []
LIMIT=19_000_000  # Reserve >=1 MB for two textual browser-page consultations.
ledger_lock=threading.Lock()

def fetch(url,name,cap,range_header=None):
    path=OUT/name
    if path.exists():return path.read_bytes()
    consumed=sum(e['body_bytes_received'] for e in events)
    cap=min(cap,LIMIT-consumed)
    if cap<=0:raise RuntimeError('Cumulative metadata transfer budget exhausted')
    headers={'User-Agent':'Phase102-metadata-audit/1.0','Accept-Encoding':'identity'}
    if range_header:headers['Range']=range_header
    e={'url':url,'file':name,'range':range_header,'body_bytes_received':0,'access':'metadata only'}
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=45) as r:
            e.update(status=r.status,content_length=r.headers.get('Content-Length'),content_range=r.headers.get('Content-Range'))
            if range_header and r.status!=206:raise RuntimeError('Range not supported; refusing whole archive body')
            chunks=[]
            while True:
                block=r.read(min(65536,cap-e['body_bytes_received']+1))
                if not block:break
                e['body_bytes_received']+=len(block)
                if e['body_bytes_received']>cap:raise RuntimeError('Per-request cap reached')
                chunks.append(block)
            data=b''.join(chunks)
            e['sha256']=hashlib.sha256(data).hexdigest();path.write_bytes(data)
            return data
    except Exception as x:
        e['error']=str(x)
        if isinstance(x,urllib.error.HTTPError):e['status']=x.code
        return None
    finally:
        with ledger_lock:
            events.append(e);LEDGER.write_text(json.dumps(events,indent=2)+'\n',encoding='utf-8',newline='\n')

if __name__=='__main__':
    for record in [7100327,5036991,5095024,11121064]:
        data=fetch(f'https://zenodo.org/api/records/{record}',f'record_{record}.json',500000)
        if data:
            d=json.loads(data)
            print(record,'license',d.get('metadata',{}).get('license'),'files',[(f['key'],f['size'],f['checksum']) for f in d.get('files',[])])
    print('received',sum(e['body_bytes_received'] for e in events))
