"""Acquire exact-original PixBox object indexes and XML metadata only, never bands."""
import base64,hashlib,json,re,urllib.parse
import acquire_phase102_metadata as a

def main():
    names=re.findall(r';(S2[AB]_MSIL1C_\S+)\.SAFE',
                     (a.OUT/'pixbox_description.txt').read_text(encoding='utf-8'))
    # Reconcile a saved bounded probe from a concurrent process, without inventing headers.
    probe=a.OUT/'gcs_pixbox_probe.json'
    if probe.exists() and not any(e['file']==probe.name for e in a.events):
        a.events.append({'file':probe.name,'body_bytes_received':probe.stat().st_size,
                         'sha256':hashlib.sha256(probe.read_bytes()).hexdigest(),
                         'accounting':'recovered saved response size; request event overwritten by concurrent ledger writer',
                         'url':'GCS exact original product prefix; see object names in response'})
    for i,n in enumerate(names):
        tile=re.search(r'_T(\d{2}[A-Z]{3})_',n)[1]
        prefix=f'tiles/{tile[:2]}/{tile[2]}/{tile[3:]}/{n}.SAFE/'
        url='https://storage.googleapis.com/storage/v1/b/gcp-public-data-sentinel-2/o?'+urllib.parse.urlencode(
            {'prefix':prefix,'fields':'items(name,size,md5Hash,generation),nextPageToken','maxResults':1000})
        raw=a.fetch(url,f'gcs_pixbox_{i:02}.json',80000)
        if not raw:continue
        d=json.loads(raw)
        if d.get('nextPageToken'):raise RuntimeError('Incomplete object listing; no automatic expansion')
        for role,pattern in [('product','/MTD_MSIL1C.xml'),('tile','/MTD_TL.xml')]:
            candidates=[x for x in d.get('items',[]) if x['name'].endswith(pattern)]
            if len(candidates)!=1:continue
            f=candidates[0]
            # Pin generation, bound XML; avoid large angles/dat astrip metadata.
            url='https://storage.googleapis.com/gcp-public-data-sentinel-2/'+urllib.parse.quote(f['name'],safe='/')+'?generation='+f['generation']
            if role=='tile':b=a.fetch(url,f'pixbox_{i:02}_tile_prefix.xml',16384,'bytes=0-16383')
            elif int(f['size'])<=120000:
                b=a.fetch(url,f'pixbox_{i:02}_product.xml',120000)
                if b:
                    assert len(b)==int(f['size'])
                    assert base64.b64encode(hashlib.md5(b).digest()).decode()==f['md5Hash']
            else:continue
        print('original product metadata',i,flush=True)
    print('received_body_bytes',sum(e['body_bytes_received'] for e in a.events),flush=True)

if __name__=='__main__':main()
