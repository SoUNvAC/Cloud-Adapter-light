"""Read-only public catalogue discovery for the single predeclared Phase200 case."""
import concurrent.futures as cf
import json,urllib.parse,urllib.request
from pathlib import Path
O=Path(__file__).resolve().parents[2]/'outputs/phase200/phase200_20261017'
def get(url):
    with urllib.request.urlopen(url,timeout=90) as r:return json.load(r)
def main():
    O.mkdir(parents=True,exist_ok=True)
    requests={
        'gcs_A':'https://storage.googleapis.com/storage/v1/b/gcp-public-data-sentinel-2/o?'+urllib.parse.urlencode({'prefix':'tiles/11/T/NK/S2A_MSIL1C_20230908','maxResults':1000}),
        'gcs_B':'https://storage.googleapis.com/storage/v1/b/gcp-public-data-sentinel-2/o?'+urllib.parse.urlencode({'prefix':'tiles/11/T/NK/S2B_MSIL1C_20230908','maxResults':1000}),
        'weights':'https://huggingface.co/api/models/NickWright/OmniCloudMask?blobs=true',
        'cdse':'https://catalogue.dataspace.copernicus.eu/odata/v1/Products?'+urllib.parse.urlencode({'$filter':"contains(Name,'MSIL1C_20230908') and contains(Name,'T11TNK')",'$top':100})}
    def one(k,u):
        try:
            d=get(u);(O/(k+'.json')).write_text(json.dumps(d,indent=2)+'\n',encoding='utf8')
            if k.startswith('gcs'):return k,{'items':len(d.get('items',[])),'products':sorted({x['name'].split('/')[4] for x in d.get('items',[])}),'nextPageToken':d.get('nextPageToken')}
            if k=='cdse':return k,[{'Name':x['Name'],'Id':x['Id']} for x in d.get('value',[])]
            return k,{'sha':d['sha'],'weights':[{k:v for k,v in x.items() if k in ('rfilename','size','lfs')} for x in d['siblings'] if '7.97' in x['rfilename']]}
        except Exception as e:return k,{'error':repr(e)}
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for k,v in pool.map(lambda kv:one(*kv),requests.items()):print(k,json.dumps(v),flush=True)
if __name__=='__main__':main()
