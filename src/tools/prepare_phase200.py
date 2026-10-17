"""Freeze public source identities and resource plan before downloading/inference."""
import json,urllib.parse,hashlib
from pathlib import Path
from phase200_common import *
def main():
    out=LOCK.parent;out.mkdir(parents=True,exist_ok=True);assert not LOCK.exists()
    (out/'.gitattributes').write_text('* -text whitespace=cr-at-eol\n')
    g=load(LOCAL/'gcs_B.json');assert not g.get('nextPageToken')
    pid='S2B_MSIL1C_20230908T183919_N0509_R070_T11TNK_20230908T221618'
    items=g['items'];objects=[]
    for band in ['B02','B03','B04','B8A','MTD_MSIL1C','MTD_TL']:
        suffix=('_'+band+'.jp2') if band.startswith('B') else ('/'+band+'.xml')
        found=[x for x in items if x['name'].endswith(suffix) and pid+'.SAFE/' in x['name'] and (not band.startswith('B') or '/IMG_DATA/' in x['name'])];assert len(found)==1,(band,len(found))
        x=found[0];objects.append(dict(band=band,name=x['name'],bytes=int(x['size']),generation=x['generation'],md5_base64=x['md5Hash'],url='https://storage.googleapis.com/gcp-public-data-sentinel-2/'+urllib.parse.quote(x['name'],safe='/')+'?generation='+x['generation']))
    old=load(REPO/'src/research_results/phase111_20261014/PRE_EXECUTION_LOCK.json')
    hf=load(LOCAL/'weights.json');assert hf['sha']==old['hf_model_revision']
    weights=[]
    for x in hf['siblings']:
        if '7.97' in x['rfilename']:
            weights.append(dict(filename=x['rfilename'],bytes=x['size'],sha256=x['lfs']['sha256'],url='https://huggingface.co/NickWright/OmniCloudMask/resolve/'+hf['sha']+'/'+x['rfilename']))
    assert len(weights)==2
    budget=sum(x['bytes'] for x in objects+weights);assert budget<3_000_000_000
    paper=next(Path('C:/Users/Administrator/Desktop/paper').glob('Limnol Oceanogr Letters*'))
    value=dict(phase=200,product_id=pid,objects=objects,weights_V4=weights,weights_V1=old['weights'],code_commit=old['code_commit'],hf_revision=hf['sha'],
        planned_download_bytes=budget,download_cap_bytes=3_000_000_000,planned_main_calls_max=256,planned_all_calls_cap=500,recovery_all_calls_cap=625,
        wall_seconds_cap=7200,recovery_wall_seconds_cap=9000,cpu_cap_bytes=16*1024**3,gpu_cap_bytes=20*1024**3,host_min_free_bytes=4_000_000_000,
        settings=dict(bands=['B04','B03','B8A'],resolution_m=20,patch_size=1000,patch_overlap=300,batch_size=1,dtype='float32',compile=False,decision='official raw-class argmax; confidence not used to threshold'),
        display_bbox_utm11N=[567500,4926000,577500,4941500],display_bbox_basis='Ducar Fig3 axes rounded outward; fixed before model output',
        paper=dict(path=str(paper),sha256=sha(paper),pages=[8,9],historical_input_weights_settings='unknown; not exact reproduction'),
        readme='Only this case, no training/fitting/threshold search. AI candidate points are not independent truth. Same 20m native DN input for both versions, author loader no radiometric offset correction; XML radiometry recorded separately. Full tile inference with lake-only evidence view.',
        source_sha256={str(p.relative_to(REPO)).replace('\\','/'):sha(p) for p in sorted((REPO/'src/tools').glob('*phase200*.py'))})
    write(LOCK,value);print(json.dumps({'download_GB':budget/1e9,'product':pid,'planned_main_calls_max':256,'plan_sha256':sha(LOCK)}))
if __name__=='__main__':main()
