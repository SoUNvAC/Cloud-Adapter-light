"""Prelaunch identity and budget checks; no model calls or reference labels."""
import ast
from phase200_common import *
def main():
    for p in (REPO/'src/tools').glob('*phase200*.py'):ast.parse(p.read_text(encoding='utf8'))
    a=load(LOCK)
    assert a['product_id'].startswith('S2B_MSIL1C_20230908') and '_T11TNK_' in a['product_id']
    assert {x['band'] for x in a['objects']}=={'B02','B03','B04','B8A','MTD_TL','MTD_MSIL1C'}
    assert len(a['weights_V1'])==len(a['weights_V4'])==2
    assert sum(x['bytes'] for x in a['objects']+a['weights_V4'])==a['planned_download_bytes']<a['download_cap_bytes']
    assert a['planned_main_calls_max']==256 and a['planned_all_calls_cap']==500
    for p,h in a['source_sha256'].items():assert sha(REPO/p)==h,p
    assert a['settings']['bands']==['B04','B03','B8A'] and a['settings']['patch_size']==1000 and a['settings']['patch_overlap']==300
    print('PASS: exact case, complete 2+2 ensembles, six source objects, download/call limits, syntax and all source hashes; zero neural calls')
if __name__=='__main__':main()
