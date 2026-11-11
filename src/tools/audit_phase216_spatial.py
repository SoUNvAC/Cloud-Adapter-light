"""Existing projected rectangle metadata audit, not pixel access."""
from pathlib import Path
import csv,json,re
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase216/phase216_20261023'
def main():
    req=json.loads((OUT/'SOURCE_POOL_REQUEST.json').read_text(encoding='utf-8'))
    pools={'val':ROOT/'outputs/phase65/source_metadata/acquired/val_metadata.csv','test':ROOT/'outputs/phase211/phase211_20261020/sources/test_metadata.csv'}
    def bounds(r):
        nums=[float(v) for v in re.findall(r'-?\d+(?:\.\d+)?',r['proj_geometry'])];xs=nums[0::2];ys=nums[1::2];return min(xs),min(ys),max(xs),max(ys)
    report={}
    for pool,path in pools.items():
        rows=list(csv.DictReader(path.open(encoding='utf-8-sig')));hits=[];sameEPSG=0;crossEPSG=0
        for r in req['selected']:
            a=bounds(r)
            for v in rows:
                if r['proj_epsg']!=v['proj_epsg']:crossEPSG+=1;continue
                sameEPSG+=1;b=bounds(v);area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
                if area>0:hits.append({'source_id':r['index'],'other_id':v['index'],'source_ROI':r['roi_id'],'other_ROI':v['roi_id'],'overlap_projected_units_squared':area,'same_product':r['s2_id']==v['s2_id']})
        report[pool]={'same_EPSG_rectangle_pairs_checked':sameEPSG,'positive_area_overlaps':hits,'cross_EPSG_pairs_not_transformed':crossEPSG,'limitation':'Exact same-CRS projected rectangle overlap only; cross-CRS relations unverified, no global geographic independence claim.'}
    (OUT/'SOURCE_SPATIAL_METADATA_AUDIT.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
