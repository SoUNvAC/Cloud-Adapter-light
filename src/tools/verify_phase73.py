"""Independent local checks of the diagnostic tables and transferred panels."""
import csv,hashlib,json
from pathlib import Path
from PIL import Image
import numpy as np

def main():
    root=Path('outputs/phase73/phase73_20261010')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    rows=list(csv.DictReader((root/'error_flows.csv').open()))
    old=read(Path('outputs/phase72/phase72_20261010/delivery/phase72_report.json'))
    refs={n:{r['product']:r for r in old['policies'][n]['rows']} for n in ['L3','B']}
    assert len(rows)==64 and len({r['group_id'] for r in rows})==64
    for r in rows:
        for n in ['L3','B']:
            cm=np.array(refs[n][r['product']]['confusion'])
            for name,a,b in [('Surface_to_Shadow',0,2),('Cloud_to_Shadow',1,2),('Shadow_to_Surface',2,0),('Shadow_to_Cloud',2,1)]:
                assert int(r[n+'_'+name])==cm[a,b]
                assert int(r[n+'_'+name+'_denominator'])==cm[a].sum()
        b=refs['B'][r['product']];a=refs['L3'][r['product']]
        assert int(r['new_FP'])-int(r['removed_FP'])==b['false_positive']-a['false_positive']
        assert int(r['rescued_TP'])-int(r['lost_TP'])==b['true_positive']-a['true_positive']
    points=list(csv.DictReader((root/'sampled_points.csv').open()))
    assert len({p['point_id'] for p in points})==len(points)
    panels=list((root/'panels').glob('*.png'));assert len(panels)==2*len(points)
    for p in points:
        assert 0<=int(p['row'])<1022 and 0<=int(p['column'])<1022
        for k in ['imagery_panel','label_panel']:
            with Image.open(root/p[k]) as im:assert im.size==(771,562);im.verify()
    verified=0
    for line in (root/'remote_sha256.txt').read_text().splitlines():
        sha,name=line.split('  ',1)
        p=root/name.removeprefix('./')
        assert hashlib.sha256(p.read_bytes()).hexdigest()==sha, str(p)
        verified+=1
    summary=read(root/'summary.json')
    for k,v in summary['totals'].items():assert sum(int(r[k]) for r in rows)==v
    result=dict(groups=64,points=len(points),panels=len(panels),transferred_files_sha_verified=verified,
                group_confusion_and_transition_identities_verified=True,all_panels_native_dimensions=True)
    (root/'local_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
