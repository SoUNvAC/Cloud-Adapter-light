"""Seal AI visual candidate locations before either model output exists."""
import csv,datetime
from phase200_common import *
def main():
    assert not (ROOT/'infer').exists() and not (ROOT/'POINTS_SEALED.json').exists()
    assert load(ROOT/'acquire/STATUS.json')['status']=='prepared'
    with (ROOT/'candidate_points.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)<=24 and len({x['point_id'] for x in rows})==len(rows)
    import numpy as np
    valid=np.load(ROOT/'valid20.npy',mmap_mode='r');a=load(ROOT/'INPUT_RECEIPT.json');t=a['transform'];w=a['display_window']
    counts={k:0 for k in ['shadow','shore','water','cloud']}
    for x in rows:
        r,c=int(x['row20']),int(x['col20']);assert valid[r,c]
        assert w['row']<=r<w['row']+w['height'] and w['col']<=c<w['col']+w['width']
        assert abs(float(x['utm_x'])-(t[2]+(c+.5)*20))<.01 and abs(float(x['utm_y'])-(t[5]-(r+.5)*20))<.01
        assert x['certainty'] in ['clear','mixed','uncertain'] and x['reviewer_type']=='AI_visual_RGB_only' and len(x['basis'])>10
        counts[x['candidate']]+=1;assert counts[x['candidate']]<=6
    write(ROOT/'POINTS_SEALED.json',dict(points_sha256=sha(ROOT/'candidate_points.csv'),input_receipt_sha256=sha(ROOT/'INPUT_RECEIPT.json'),true_color_sha256=sha(ROOT/'true_color.png'),counts=counts,missing={k:6-v for k,v in counts.items()},
        reviewer_type='AI_visual_RGB_only',independent_truth=False,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),model_calls_so_far=0))
    print(json.dumps(load(ROOT/'POINTS_SEALED.json')))
if __name__=='__main__':main()
