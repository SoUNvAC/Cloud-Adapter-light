"""Domain-wide monotone RGB quantile mapping, no labels or per-image fit."""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path('/home/scv/shared/phase443_20261103')

def fit():
    import sys,os,importlib.util
    sys.path.insert(0,'/home/scv/shared/phase431_20261102/code')
    from mmengine.dataset import Compose
    from mmseg.utils import register_all_modules
    register_all_modules()
    src=Path('/home/scv/Cloud-Adapter-light/src');os.chdir(src)
    spec=importlib.util.spec_from_file_location('catalogue443',src/'cloud_adapter/datasets/phase65_catalogue.py');spec.loader.exec_module(importlib.util.module_from_spec(spec))
    rows=json.loads((R/'INPUTS.json').read_text());pixels={'A':[],'B':[]};counts={}
    for row in rows:
        d=row['domain'];pipe=Compose([dict(type='LoadImageFromFile' if d=='A' else 'LoadPhase65RGB'),dict(type='Resize',scale=(512,512))])
        x=pipe(dict(img_path=row['image']))['img'][...,::-1].astype(np.float32)
        assert x.shape==(512,512,3) and np.isfinite(x).all()
        pixels[d].append(x[2::4,2::4].reshape(-1,3));counts[row['group']]=counts.get(row['group'],0)+1
    # Equal image weights are equal scene weights only because all ROIs have5 images and all target groups1.
    for d in pixels:assert len(set(n for g,n in counts.items() if g.startswith(d+':')))==1
    q=np.linspace(0,1,4097);quantiles={d:np.quantile(np.concatenate(v),q,axis=0) for d,v in pixels.items()}
    target=(quantiles['A']+quantiles['B'])/2
    arrays=dict(q=q,target=target)
    for d in pixels:
        arrays[d+'_quantiles']=quantiles[d]
        for c in range(3):
            knots,inv=np.unique(quantiles[d][:,c],return_inverse=True)
            values=np.bincount(inv,weights=target[:,c])/np.bincount(inv)
            assert np.all(np.diff(values)>=0)
            arrays[d+str(c)+'_x']=knots;arrays[d+str(c)+'_y']=values
    np.savez_compressed(R/'ALIGNMENT.npz',**arrays)
    (R/'ALIGNMENT_PROTOCOL.json').write_text(json.dumps(dict(method='Domain-wide per-RGB-channel monotone quantile matching to equal-domain quantile midpoint',fit_pixels_per_image=16384,grid='rows/cols 2::4 after unchanged512 resize',quantiles=4097,scene_weighting='Equal scene; all source ROIs5 images and target groups1',ties='Duplicate source quantiles mapped to mean corresponding target quantiles; deterministic no dithering',application='Float32 before existing RGB/ImageNet normalization; unchanged spatial pixels and masks; no added clip',labels_used=False,scope='Uses unlabeled pixels of both evaluation datasets; appearance diagnostic not heldout generalization',brightness='RGB-weighted intensity .2126R+.7152G+.0722B, not physical luminance',range={d:[v.min(0).tolist(),v.max(0).tolist()] for d,v in quantiles.items()}),indent=2))

class Alignment:
    def __init__(self,root):self.root=root;self.maps=dict(np.load(root/'ALIGNMENT.npz'));self.stats=[]
    def apply(self,bgr,d,index):
        rgb=bgr[::-1].transpose(1,2,0).astype(np.float32);out=np.empty_like(rgb)
        for c in range(3):out[...,c]=np.interp(rgb[...,c],self.maps[d+str(c)+'_x'],self.maps[d+str(c)+'_y'])
        entry=dict(index=index,domain=d,outside_fit_range=[float(((rgb[...,c]<self.maps[d+str(c)+'_x'][0])|(rgb[...,c]>self.maps[d+str(c)+'_x'][-1])).mean()) for c in range(3)])
        for name,a in [('before',rgb),('after',out)]:
            vals=np.concatenate([a,(a@np.array([.2126,.7152,.0722]))[...,None]],axis=-1).reshape(-1,4)
            sample=vals.reshape(512,512,4)[2::4,2::4].reshape(-1,4)
            entry[name]=dict(mean=vals.mean(0).tolist(),second=np.square(vals).mean(0).tolist(),quantiles=np.quantile(sample,np.linspace(0,1,257),axis=0).tolist(),outside_0_255=float(((a<0)|(a>255)).mean()),endpoint_fraction=float(((a<=0)|(a>=255)).mean()))
        self.stats.append(entry)
        return np.ascontiguousarray(out.transpose(2,0,1)[::-1])
    def finish(self):(self.root/'APPEARANCE_STATS.json').write_text(json.dumps(self.stats))

if __name__=='__main__':fit()
