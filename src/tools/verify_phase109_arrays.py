"""Phase109: synthetic arrays only; never import torch, rasterio or model packages."""
import argparse,ast,hashlib,json,pathlib,typing,__future__
import numpy as np

def extract(path,names):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected}==set(names)
    env={'np':np,'Optional':typing.Optional}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),env)
    return env

def softmax(a):
    z=a-np.max(a,axis=0,keepdims=True);e=np.exp(z);return e/e.sum(axis=0,keepdims=True)

def main(root):
    s=root/'sources';checks=[]
    def ok(name,condition):
        assert bool(condition),name;checks.append({'name':name,'passed':True})
    current=extract(s/'omni_omnicloudmask_model_utils.py',['channel_norm'])
    legacy=extract(s/'v1_20240925__omnicloudmask__model_utils.py',['channel_norm'])
    modern_r=extract(s/'omnicloudmask__raster_utils.py',['mask_prediction','make_patch_indexes'])
    legacy_r=extract(s/'v1_20240925__omnicloudmask__raster_utils.py',['mask_prediction','make_patch_indexes'])
    a=np.array([[[0,1,2],[3,4,0]],[[0,5,5],[5,5,0]],[[0,0,0],[0,0,0]]],dtype=np.float32)
    n=current['channel_norm'](a);old=legacy['channel_norm'](a)
    ok('author_channel_norm_zero_constant_all_nodata',np.array_equal(n,old) and np.all(n[a==0]==0) and np.all(n[1:]==0))
    ok('author_channel_norm_population_std',np.isclose(n[0][a[0]!=0].mean(),0) and np.isclose(n[0][a[0]!=0].std(),1))
    b=np.where(a!=0,3*a+7,0)
    ok('per_channel_positive_affine_with_same_valid_support',np.allclose(n,current['channel_norm'](b),atol=1e-6))
    image=np.ones((3,4,4),dtype=np.float32);image[0,1,1]=0;image[:,2,2]=0
    nc,mask=modern_r['mask_prediction'](image,np.ones((4,4,4),np.float32))
    nv=legacy_r['mask_prediction'](image,np.ones((4,4,4),np.float32))
    ok('legacy_vs_current_partial_nodata_semantics_detected',mask[1,1]==1 and nc[3,1,1]==1 and nv[3,1,1]==0 and mask[2,2]==0)
    indices=modern_r['make_patch_indexes'](13,11,8,3)
    ok('author_window_indices_legacy_current_match',indices==legacy_r['make_patch_indexes'](13,11,8,3))
    cover=np.zeros((11,13),np.int32)
    for top,bottom,left,right in set(indices):cover[top:bottom,left:right]+=1
    ok('edge_windows_no_padding_cover_entire_grid',cover.min()>0 and all(b-t==8 and r-l==8 and 0<=t<b<=11 and 0<=l<r<=13 for t,b,l,r in indices))
    logits=np.zeros((2,4,2,2),np.float64);logits[0,0]=9;logits[1,3]=1
    correct=softmax(logits.mean(axis=0));wrong=np.stack([softmax(x) for x in logits]).mean(axis=0)
    ok('average_outputs_then_softmax_not_probability_average',not np.allclose(correct,wrong))
    confidence=np.clip(correct+0.001,0.001,0.999)
    ok('author_confidence_not_probability_simplex',np.all(confidence>=.001) and np.all(confidence<=.999) and not np.allclose(confidence.sum(axis=0),1))
    ok('class_axis_shadow_channel_is_three',confidence.shape==(4,2,2) and confidence[3].shape==(2,2))
    tracker=np.zeros((4,3,4));weight=np.zeros((3,4));values=np.array([1,2,3,4.])[:,None,None]
    for left,right,gain in [(0,3,1.),(1,4,2.)]:
        tracker[:,:,left:right]+=values*gain;weight[:,left:right]+=gain
    stitched=tracker/weight
    ok('class_independent_weighted_overlap_constant_reconstruction',np.allclose(stitched,values))
    ok('hard_argmax_equivalent_after_positive_weight_division',np.array_equal(tracker.argmax(axis=0),stitched.argmax(axis=0)))
    valid=mask.astype(bool);valid[0,0]=False
    score=np.where(valid,confidence[3,0,0],np.nan)
    ok('uncovered_and_nodata_excluded_not_uniform_probability',np.isnan(score[0,0]) and np.isnan(score[2,2]) and np.isfinite(score[1,1]))
    # Metadata/source checks, no imagery, weights, labels, scores or model imports.
    legacy_rows=list(__import__('csv').DictReader((s/'v1_20240925__model_download_links.csv').open()))
    current_rows=[v for v in __import__('csv').DictReader((s/'omni_omnicloudmask_model_download_links.csv').open()) if v['version']=='1']
    ok('v1_requires_both_historical_model_ids',len(legacy_rows)==len(current_rows)==2 and {v['google_drive_id'] for v in legacy_rows}=={v['google_drive_id'] for v in current_rows})
    h=json.loads((s/'hf_model_info.json').read_text());match=[]
    for row in current_rows:
        name=pathlib.Path(row['file_name']).stem+'.safetensors';f=next(x for x in h['siblings'] if x['rfilename']==name);assert f['size']==f['lfs']['size'];match.append(f)
    ok('v1_declared_safetensors_payload',sum(x['size'] for x in match)==202479912)
    paths=[s/'omni_omnicloudmask_model_utils.py',s/'v1_20240925__omnicloudmask__model_utils.py',s/'omnicloudmask__raster_utils.py',s/'v1_20240925__omnicloudmask__raster_utils.py']
    result={'phase':109,'date':'2026-10-14','checks':checks,'n_passed':len(checks),'test_scope':'author NumPy-only functions extracted by AST plus algebraic synthetic-output checks; no upstream Torch execution or network','network_forwards':0,'weight_loads':0,'fits':0,'real_output_consistency_verified':False,'calibration_verified':False,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (root/'ARRAY_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({'passed':len(checks),'network_forwards':0,'weight_loads':0}))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=pathlib.Path,default=pathlib.Path('outputs/phase109/phase109_20261013'));main(parser.parse_args().root)
