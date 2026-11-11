"""CPU-only final head/call audit and small shared delivery. No model calls."""
from pathlib import Path
import collections,hashlib,json,tarfile,time
ROOT=Path('/home/scv/shared/phase216_20261023');SRC=Path('/home/scv/Cloud-Adapter-light/src')
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    import torch,numpy as np
    from PIL import Image
    torch.set_num_threads(2)
    assert (ROOT/'EXIT_CODE').read_text().strip()=='0'
    complete=js(ROOT/'COMPLETE.json');assert complete['source_SHA_unchanged']
    assert sha(SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth')=='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
    ledger=[json.loads(x) for x in (ROOT/'CALL_LEDGER.jsonl').read_text().splitlines()]
    assert [r['ordinal'] for r in ledger]==list(range(1,len(ledger)+1))
    counts=collections.Counter(r['kind'] for r in ledger);assert counts==complete['counts'] and len(ledger)==25537
    assert counts['evaluation_full_segmentor']==5530 and counts['diagnostic_Source_full_segmentor']==2
    assert complete['GPU_stage_seconds']<14400 and complete['GPU_peak_allocated_bytes']<=20*1024**3 and complete['CPU_peak_bytes']<=16*1024**3
    schedule=js(ROOT/'TARGET_SOURCE_SCHEDULE.json');assert len(schedule)==4000
    request=js(ROOT/'SOURCE_POOL_REQUEST.json');sel=request['selected'];assert len(sel)==256
    assert all(r['iteration']==i+1 and r['source_sample_id']==sel[i%256]['sample_id'] for i,r in enumerate(schedule))
    assert sorted(collections.Counter(r['source_sample_id'] for r in schedule).values())==[15]*96+[16]*160
    initial=torch.load(ROOT/'heads/initial.pth',map_location='cpu')['state_dict']
    original=torch.load(SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth',map_location='cpu')['state_dict']
    assert all(torch.equal(t,original['decode_head.'+n]) for n,t in initial.items())
    headsha={p.name:sha(p) for p in (ROOT/'heads').glob('*.pth')};assert headsha==complete['head_SHA']
    # Independent target native-label/crop/flip check, no original transform code.
    mask_cache={};label_map=np.array([2,2,0,0,0,1,0],np.int64)
    for row in schedule:
        path=row['mask_path']
        if path not in mask_cache:
            with Image.open(path) as im:mask_cache[path]=np.asarray(im).copy()
        native=mask_cache[path];assert native.ndim==2
        draws=row['draws'];assert draws[0]['function']==draws[1]['function']=='randint'
        y,x=int(draws[0]['value']),int(draws[1]['value']);v=native[y:y+512,x:x+512]
        mapped=np.full(v.shape,255,np.int64);valid=v<len(label_map);mapped[valid]=label_map[v[valid]]
        if row['flip']:
            assert row['flip_direction']=='horizontal';mapped=np.flip(mapped,axis=1)
        assert hashlib.sha256(mapped[None].copy().tobytes()).hexdigest()==row['mask_SHA'],'independent target parent/crop mapping mismatch'
    assert len(mask_cache)==60
    norms={};lr_sequences={}
    for arm in ['C','D']:
        end=torch.load(ROOT/f'heads/{arm}_iter4000.pth',map_location='cpu')['state_dict'];assert set(end)==set(initial)
        norms[arm]=sum(float(((end[n].double()-v.double())**2).sum()) for n,v in initial.items())**.5
        for a in [.25,.5,.75]:
            head=torch.load(ROOT/f'heads/{arm}_a{a}.pth',map_location='cpu')['state_dict']
            assert all(torch.equal(head[n],(v.double()+a*(end[n].double()-v.double())).to(v.dtype)) for n,v in initial.items())
        losses=[json.loads(x) for x in (ROOT/f'{arm}_LOSSES.jsonl').read_text().splitlines()];assert len(losses)==4000
        init=js(ROOT/f'{arm}_INITIAL.json');assert sum(g['parameters'] for g in init['optimizer_groups'])==1316516
        lr_sequences[arm]=[r['lr'] for r in losses]
        assert all(r['step']==i+1 and r['target_input_SHA']==schedule[i]['image_SHA'] for i,r in enumerate(losses))
        assert arm=='D' or all(r['source_KL']==0 and r['source_sample_id'] is None for r in losses)
        assert arm=='C' or all(r['source_sample_id']==schedule[i]['source_sample_id'] and r['source_KL']>=-1e-6 for i,r in enumerate(losses))
        assert js(ROOT/f'{arm}_TRAIN_COMPLETE.json')['steps']==4000
    assert lr_sequences['C']==lr_sequences['D']
    access=js(ROOT/'ACCESS_AUDIT.json');assert not access['forbidden_attempts'] and access['source_train_label_reads']==0
    receipt={'status':'passed','epoch':time.time(),'calls_from_independent_ledger':dict(counts),'total':len(ledger),'initial_matches_Source':True,'all_internal_heads_exact_CPU_float64':True,'endpoint_delta_L2':norms,'head_SHA':headsha,'two_4000_schedules_exact':True,'target_parent_crop_flip_masks_independently_checked':4000,'target_training_scenes':60,'identical4000_LRs':True,'source_sequence_15or16':True,'new_model_calls':0}
    (ROOT/'REMOTE_CPU_ACCEPTANCE.json').write_text(json.dumps(receipt,indent=2))
    names=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix!='.pth' and not p.name.endswith('.tmp') and p.name!='DELIVERY_MANIFEST.json']
    manifest={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in names}
    (ROOT/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    archive=ROOT.parent/'phase216_delivery.tar.gz';assert not archive.exists()
    with tarfile.open(archive,'w:gz') as tar:
        for p in names+[ROOT/'DELIVERY_MANIFEST.json']:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    snapshot=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'SHA':sha(archive),'files':len(names),'remote_scope_snapshot_bytes':snapshot,'remote_archive_bytes':archive.stat().st_size,'CPU_acceptance_peak_bytes':__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss*1024,'receipt':receipt}))
if __name__=='__main__':main()
