"""Phase120 fixed identities, objective, and exact author encoder boundary."""
import hashlib
import inspect
from pathlib import Path
from phase119_common import (BASE,REPO,OLD,load,write,sha,ram,coordinates,image,
    catalog_image,verify_native_and_cache,setup as setup119)
ROOT=Path('/home/scv/shared/phase120_20261015')
LOCK=REPO/'src/research_results/phase120_20261015/PRE_EXECUTION_LOCK.json'

def setup():
    setup119()
    cfg=load(LOCK)
    amendment=REPO/'src/research_results/phase120_20261015/ENGINEERING_RECOVERY_LOCK.json'
    revised=load(amendment)['revised_source_sha256'] if amendment.exists() else {}
    for p,h in cfg['source_sha256'].items():assert sha(REPO/p)==revised.get(p,h),p
    return cfg

def decoder_names(model):
    from fastai.vision.models.unet import DynamicUnet
    from fastai.vision.learner import create_unet_model
    assert isinstance(model,DynamicUnet)
    # DynamicUnet.__init__ explicitly constructs layers=[encoder,...], then SequentialEx.
    source=inspect.getsource(DynamicUnet.__init__)
    assert 'layers = [encoder' in source
    assert model.layers[0] is model[0]
    enc=model.layers[0];ids={id(v) for v in enc.parameters()};buf={id(v) for v in enc.buffers()}
    frozen=[n for n,v in model.named_parameters() if id(v) in ids]
    frozen += [n for n,v in model.named_buffers() if id(v) in buf]
    train=[n for n,v in model.named_parameters() if id(v) not in ids]
    dec=[n for n in model.state_dict() if n not in frozen]
    assert len(ids)>0 and train and len(set(frozen+dec))==len(model.state_dict())
    return enc,train,dec

def digest_state(state):
    h=hashlib.sha256()
    for n,v in sorted(state.items()):
        h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def objectives(student,target,teacher=None):
    import torch
    F=torch.nn.functional
    parent=torch.stack((student[:,0],torch.logsumexp(student[:,1:3],dim=1),student[:,3]),dim=1)
    valid=target!=255
    assert valid.any()
    ce=F.cross_entropy(parent,target,ignore_index=255)
    kl=student.sum()*0
    if teacher is not None:
        kl=F.kl_div(F.log_softmax(student/2,dim=1),F.softmax(teacher.detach()/2,dim=1),reduction='none').sum(1)[valid].mean()*4
    return ce,kl
