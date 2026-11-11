"""Read-only runtime/config/data preflight for user-authorized two-head retraining."""
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

ROOT=Path('/home/scv/Cloud-Adapter-light'); SRC=ROOT/'src'
os.chdir(SRC);sys.path.insert(0,str(SRC));os.environ['XFORMERS_DISABLED']='1'
out={'status':'preflight_only','models_forwarded':0,'files':{},'targets':{},'errors':[]}
def file(p):
    p=Path(p);p=p if p.is_absolute() else SRC/p
    q={'path':str(p),'exists':p.is_file()}
    if p.is_file():q['bytes']=p.stat().st_size
    return q
for mod in ['torch','mmengine','mmcv','mmseg','mmdet']:
    try:
        m=importlib.import_module(mod);out[mod]={'version':m.__version__,'path':m.__file__}
    except Exception as e:out['errors'].append(mod+':'+repr(e))
try:
    import torch
    out['cuda']={'available':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'torch_cuda':torch.version.cuda,'compiled_arches':torch.cuda.get_arch_list()}
    smi=shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
    q=subprocess.run([smi,'--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);out['GPU_query']=q.stdout.strip()
    from mmengine.config import Config
    os.environ['PHASE64_METHOD']='shared_parent';os.environ['PHASE64_PARENT_CHECKPOINT']=str(SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth')
    for target in ['l8','sparcs']:
        os.environ['PHASE64_TARGET']=target
        cfg=Config.fromfile('configs/protocol/phase64_target_parent_rgb.py')
        ds=cfg.train_dataloader.dataset;vs=cfg.val_dataloader.dataset
        out['targets'][target]={'config_model_type':cfg.model.type,'init_cfg':dict(cfg.model.backbone.get('init_cfg',{})),'optimizer':str(cfg.optim_wrapper),'max_iters':cfg.train_cfg.max_iters,'val_interval':cfg.train_cfg.val_interval,'randomness':dict(cfg.randomness),'manifest':file(ds.manifest_path),'train_dataset':dict(ds),'val_dataset':dict(vs),'custom_hooks':[dict(x) for x in cfg.get('custom_hooks',[])]}
        if 'checkpoint' in cfg.model.backbone.get('init_cfg',{}):out['targets'][target]['DINO_init_file']=file(cfg.model.backbone.init_cfg.checkpoint)
        mp=Path(ds.manifest_path); rows=list(csv.DictReader(mp.open(encoding='utf-8-sig')))
        if target=='l8':
            meta=Path(ds.usgs_shadow_metadata_path);status={r['scene']:r['usgs_shadows'].lower() for r in csv.DictReader(meta.open(encoding='utf-8-sig'))};rows=[r for r in rows if status[r['scene']]=='yes']
        stats={}
        for split in ['target_train','target_val']:
            rr=[r for r in rows if r['new_split']==split]
            missing=[{k:r[k] for k in ['name','image_path','mask_path']} for r in rr if not Path(r['image_path']).is_file() or not Path(r['mask_path']).is_file()]
            stats[split]={'inputs':len(rr),'scenes':len({r['scene'] for r in rr}),'missing_image_or_mask_files':len(missing),'first_missing':missing[:2]}
        out['targets'][target]['metadata_counts']=stats
    source=SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth';ck=torch.load(source,map_location='cpu');sd=ck['state_dict']
    out['Source_state']={'entries':len(sd),'head_parameters_entries':len([n for n in sd if n.startswith('decode_head.')]),'non_head_entries':len([n for n in sd if not n.startswith('decode_head.')]),'first_backbone_keys':[n for n in sd if n.startswith('backbone.')][:8],'total_tensor_bytes':sum(v.numel()*v.element_size() for v in sd.values() if isinstance(v,torch.Tensor))}
except Exception as e:out['errors'].append('config_or_source:'+repr(e))
try:
    exe='/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    command='Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | Select-Object DeviceID,FreeSpace,Size | ConvertTo-Json -Compress'
    q=subprocess.run([exe,'-NoProfile','-NonInteractive','-Command',command],capture_output=True,text=True,timeout=35)
    out['host_fixed_disks']=json.loads(q.stdout.strip()) if q.returncode==0 else {'error':q.stderr}
except Exception as e:out['host_fixed_disks']={'unknown':repr(e)}
print('PHASE212_PREFLIGHT_JSON='+json.dumps(out,default=str))
