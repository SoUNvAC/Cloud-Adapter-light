"""Read-only Phase400 admission. No model construction, training or inference."""
from pathlib import Path
import hashlib,json,subprocess,sys,os,time
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 import torch,numpy as np,shutil
 from mmengine.config import Config
 src=Path('/home/scv/Cloud-Adapter-light/src');sys.path.insert(0,str(src));os.chdir(src);shared=Path('/home/scv/shared');source=src/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth';prepared=shared/'data/sentinel2_cloud_mask_catalogue_4172871/phase65a_explore_20261008/manifest.json'
 smi=shutil.which('nvidia-smi') or next((str(p) for p in [Path('/usr/lib/wsl/lib/nvidia-smi'),Path('/usr/bin/nvidia-smi')] if p.exists()),None)
 d={'read_only':True,'model_calls':0,'training_updates':0,'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'source_exists':source.exists(),'manifest_exists':prepared.exists(),'run_exists':(shared/'phase400_20261031').exists(),'disk':{},'GPU':subprocess.check_output([smi,'--query-gpu=name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True) if smi else {'name':torch.cuda.get_device_name(),'free_total':torch.cuda.mem_get_info()},'GPU_processes':subprocess.check_output([smi,'--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True) if smi else 'not available','cpu_count':os.cpu_count()}
 for p in (src,shared):d['disk'][str(p)]=shutil.disk_usage(p)._asdict()
 d['meminfo']={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
 if source.exists():d['source_SHA']=sha(source);assert d['source_SHA']=='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
 if prepared.exists():
  m=json.loads(prepared.read_text());rows=[r for r in m['rows'] if (r['split']=='fit' and r['is_representative']) or r['split']=='development_val'];d['manifest_SHA']=sha(prepared);d['manifest_scope']=m['scope'];d['original_split_SHA']=m['original_split_sha256'];d['counts']={'fit_products_all':sum(r['split']=='fit' for r in m['rows']),'fit_representatives':sum(r['split']=='fit' for r in rows),'development':sum(r['split']=='development_val' for r in rows)};d['prepared_array_missing']=[str(prepared.parent/r[k]) for r in rows for k in ('rgb_path','mask_path','image_valid_path') if not (prepared.parent/r[k]).is_file()];d['prepared_root']=str(prepared.parent)
 roots=[src/'data/cloudsen12_high_l1c',shared/'data/cloudsen12_high_l1c'];d['source_val_candidates']=[{'root':str(p),'images':len(list((p/'img_dir/val').glob('*.png'))),'labels':len(list((p/'ann_dir/val').glob('*.png')))} for p in roots]
 os.environ['PHASE65_PREPARED_MANIFEST']=str(prepared);os.environ['PHASE65_SOURCE_CHECKPOINT']=str(source);cfg=Config.fromfile(str(src/'configs/protocol/phase65a_explore_rgb.py'));d['resolved_65a']={k:cfg[k].to_dict() if hasattr(cfg[k],'to_dict') else cfg[k] for k in ('optim_wrapper','param_scheduler','train_cfg','train_pipeline','test_pipeline','model')};d['resolved_65a']['data_preprocessor']=cfg.model.get('data_preprocessor');d['phase65_config_SHA']=sha(src/'configs/protocol/phase65a_explore_rgb.py')
 d['host_volumes']=json.loads(subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',"Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,FreeSpace | ConvertTo-Json -Compress"],text=True));print(json.dumps(d,default=str))
if __name__=='__main__':main()
