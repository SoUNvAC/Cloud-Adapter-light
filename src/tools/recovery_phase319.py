"""Recovery01: serialize NumPy scalar metadata, retain every original rule and code byte."""
from phase319_common import *
import subprocess,traceback
def inputs_only():
 import inputs_phase319 as worker
 import numpy as np
 def save(p,v):
  def scalar(x):
   if isinstance(x,np.generic):return x.item()
   raise TypeError(type(x).__name__)
  p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False,default=scalar)+'\n',encoding='utf-8')
 worker.write=save;worker.main()
def main():
 guard();old=load(ROOT/'STAGES.json');assert old['prepare']['exit']==0 and old['inputs']['exit']==1
 assert 'Object of type bool_ is not JSON serializable' in (ROOT/'inputs.log').read_text()
 assert not (ROOT/'GEOMETRY_INPUTS.json').exists() and not (ROOT/'SELECTION_LOCK.json').exists() and not (ROOT/'QUALITY_LOCK.json').exists() and not (ROOT/'recovery01').exists()
 recovery=ROOT/'recovery01';recovery.mkdir()
 for n in ['STATUS.json','STAGES.json','FAILURE.json','RESOURCES.jsonl','inputs.log','prepare.log','ACCESS.jsonl','EXIT_CODE']:
  if (ROOT/n).exists():shutil.copyfile(ROOT/n,recovery/n)
 code=sha(Path(__file__));shutil.copyfile(Path(__file__),recovery/Path(__file__).name)
 write(ROOT/'RECOVERY01_LOCK.json',dict(status='frozen',sha256=code,path=str(Path(__file__)),original_preregistration_sha256=sha(ROOT/'PREREGISTRATION.json'),reason='all160 coordinate checks completed; final JSON write failed only on numpy.bool_; original rows were not persisted',change='JSON default converts NumPy scalar .item() only in metadata writer; all original code/criteria/geometry unchanged',repeated_input_identity_passes=1,geometry_attempts_before_recovery=0,quality_reads_before_recovery=0,epoch=time.time()))
 start=time.monotonic();prior=sum(s['seconds'] for s in old.values());stages={};env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
 for stage in ['inputs','geometry','quality','audit','report']:
  cmd=[sys.executable,'-B',str(Path(__file__)),'--inputs'] if stage=='inputs' else [sys.executable,'-B',str(R/'src/tools'/(stage+'_phase319.py'))]
  log=ROOT/('inputs_recovery01.log' if stage=='inputs' else stage+'.log');assert not log.exists();guard();t=time.monotonic();peak=0
  write(ROOT/'STATUS.json',dict(status='running',stage=stage,recovery='01',pid=os.getpid(),epoch=time.time()))
  with log.open('xb') as f:
   p=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
   try:
    while p.poll() is None:
     resources=guard();assert prior+time.monotonic()-start<10800
     try:rss=int(next(l.split()[1] for l in Path('/proc',str(p.pid),'status').read_text().splitlines() if l.startswith('VmRSS:')))*1024;peak=max(peak,rss);assert rss<16*1024**3
     except (FileNotFoundError,StopIteration):pass
     append(ROOT/'RESOURCES_RECOVERY01.jsonl',dict(stage=stage,epoch=time.time(),tree_RSS=rss if 'rss' in locals() else 0,**resources));time.sleep(2)
    rc=p.returncode
   except BaseException:p.terminate();p.wait(timeout=30);raise
  stages[stage]=dict(exit=rc,seconds=time.monotonic()-t,peak_tree_RSS=peak);write(ROOT/'STAGES_RECOVERY01.json',stages);assert rc==0,'Recovery01 stage failed; preserve and diagnose, no blind retry: '+stage
 assert sha(Path(__file__))==code
 result=dict(status='completed',epoch=time.time(),seconds_including_initial_failure=prior+time.monotonic()-start,decision=load(ROOT/'DECISION.json')['decision'],initial_stages=old,recovery_stages=stages,resources=guard(),neural_calls=0,training_updates=0,recovery01_sha256=code)
 write(ROOT/'STATUS.json',result);(ROOT/'EXIT_CODE').write_text('0\n')
 (ROOT/'phase319.md').write_text('# Phase319 实验记录\n\n- 目标：太阳几何能否挑出更可靠且有足够保留量的Shadow。\n- 改动：固定317缓存和160 fit图；真实MTL/原网格/原L8坐标；唯一200–12000m、±5度核、反方向及同k高分；先封存后独立CPU参考审计。零训练、拟合和前向。\n- 环境：gzs cloud-lite-pt210、至多4CPU线程；已有rasterio投影/WGS84测地基准及往返QA/proc监控，未安装或升级依赖。\n- 止损：CPU16GiB/3h，空闲RAM和磁盘>=4GiB，双机新持久含预留<=2GB；规则和投影近似准入误差均未降低。\n- 首次失败：160产品坐标核验完成后，NumPy bool_不能写JSON；无几何选择/质量读取。原失败完整保留recovery01/；独立恢复仅转JSON标量，因原记录未写成而重复一次坐标身份检查，未改变参数或既有代码。\n- 结果：'+result['decision']+'；见GO_NO_GO/RESULT/完整160图缺项，not_testable不解释为几何无效。\n- 实际资源：'+json.dumps(result,ensure_ascii=False)+'\n- 网络与完整备份：仅在本地实际验收后补回执，原日志不覆盖。\n',encoding='utf-8');print('Phase319 Recovery01 completed',result['decision'],flush=True)
if __name__=='__main__':
 if '--inputs' in sys.argv:inputs_only()
 else:
  try:main()
  except BaseException:write(ROOT/'FAILURE_RECOVERY01.json',dict(traceback=traceback.format_exc(),epoch=time.time(),automatic_retry=False));write(ROOT/'STATUS.json',dict(status='failed',recovery='01'));(ROOT/'EXIT_CODE').write_text('1\n');raise
