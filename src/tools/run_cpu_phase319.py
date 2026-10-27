"""Single serial CPU experiment. No automatic retries, no GPU or scientific local work."""
from phase319_common import *
import subprocess,traceback
def main():
 assert not ROOT.exists();start=time.monotonic();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1');stages={}
 for stage in ['prepare','inputs','geometry','quality','audit','report']:
  script=R/'src/tools'/(stage+'_phase319.py');command=[sys.executable,'-B',str(script)];guard();t=time.monotonic();peak=0
  # prepare owns creation of the unique run directory and immutable preregistration.
  if stage=='prepare':
   p=subprocess.run(command,cwd=R,env=env,capture_output=True,timeout=180);assert ROOT.exists(),'prepare failed before root creation: '+p.stderr.decode(errors='replace');(ROOT/'prepare.log').write_bytes(p.stdout+p.stderr);rc=p.returncode
  else:
   write(ROOT/'STATUS.json',dict(status='running',stage=stage,epoch=time.time(),pid=os.getpid()))
   with (ROOT/(stage+'.log')).open('xb') as f:
    p=subprocess.Popen(command,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
    try:
     while p.poll() is None:
      guard();assert time.monotonic()-start<10800,'total CPU wall budget exceeded'
      try:
       rss=int(next(l.split()[1] for l in Path('/proc',str(p.pid),'status').read_text().splitlines() if l.startswith('VmRSS:')))*1024;peak=max(peak,rss);assert rss<16*1024**3
      except (FileNotFoundError,StopIteration):pass
      append(ROOT/'RESOURCES.jsonl',dict(stage=stage,epoch=time.time(),tree_RSS=rss if 'rss' in locals() else 0,**guard()));time.sleep(2)
     rc=p.returncode
    except BaseException:p.terminate();p.wait(timeout=30);raise
  stages[stage]=dict(exit=rc,seconds=time.monotonic()-t,peak_tree_RSS=peak);write(ROOT/'STAGES.json',stages);assert rc==0,'stage failed; preserve evidence, no automatic rerun: '+stage
 write(ROOT/'STATUS.json',dict(status='completed',epoch=time.time(),seconds=time.monotonic()-start,decision=load(ROOT/'DECISION.json')['decision'],stages=stages,resources=guard(),neural_calls=0,training_updates=0));(ROOT/'EXIT_CODE').write_text('0\n')
 decision=load(ROOT/'DECISION.json')
 (ROOT/'phase319.md').write_text('# Phase319 实验记录\n\n- 目标：太阳方向能否提供胜过同数量原q高分及反方向的Shadow可靠性增量。\n- 改动：固定Source缓存、160 fit图；真实MTL/CRS/affine、原L8坐标恒等；200–12000m/±5度唯一核，先封存后独立CPU真值审计；无训练/拟合/新增前向。\n- 环境：gzs cloud-lite-pt210，CPU至多4线程。缺少pyproj/psutil，使用已有rasterio CRS、带基准点/往返验收的WGS84 Vincenty和/proc监控，未下载/升级。\n- 止损：CPU16GiB/3h，宿主空闲内存磁盘4GiB，双机新持久含预留<=2GB；无自动重跑或阈值修改。\n- 结果：'+decision['decision']+'；完整指标和逐项判据见GO_NO_GO.csv、RESULT.md，原14天窗口逐字继承。\n- 实际资源与阶段退出：'+json.dumps(load(ROOT/'STATUS.json'),ensure_ascii=False)+'\n- 网络和备份：在本地验SHA收尾后补充真实回执；本记录不预称已完成备份。\n',encoding='utf-8')
 print('Phase319 CPU experiment completed',decision['decision'],flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  if ROOT.exists():write(ROOT/'FAILURE.json',dict(traceback=traceback.format_exc(),epoch=time.time(),automatic_retry=False));write(ROOT/'STATUS.json',dict(status='failed',epoch=time.time()));(ROOT/'EXIT_CODE').write_text('1\n')
  raise
