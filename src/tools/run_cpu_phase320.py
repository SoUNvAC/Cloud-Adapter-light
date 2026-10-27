"""Serial two-part CPU diagnosis under monitored limits; no automatic stage retries."""
from phase320_common import *
import subprocess,traceback
def main():
 assert not ROOT.exists();start=time.monotonic();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1');stages={}
 for stage in ['prepare','test','errors','threshold','audit','report']:
  guard();command=[sys.executable,'-B',str(R/'src/tools'/(stage+'_phase320.py'))];t=time.monotonic();peak=0
  if stage=='prepare':
   p=subprocess.run(command,cwd=R,env=env,capture_output=True,timeout=600);assert ROOT.exists(),'prepare failed before root creation: '+p.stderr.decode(errors='replace');(ROOT/'prepare.log').write_bytes(p.stdout+p.stderr);rc=p.returncode
  else:
   write(ROOT/'STATUS.json',dict(status='running',stage=stage,epoch=time.time(),pid=os.getpid()))
   with (ROOT/(stage+'.log')).open('xb') as f:
    p=subprocess.Popen(command,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
    try:
     while p.poll() is None:
      resources=guard();assert time.monotonic()-start<7200,'normal CPU time budget exceeded; preserve partial results and registered scratch'
      try:rss=int(next(l.split()[1] for l in Path('/proc',str(p.pid),'status').read_text().splitlines() if l.startswith('VmRSS:')))*1024;peak=max(peak,rss);assert rss<16*1024**3
      except (FileNotFoundError,StopIteration):pass
      append(ROOT/'RESOURCES.jsonl',dict(stage=stage,epoch=time.time(),tree_RSS=rss if 'rss' in locals() else 0,**resources));time.sleep(2)
     rc=p.returncode
    except BaseException:p.terminate();p.wait(timeout=30);raise
  stages[stage]=dict(exit=rc,seconds=time.monotonic()-t,peak_tree_RSS=peak);write(ROOT/'STAGES.json',stages);assert rc==0,'stage failed; inspect original log and retain all inputs/scratch: '+stage
 result=dict(status='completed',epoch=time.time(),seconds=time.monotonic()-start,stages=stages,resources=guard(),neural_calls=0,training_updates=0,scratch_remaining_files=[p.name for p in (ROOT/'tmp').glob('*') if p.is_file()]);assert not result['scratch_remaining_files']
 write(ROOT/'STATUS.json',result);(ROOT/'EXIT_CODE').write_text('0\n')
 (ROOT/'phase320.md').write_text('# Phase320 实验记录\n\n- 目标：一次交付错Shadow原生类别/边界/原几何状态分解，及PL单一全局阈值与CB默认点的取舍诊断。\n- 改动：只读既有317/319缓存和验收参考；完整SPARCS60/L8fit100；原val CMC32、SPARCS10、L8963图/8scene；精确float32排序/完整并列块、原AP口径、独立真值与计数复核。\n- 环境：gzs cloud-lite-pt210，CPU<=4线程；零训练、拟合、模型加载、前向、下载。\n- 止损：CPU2h，另1h修复复验余量；16GiB、排序临时6GiB、启动磁盘>=8GiB、宿主空闲>=4GiB、新持久双机含预留<=1GiB；仅删除明确登记的本轮排序临时文件，无自动重试。\n- 结果：详见RESULT/CONCLUSIONS与全部规定表。阈值为oracle_diagnostic，不冒充部署方法；标签不支持的暗地表/地形阴影/标注错误原因保持未知，历史窗口与候选结论不重写。\n- 实际成本：'+json.dumps(result,ensure_ascii=False)+'\n- 网络及完整备份：仅在本地流式回传与SHA实际验收后补真实回执；不设置定时任务，不自动执行321。\n',encoding='utf-8');print('Phase320 both CPU diagnostics complete',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  if ROOT.exists():write(ROOT/'FAILURE.json',dict(traceback=traceback.format_exc(),epoch=time.time(),automatic_retry=False));write(ROOT/'STATUS.json',dict(status='failed',epoch=time.time()));(ROOT/'EXIT_CODE').write_text('1\n')
  raise
