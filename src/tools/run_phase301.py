"""Resource-bounded, offline local CPU execution; preserve previous deliveries."""
import ctypes
from ctypes import wintypes
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import os
import time
if sys.platform == 'win32':
    from run_phase300_local import usage

R = Path(__file__).resolve().parents[2]
REMOTE = os.environ.get('PHASE301_REMOTE') == '1'
ROOT = Path('/home/scv/shared/phase301_20261020') if REMOTE else R/'outputs/phase301/phase301_20261020'
O = ROOT/'verified_run'


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length',wintypes.DWORD),('load',wintypes.DWORD)]+[(name,ctypes.c_ulonglong) for name in
        ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]


def available():
    if REMOTE:
        return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    value = MemoryStatus()
    value.length = ctypes.sizeof(value)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value))
    return value.avail_phys


def write(name,data):
    path = O/name
    temp = path.with_suffix('.writing')
    temp.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def main():
    assert not O.exists(), 'Use a fresh verified_run; do not overwrite a prior execution'
    O.mkdir(parents=True)
    (O/'logs').mkdir()
    write('START_LOCK.json',dict(start_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        date='20261020',neural_calls=0,training=0,downloads=0,
        protocol_sha256=hashlib.sha256((R/'src/research_plans/phase301.md').read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
        prior_output='outputs/phase301/phase301_20261020; preserved without overwrite'))
    started = time.monotonic()
    status = dict(status='running',peak_bytes=0,minimum_available_memory=available(),
                  minimum_disk_free=shutil.disk_usage(R).free,PID=None)
    child = None
    try:
        with (O/'logs/analysis.log').open('xb') as log:
            child = subprocess.Popen([sys.executable,'-B',str(R/'src/tools/analyze_phase301.py')],cwd=R,
                                     stdout=log,stderr=subprocess.STDOUT)
            status['PID'] = child.pid
            while child.poll() is None:
                status.update(seconds=time.monotonic()-started,
                    peak_bytes=max(status['peak_bytes'],process_memory(child.pid)),
                    minimum_available_memory=min(status['minimum_available_memory'],available()),
                    minimum_disk_free=min(status['minimum_disk_free'],shutil.disk_usage(R).free),
                    new_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()))
                assert status['seconds']<=3600 and status['peak_bytes']<=16*1024**3
                assert status['minimum_available_memory']>=4000000000 and status['minimum_disk_free']>=4000000000
                assert status['new_bytes']<=1000000000
                write('logs/RESOURCE_STATUS.json',status)
                time.sleep(1)
            assert child.returncode==0, 'See logs/analysis.log'
        accepted = json.loads((O/'CPU_ACCEPTANCE.json').read_text(encoding='utf-8-sig'))
        status.update(status='completed',seconds=time.monotonic()-started,
                      peak_bytes=max(status['peak_bytes'],accepted['peak_bytes']),neural_calls=0)
        write('RESOURCE_RECEIPT.json',status)
        write('logs/RESOURCE_STATUS.json',status)
        if REMOTE:
            with (O/'logs/finalize.log').open('xb') as log:
                subprocess.run([sys.executable,'-B',str(R/'src/tools/finalize_phase301.py')],cwd=R,
                               stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,3600-int(status['seconds'])))
            (ROOT/'EXIT_CODE').write_text('0\n')
        print(json.dumps(status))
    except BaseException:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()
        import traceback
        status.update(status='failed',failure=traceback.format_exc(),seconds=time.monotonic()-started)
        write('FAILURE.json',status)
        if REMOTE: (ROOT/'EXIT_CODE').write_text('1\n')
        raise


def process_memory(pid):
    if REMOTE:
        path = Path('/proc')/str(pid)/'status'
        if not path.exists(): return 0
        values = [int(line.split()[1])*1024 for line in path.read_text().splitlines() if line.startswith(('VmHWM:','VmRSS:'))]
        return max(values,default=0)
    return usage(pid)


if __name__=='__main__': main()
