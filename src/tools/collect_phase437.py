"""Fetch small Phase437 evidence files and current remote status."""
import json
from pathlib import Path
from collect_phase241 import ssh

script="""from pathlib import Path
import json,subprocess
r=Path('/home/scv/shared/phase437_20261103')
files={str(p.relative_to(r)):p.read_text() for p in r.rglob('*') if p.is_file() and p.suffix in ('.json','.md','.csv') and p.stat().st_size<4000000}
tails={p.name:p.read_text(errors='replace')[-1800:] for p in r.glob('*.log')}
gpu=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv']).decode()
print(json.dumps(dict(files=files,tails=tails,gpu=gpu)))
"""
if __name__=='__main__':
    data=json.loads(ssh(script))
    root=Path(__file__).resolve().parents[2]/'outputs/phase437/phase437_20261103'
    root.mkdir(parents=True,exist_ok=True)
    for name,content in data['files'].items():
        path=root/name
        assert path.resolve().is_relative_to(root.resolve())
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(content,encoding='utf-8')
    (root/'LAST_STATUS.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    checks={k:json.loads(v) for k,v in data['files'].items() if Path(k).name in ('INITIAL.json','UPDATE_CHECK.json','FAILURE.json','STATUS.json','COMPLETE.json')}
    for value in checks.values():
        if isinstance(value,dict):
            for key in ('best','final','summary'):
                if key in value:value[key]='See downloaded file'
    print(json.dumps(dict(gpu=data['gpu'],tails=data['tails'],checks=checks),ensure_ascii=False))
