"""Recovery transport: prune prediction directories before filesystem traversal."""
from pathlib import Path
import inspect
import collect_phase241 as transport
code=inspect.getsource(transport.ssh).replace('timeout=90','timeout=300').replace("'ServerAliveCountMax=2'","'ServerAliveCountMax=12'")
exec(compile(code,'transport_timeout_override','exec'),transport.__dict__)
original=transport.ssh
def compact_ssh(script):
 lines=script.splitlines();out=[]
 for line in lines:
  if line.startswith('files={str(p.relative_to(r))'):
   out.extend(['import os','files={}',"for root,dirs,names in os.walk(r):"," dirs[:]=[n for n in dirs if n not in {'independent','ablation_off','region_off','direction_off'}]"," for name in names:","  p=Path(root)/name","  if p.suffix in ['.json','.csv','.md','.npz','.jsonl']:files[str(p.relative_to(r))]=p.read_bytes()"])
  else:out.append(line)
 return original('\n'.join(out))
transport.ssh=compact_ssh
p=Path(__file__).with_name('collect_all492.py')
s=p.read_text(encoding='utf-8').replace('if not p.exists():',"if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=w['sha256']:")
exec(compile(s,str(p),'exec'),{'__file__':str(p),'__name__':'__main__'})
