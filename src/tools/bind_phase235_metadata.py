"""One bounded SSH metadata/hash binding of the eighteen fixed train raw masks."""
from pathlib import Path
import base64,hashlib,json,subprocess,time
REPO=Path(__file__).resolve().parents[2];R=REPO/'outputs/phase235/phase235_20261027'
def main():
 R.mkdir(parents=True,exist_ok=True);assert not (R/'TRAIN_RAW_METADATA.json').exists();rows=json.loads((REPO/'src/research_plans/phase235_frozen_20261027/L8_TRAIN_INPUTS.json').read_text());paths={v['scene']:f"/home/scv/shared/data/l8_biome_raw/l8biome/{v['biome']}/{v['scene']}/{v['scene']}_fixedmask.TIF" for v in rows};assert len(paths)==18
 code='''from pathlib import Path
import hashlib,json,sys
paths=json.load(sys.stdin);out={}
for scene,name in paths.items():
 p=Path(name);assert p.resolve().is_relative_to(Path('/home/scv/shared/data/l8_biome_raw/l8biome'));h=hashlib.sha256()
 if p.is_file():
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
 out[scene]={'path':name,'exists':p.is_file(),'bytes':p.stat().st_size if p.is_file() else None,'SHA':h.hexdigest() if p.is_file() else None}
print(json.dumps(out))
''';cmd="/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -c \"import base64;exec(base64.b64decode('"+base64.b64encode(code.encode()).decode()+"'))\""
 for a in range(4):
  p=subprocess.run(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs',cmd],input=json.dumps(paths).encode(),capture_output=True,timeout=90);(R/f'bind_metadata_attempt{a}.log').write_bytes(p.stderr)
  if p.returncode==0:break
  if a<3:time.sleep(5)
 assert p.returncode==0,p.stderr.decode();v=json.loads(p.stdout);(R/'TRAIN_RAW_METADATA.json').write_text(json.dumps({'epoch':time.time(),'scenes':v,'only_file_stat_and_hash_no_pixel_decode':True},indent=2));print(json.dumps({'bound_scenes':len(v),'missing':[g for g,x in v.items() if not x['exists']]}))
if __name__=='__main__':main()
