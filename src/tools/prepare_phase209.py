"""Freeze original calibration membership and the sole R4 scoring plan."""
import hashlib,json,shutil,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase209/phase209_20261020'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 O.mkdir(parents=True,exist_ok=True)
 assert not (O/'EXECUTION_LOCK.json').exists()
 member=R/'outputs/phase122/phase122_20261015/MEMBER_INPUT_LOCK.json';m=load(member)
 plan=load(R/'outputs/phase122/phase122_20261015/delivery/INPUT_PATCH_PLAN.json')
 rows=[q for q in plan['rows'] if q['pool']=='Calibration'];cal={g['group_id']:g for g in m['calibration']};fit={g['group_id'] for g in m['fit']}
 dev={q['group']['group_id'] for q in load(R/'outputs/phase208/phase208_20261020/delivery/EVALUATION_INPUT_PLAN.json')['rows'] if q['pool']=='Catalogue'}
 assert len(cal)==53 and len(fit)==100 and len(dev)==32 and not(cal.keys()&fit or cal.keys()&dev or fit&dev)
 for q in rows:
  g=cal[q['group']['group_id']]
  for k in ['product','group_id','image_member_sha256','mask_member_sha256','common_valid']:assert g[k]==q['group'][k]
 assert len(rows)==53 and sum(len(q['actual_patch_coordinates']) for q in rows)==212
 out=dict(rows=rows,member_lock_sha256=sha(member),calibration_groups=list(cal),fit_groups=sorted(fit),development_groups=sorted(dev),zero_group_overlap=True,network_main_calls=212,gate_calls=0,constructor_allowance=20,planned_upper_calls=232,cap=4000,GPU_seconds_cap=7200,CPU_bytes_cap=16*1024**3,GPU_bytes_cap=20*1024**3,new_bytes_cap=4000000000,host_free_bytes_floor=4000000000,calibration_only=True,reference_only_local=True,calendar_cutoff=None)
 (O/'CALIBRATION_PLAN.json').write_text(json.dumps(out,indent=2),encoding='utf8')
 for n in ['V4_WEIGHT_LOCK.json'] :shutil.copyfile(R/'outputs/phase208/phase208_20261020/delivery'/n,O/n)
 cache=dict(phase122='V1 RegNet R, not eligible as V4 R4',phase202='V1 E plus fitted readouts, not eligible as V4 R4',phase205_to_208='V4 R4 has evaluation only; no calibration scoring authorized/performed in those protocols',qualified_cache_found=False,decision='one original V4 R4 pass over all53; no M/B/E4 calibration',V4_weight_sha256=next(w['sha256'] for w in load(O/'V4_WEIGHT_LOCK.json')['weights'] if 'regnet' in w['filename']))
 (O/'CACHE_IDENTITY_AUDIT.json').write_text(json.dumps(cache,indent=2),encoding='utf8')
 sources=['phase119_common.py','phase118_common.py','phase200_common.py','install_phase111_runtime.py']
 lock=dict(files={n:sha(O/n) for n in ['CALIBRATION_PLAN.json','V4_WEIGHT_LOCK.json','CACHE_IDENTITY_AUDIT.json']},sources={n:sha(R/'src/tools'/n) for n in sources},protocol_sha256=sha(R/'src/research_plans/PHASE209_V4_THRESHOLD_VS_ADAPTER_20261020.md'),previous208_manifest_sha256=sha(R/'outputs/phase208/phase208_20261020/DELIVERY_COMPLETE.json'),new_code_sha256=sha(R/'src/tools/execute_phase209.py'),planned_calls=232,hard_cap=4000)
 (O/'EXECUTION_LOCK.json').write_text(json.dumps(lock,indent=2),encoding='utf8')
 with zipfile.ZipFile(O/'phase209_bootstrap.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for n in ['CALIBRATION_PLAN.json','V4_WEIGHT_LOCK.json','CACHE_IDENTITY_AUDIT.json','EXECUTION_LOCK.json']:z.write(O/n,n)
  z.write(R/'src/tools/execute_phase209.py','code/execute_phase209.py')
 print(json.dumps(dict(groups=53,main_calls=212,planned_upper=232,cap=4000,bootstrap_sha256=sha(O/'phase209_bootstrap.zip'))))
if __name__=='__main__':main()
