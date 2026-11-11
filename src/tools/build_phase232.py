"""Prepare a new, SHA-locked extraction; never alter historical execution code."""
from pathlib import Path
import ast,hashlib,json,shutil,zipfile
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase232/phase232_20261027';T=R/'src/tools'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
 assert not O.exists()
 f=R/'src/research_plans/phase232_frozen_20261027';lock=js(f/'LOCK.json')
 for n,h in {**js(f/'SHA256.json'),**lock['identities']}.items():assert sha(R/n)==h,n
 O.mkdir(parents=True);(O/'code').mkdir()
 s=(T/'execute_phase231.py').read_text(encoding='utf-8').replace('phase231_20261026','phase232_20261027').replace("lock['models_to_calibrate']","lock['models']").replace('240_000_000','420_000_000')
 s=s.replace("for d in ['d_cache','labels']:(R/d).mkdir()","(R/'d_cache').mkdir();OLD=Path('/home/scv/shared/phase231_20261026');old_receipts={(v['model'],v['scene']):v for v in js(OLD/'CALIBRATION_RECEIPTS.json')};assert js(R/'CALIBRATION_ACTUAL_INPUTS.json')==js(OLD/'CALIBRATION_ACTUAL_INPUTS.json')")
 start=s.index("   dp=R/f'd_cache/");end=s.index('   if i==0:',start)
 s=s[:start]+'''   old=old_receipts[name,v['scene']];op=OLD/old['d_path'];yp=OLD/old['label_path'];assert sha(op)==old['d_SHA'] and sha(yp)==old['label_SHA'];assert np.array_equal(d,np.load(op,allow_pickle=False)) and np.array_equal(y,np.load(yp,allow_pickle=False)),current+' drift'
   logs=np.log(np.maximum(p.astype(np.float64),1e-12));d3=logs[1]-np.logaddexp(logs[0],logs[2]);dp=R/f'd_cache/{name}__{v["scene"]}.npy';np.save(dp,d3,allow_pickle=False)
   yy=y[valid];dv=d3[valid];ce=np.logaddexp(np.logaddexp(logs[0],logs[1]),logs[2])[valid]-np.take_along_axis(logs,y.clip(0,2)[None],axis=0)[0][valid];bce=np.logaddexp(0.,dv)-(yy==1)*dv
   K=float(np.mean(ce-bce,dtype=np.float64));assert abs(float(ce.mean())-(K+float(bce.mean())))<1e-12
   rec={'model':name,'scene':v['scene'],'d_path':str(dp.relative_to(R)),'d_SHA':sha(dp),'label_path':str(yp),'label_SHA':sha(yp),'image_SHA':v['image_sha256'],'mask_SHA':v['mask_sha256'],'support':[int((y==k).sum()) for k in range(3)],'ignore':int((y==255).sum()),'K':K,'direct_L3_zero':float(ce.mean()),'d2_bitwise_equal':True,'label_bitwise_equal':True};receipts.append(rec);save('CALIBRATION_RECEIPTS.json',receipts)
''' + s[end:]
 (T/'execute_phase232.py').write_text(s,encoding='utf-8')
 c=(T/'controller_phase231.py').read_text().replace('phase231','phase232').replace('phase232_20261026','phase232_20261027')
 (T/'controller_phase232.py').write_text(c,encoding='utf-8')
 for n in ['LOCK.json','CALIBRATION_INPUTS.json']:shutil.copy2(f/n,O/n)
 for n in ['MODEL_PATHS.json','RUNTIME_CODE_SHA.json','EXPECTED_MODEL_ACCEPTED.json']:shutil.copy2(R/'outputs/phase231/phase231_20261026'/n,O/n)
 shutil.copy2(R/'src/research_plans/PHASE232_THREE_CLASS_SINGLE_BIAS_CONTROL.md',O/'PROTOCOL.md')
 for n in ['execute_phase232.py','controller_phase232.py','fit_phase232.py','phase231_bias.py']:
  ast.parse((T/n).read_text());shutil.copy2(T/n,O/'code'/n)
 save(O/'PREPARE_ACCEPTED.json',{'status':'passed','frozen_identities_verified':True,'planned_calls':183,'statistic_storage':'remote only; no old labels or score package copied','validation_opened':False})
 save(O/'BUNDLE_SHA.json',{p.relative_to(O).as_posix():sha(p) for p in O.rglob('*') if p.is_file()})
 with zipfile.ZipFile(O/'bootstrap.zip','x',zipfile.ZIP_DEFLATED) as z:
  for p in O.rglob('*'):
   if p.is_file() and p.name!='bootstrap.zip':z.write(p,p.relative_to(O).as_posix())
 (O/'HANDOFF.md').write_text('Current delivery: PREPARED, not launched. New protocol Phase232; 180 main plus 3 repeats, caps220/1800GPU seconds. d2 and labels must equal231 per input. d3 float64 and K remote only, reuse231 labels. Fit3 biases equal-scene all-class CE using fixed60 bisections. All3 biases SHA seal before any230 validation array. Independent solver must replay on shared d3, no new candidates. Then local Raw/B2/B3 nine rows, full transitions and rank invariants; no validation forwards, bootstrap, test or newphase. CPU3600seconds/8GiB; combined512MiB.\n',encoding='utf-8')
 print(json.dumps({'bootstrap_SHA':sha(O/'bootstrap.zip'),'bytes':(O/'bootstrap.zip').stat().st_size}))
if __name__=='__main__':main()
