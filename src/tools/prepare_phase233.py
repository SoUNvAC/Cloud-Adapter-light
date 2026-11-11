"""Build a fresh L8-only, identity-locked Fill sensitivity execution."""
from pathlib import Path
import ast,hashlib,json,shutil,zipfile
R=Path(__file__).resolve().parents[2];T=R/'src/tools';O=R/'outputs/phase233/phase233_20261027'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
 assert not O.exists();f=R/'src/research_plans/phase233_frozen_20261027';L=js(f/'LOCK.json')
 for n,h in {**L['identities'],**js(f/'SHA256.json')}.items():assert sha(R/n)==h,n
 rows=js(f/'L8_VAL_INPUTS.json');refs=js(f/'L8_NATIVE_REFERENCES.json');assert rows==js(R/'outputs/phase229/phase229_20261026/INPUT_SUPPORT.json')['L8'];assert len(rows)==len(refs)==963 and len({v['scene'] for v in rows})==8;assert {v['name'] for v in rows}=={v['name'] for v in refs};assert sum(v['invalid'] for v in refs)==33961651
 O.mkdir(parents=True);(O/'code').mkdir();(O/'expected').mkdir()
 for n in ['LOCK.json','L8_VAL_INPUTS.json','L8_NATIVE_REFERENCES.json']:shutil.copy2(f/n,O/n)
 save(O/'INPUT_SUPPORT.json',{'L8':rows})
 old=R/'outputs/phase229/phase229_20261026'
 for n in ['MODEL_PATHS.json','RUNTIME_CODE_SHA.json']:shutil.copy2(old/n,O/n)
 for model in L['models']:shutil.copy2(old/f'delivery/evaluation/L8_{model}.json',O/f'expected/{model}.json')
 shutil.copy2(old/'delivery/MODEL_ACCEPTED.json',O/'EXPECTED_MODEL_ACCEPTED.json')
 shutil.copy2(R/'src/research_plans/PHASE233_L8_FILL_SUPPORT_SENSITIVITY.md',O/'PROTOCOL.md')
 s=(T/'execute_phase229.py').read_text(encoding='utf-8');s=s[:s.index(' parity=[]')];s=s.replace('phase229_20261026','phase233_20261027').replace('assert len(allowed)==1946','assert len(allowed)==1926')
 pos=s.index(" os.chdir(SRC)")
 pre=''' import numpy as np
 refs=js(R/'L8_NATIVE_REFERENCES.json');refmap={v['name']:v for v in refs};assert len(refmap)==963
 oldmap={model:{v['name']:v for v in js(R/f'expected/{model}.json')['per_input']} for model in L['models']}
 assert all(len(v)==963 and all('prediction_SHA' in x for x in v.values()) for v in oldmap.values())
 ref_allowed={str(Path(v['file']).resolve()) for v in refs}
 def arrays_audit(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(args[0]))
   if p.suffix.lower() in ['.npz','.npy'] and str(p.resolve()) not in ref_allowed:raise RuntimeError('233 forbids nonreference arrays: '+str(p))
 sys.addaudithook(arrays_audit)
 def native_reference(v):
  entry=refmap[v['name']];p=Path(entry['file']);assert sha(p)==entry['sha256']
  with np.load(p,allow_pickle=False) as z:native=z['native'];valid=z['valid'];truth=z['truth']
  assert native.shape==valid.shape==truth.shape==(512,512) and valid.dtype==bool and np.isin(native,[0,64,128,192,255]).all();assert np.array_equal(valid,native!=0)
  parent=np.zeros(native.shape,np.uint8);parent[native==64]=2;parent[np.isin(native,[192,255])]=1
  assert np.array_equal(truth[valid],parent[valid]) and np.all(truth[~valid]==255);assert int((~valid).sum())==entry['invalid'] and int(valid.sum())==entry['support']
  return parent,valid
 stage='reference_preflight';status();fill=0;total=0
 from PIL import Image
 lut=np.array([0,2,1,1],np.uint8)
 for v in support['L8']:
  parent,valid=native_reference(v);png=np.asarray(Image.open(v['mask_path']));assert png.shape==(512,512) and np.isin(png,[0,1,2,3]).all();assert np.array_equal(parent,lut[png]);fill+=int((~valid).sum());total+=parent.size
 assert fill==33961651 and total==252444672;save('REFERENCE_ACCEPTED.json',{'inputs':963,'scenes':8,'Fill_pixels':fill,'OLD_pixels':total,'VALID_pixels':total-fill,'native_valid_truth_and_oldPNG_exact':True,'reference_source':'Phase317 only native/valid/truth, no models or predictions','reference_manifest_SHA':sha(R/'L8_NATIVE_REFERENCES.json')})
'''
 s=s[:pos]+pre+s[pos:]
 start=s.index(' def one(');s=s[:start]+''' (R/'evaluation').mkdir()
 for name in L['models']:
  stage='matrix';sethead(name);per=[];oldgroups=collections.defaultdict(lambda:np.zeros((3,3),np.int64));newgroups=collections.defaultdict(lambda:np.zeros((3,3),np.int64));histgroups=collections.defaultdict(lambda:np.zeros(3,np.int64))
  for i,v in enumerate(support['L8'],1):
   current=f'{name}/{i}/963';parent,valid=native_reference(v);x=torch.from_numpy(load_rgb_image(Path(v['image_path']),512)).cuda();count('matrix_complete_input')
   with torch.inference_mode():
    z=wrapper(x).float().clamp_min(0);pred=(z/z.sum(1,keepdim=True).clamp_min(1e-12)).argmax(1)[0].cpu().numpy().astype(np.uint8)
   old=np.bincount(3*parent.ravel()+pred.ravel(),minlength=9).reshape(3,3);new=np.bincount(3*parent[valid]+pred[valid],minlength=9).reshape(3,3);hist=np.bincount(pred[~valid],minlength=3);prediction_SHA=hashlib.sha256(pred.tobytes()).hexdigest();expected=oldmap[name][v['name']]
   assert old.tolist()==expected['confusion'] and prediction_SHA==expected['prediction_SHA'],current+' old prediction drift';delta=old-new;assert np.array_equal(delta[0],hist) and not delta[1:].any();assert expected['image_SHA']==v['image_sha256'] and expected['mask_SHA']==v['mask_sha256']
   row={'name':v['name'],'scene':v['scene'],'OLD':old.tolist(),'VALID':new.tolist(),'Fill_prediction_histogram':hist.tolist(),'Fill_pixels':int(hist.sum()),'prediction_SHA':prediction_SHA,'old_prediction_SHA_exact':True,'old_confusion_exact':True,'reference_SHA':refmap[v['name']]['sha256'],'image_SHA':v['image_sha256'],'mask_SHA':v['mask_sha256']};per.append(row);oldgroups[v['scene']]+=old;newgroups[v['scene']]+=new;histgroups[v['scene']]+=hist
   if i%100==0:status()
  save(f'evaluation/{name}.json',{'model':name,'per_input':per,'per_scene':{g:{'OLD':oldgroups[g].tolist(),'VALID':newgroups[g].tolist(),'Fill_prediction_histogram':histgroups[g].tolist()} for g in sorted(oldgroups)}});status()
 assert COUNTS['matrix_complete_input']==4815 and sum(COUNTS.values())<=5200 and not denied;done.set();torch.cuda.synchronize()
 save('COMPLETE.json',{'status':'complete','counts':dict(COUNTS),'total_calls':sum(COUNTS.values()),'GPU_stage_seconds':time.monotonic()-START,'GPU_peak_bytes':torch.cuda.max_memory_allocated(),'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved(),'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'training_updates':0,'test_pixels_read':0,'forbidden_accesses':denied,'all_OLD_predictions_confusions_exact229':True,'no_spatial_prediction_cache':'229 persisted only per-input SHA and confusion; no spatial predictions; no317 model predictions used'});stage='complete';status();(R/'EXIT_CODE').write_text('0\\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  done.set();(R/'FAILURE.txt').write_text(traceback.format_exc());save('FAILURE_COST.json',{'counts':dict(COUNTS),'seconds':time.monotonic()-START});status();(R/'EXIT_CODE').write_text('1\\n');raise
'''
 s=s.replace("receipts[name]={'endpoint_SHA'", "assert digest.hexdigest()==js(R/'EXPECTED_MODEL_ACCEPTED.json')['heads'][name]['interpolated_tensor_digest']\n  receipts[name]={'endpoint_SHA'")
 ast.parse(s);(T/'execute_phase233.py').write_text(s,encoding='utf-8');shutil.copy2(T/'execute_phase233.py',O/'code/execute_phase233.py')
 save(O/'PREPARE_ACCEPTED.json',{'status':'passed','all_frozen_SHA':True,'planned_calls':4815,'cap_calls':5200,'OLD_pixels':252444672,'Fill_pixels':33961651,'reference_only_317':True,'unchanged_229_inputs_models':True})
 save(O/'BUNDLE_SHA.json',{p.relative_to(O).as_posix():sha(p) for p in O.rglob('*') if p.is_file()})
 with zipfile.ZipFile(O/'bootstrap.zip','x',zipfile.ZIP_DEFLATED) as z:
  for n in [*js(O/'BUNDLE_SHA.json'),'BUNDLE_SHA.json']:z.write(O/n,n)
 (O/'HANDOFF.md').write_text('Current delivery: PREPARED. Only963L8val inputs/8scenes,5fixedmodels alpha.25. Native317reference SHA/valid/native/truth and exact oldPNG reconstruction required before inference.4815plannedcalls cap5200/3600GPU seconds/CPU16GiB/GPU20GiB/new512MiB. EverypredictionSHA/OLDcm exact229 orstop. VALID excludes nativeFill0 only; old-new onlySurface row equals Fillhist. ExpectedFill33961651/old252444672. No otherpool pixels, training, fitting, paramsearch. Retrieve onlyshared small counts, analyze/independentlyverify 10rows,3comparison paired8scene bootstrap10000seed22901 sharedindicesbothseeds. KeepOLD and cite229SPdirectiononly. No automaticpush/historycleanup/work_dirsync/nextphase.\n',encoding='utf-8')
 print(json.dumps({'bootstrap_bytes':(O/'bootstrap.zip').stat().st_size,'SHA':sha(O/'bootstrap.zip')}))
if __name__=='__main__':main()
