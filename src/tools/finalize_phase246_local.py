"""Validate received intake arithmetic, preserving every pending status."""
from pathlib import Path
import json,csv,hashlib

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase246/phase246_20261031'

def main():
 a=json.loads((OUT/'DATA_ACCEPTANCE.json').read_text());archives=json.loads((OUT/'ARCHIVE_SUMMARY.json').read_text());recovered={r['scene']:r for r in json.loads((OUT/'METADATA_RECOVERY.json').read_text())['results']};checks={};table=[]
 for r in a['results']:
  s=r['scene'];grids=r.get('grids',{});native={int(k):int(v) for k,v in r.get('native_label_counts',{}).items()};joint={int(k):int(v) for k,v in r.get('joint_label_counts',{}).items()};classes=r.get('joint_classes',{})
  if native:
   h,w=grids['reference']['shape'];checks[s+'_native_total']=sum(native.values())==h*w
   checks[s+'_joint_total']=sum(joint.values())==r['input_valid_pixels']
   checks[s+'_joint_subset']=all(v<=native.get(k,0) for k,v in joint.items())
   checks[s+'_class_total']=sum(classes.values())==sum(joint.values())-joint.get(0,0)
   checks[s+'_class_mapping']=classes=={'Surface':joint.get(128,0),'Cloud':joint.get(192,0)+joint.get(255,0),'Shadow':joint.get(64,0)}
   checks[s+'_dn_hist_total']=all(sum(v.values())==r['input_valid_pixels'] for v in r['RGB_DN_histogram'].values())
  fields=recovered[s]['fields'];parts=s.split('_');wrs=parts[1];date=parts[2]
  identity=fields.get('WRS_PATH')==str(int(wrs[:3])) and fields.get('STARTING_ROW')==fields.get('ENDING_ROW')==str(int(wrs[3:])) and fields.get('ACQUISITION_DATE')==date
  checks[s+'_recovered_MTL_identity']=identity
  if grids:
   g=grids['reference'];tr=g['transform'];h,w=g['shape']
   checks[s+'_MTL_shape']=int(fields['PRODUCT_SAMPLES_REF'])==w and int(fields['PRODUCT_LINES_REF'])==h
   checks[s+'_MTL_pixel_center_UL']=abs(tr[2]+.5*tr[0]-float(fields['PRODUCT_UL_CORNER_MAPX']))<1e-6 and abs(tr[5]+.5*tr[4]-float(fields['PRODUCT_UL_CORNER_MAPY']))<1e-6
   checks[s+'_MTL_pixel_center_LR']=abs(tr[2]+(w-.5)*tr[0]-float(fields['PRODUCT_LR_CORNER_MAPX']))<1e-6 and abs(tr[5]+(h-.5)*tr[4]-float(fields['PRODUCT_LR_CORNER_MAPY']))<1e-6
  checks[s+'_RGB_radiance_coefficients']=all(k+str(b) in fields for b in [1,2,3] for k in ['LMAX_BAND','LMIN_BAND','QCALMIN_BAND','QCALMAX_BAND'])
  table.append({'scene':s,'status':'native_grid_pass_label_semantics_pending','same_grid':r.get('same_grid'),'MTL_identity_matches':identity,'Surface':None,'Cloud':None,'Shadow':None,'Fill':None,'raw_code_0':native.get(0,0),'raw_code_64':native.get(64,0),'raw_code_128':native.get(128,0),'raw_code_192':native.get(192,0),'raw_code_255':native.get(255,0),'radiometry':'LMIN/LMAX/QCAL/SUN/date recovered; model input transform not frozen','confirmation_ready':False})
 checks['fixed_three_scenes']=len(a['results'])==3 and {r['scene'] for r in a['results']}=={'Irish_026046_2001-03-24','Irish_035042_2001-08-14','Irish_039025_2001-07-25'}
 checks['no_model_training']=a['new_model_calls']==a['new_training']==0
 checks['archives_complete']=len(archives['results'])==3 and all(r.get('gzip_CRC_verified') for r in archives['results'])
 (OUT/'VALIDATION.json').write_text(json.dumps({'arithmetic_checks_passed':all(checks.values()),'checks':checks,'note':'Arithmetic consistency is not label-quality certification or model-input admission'},indent=2),encoding='utf-8')
 (OUT/'SEMANTIC_HOLD.json').write_text(json.dumps({'status':'all three scenes held; no reference mapping accepted','reason':'actual mask2019 raw-code distribution does not establish compatibility with official generic legacy legend; primary SSL4EO-L paper documents inconsistent mapping in this release','preserved_original_file':'DATA_ACCEPTANCE.json','invalidated_semantic_fields':['joint_classes','ignored_native_Fill','reference_valid_but_RGB_missing','MTL_identity_matches (extensionless MTL recovered separately)'],'still_valid_fields':['native_label_counts','joint_label_counts (joint with RGB validity only)','grids','native_file_sha256','RGB_DN_histogram'],'class_support_reported':False,'no_remapping_performed':True},indent=2),encoding='utf-8')
 with (OUT/'PER_SCENE_ACCEPTANCE.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 print(json.dumps(table,ensure_ascii=False,indent=2));print('checks_passed',all(checks.values()))

if __name__=='__main__':main()
