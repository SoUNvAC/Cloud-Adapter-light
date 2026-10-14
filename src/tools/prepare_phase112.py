"""Freeze fixed exploratory comparison before imagery inference/reference joining."""
import json,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[2]
O=R/'src/research_results/phase112_20261014'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False,parents=True)
 (O/'.gitattributes').write_text('* -text whitespace=cr-at-eol\n')
 files=[R/'src/research_plans/PHASE112_OCM_PIXBOX_COMPARISON_20261014.md',R/'src/research_results/phase110_20261014/B8A_OBJECTS.json',R/'src/research_results/phase104_20261013/object_manifest.json',R/'src/research_results/phase111_20261014/PRE_EXECUTION_LOCK.json',R/'src/research_results/phase111_20261014/FINAL_RECOVERY_VERIFICATION.json',R/'src/research_results/phase93_20261012/two_model_lock.json',R/'src/research_results/phase105_20261013/SCORES_SEALED.json',R/'src/research_results/phase106_20261013/SCORES_SEALED.json']
 files += [R/'src/research_results/phase105_20261013/task_lock.json',R/'src/research_results/phase106_20261013/input_score_lock.json']
 files+=sorted((R/'src/tools').glob('*phase112*.py'))
 prior=json.loads(files[4].read_text());assert prior['status']=='passed' and prior['cumulative_calls']==297
 objects=json.loads(files[1].read_text())['products'];existing=json.loads(files[3].read_text())['selected_product_id']
 assert len(objects)==16 and len({x['product_id'] for x in objects})==16
 obj={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':112,'source_sha256':{str(p.relative_to(R)).replace('\\','/'):sha(p) for p in files},'products':objects,'reuse_product':existing,'reuse_score_sha256':prior['score_sha256'],'order':[x['product_id'] for x in objects if x['product_id']!=existing],'main_cap':1920,'construction_cap':11,'repeat_cap':4,'total_cap':1935,'wall_seconds_cap':7200,'ram_cap':16*1024**3,'vram_cap':20*1024**3,'host_min_free':4000000000,'intermediate_cap':4000000000,'asset_download_cumulative_cap':4000000000,'thresholds':{'G7':json.loads(files[5].read_text())['models']['G7']['threshold'],'Omni':0.5},'comparison':'G7_153 minus Omni','bootstrap':{'seed':112014,'resamples':10000,'lower_must_exceed':0,'width_max_pp':20},'minimum_AP_groups':8,'positive_min':5,'negative_min':20,'no_training_or_fitting':True,'old_exposed_pool':True}
 (O/'PRE_EXECUTION_LOCK.json').write_text(json.dumps(obj,indent=2)+'\n')
 print('Phase112 frozen',len(obj['order']),'new scenes',sum(x['bytes'] for x in objects if x['product_id']!=existing),'B8A bytes')
if __name__=='__main__':main()
