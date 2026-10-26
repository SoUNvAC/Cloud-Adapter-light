"""Apply explicit review to existing tables only; preserve prereview report and all science."""
import base64
from phase318_common import *
def main():
 request=json.loads(base64.b64decode(sys.argv[1]));assert request['outcome'] in ['close','consider_necessary_repeat'] and request['rationale'];assert load(ROOT/'STATUS.json')['status']=='completed' and load(ROOT/'INDEPENDENT_ACCEPTANCE.json')['status']=='passed' and not (ROOT/'DELIVERY_REVIEW.json').exists()
 if request['outcome']=='consider_necessary_repeat':assert len(set(request['evidence_pools']))>=2 and set(request['evidence_pools'])<=set(['CMC','SPARCS','L8'])
 for n,h in load(ROOT/'GPU_CODE_LOCK.json').items():assert sha(R/n)==h,n
 for n,h in load(ROOT/'CPU_RUN_LOCK.json')['code'].items():assert sha(R/n)==h,n
 assert load(ROOT/'BACKUP_FORMAT_ACCEPTANCE.json')['status']=='passed'
 for n in ['RESULT.md','phase318.md','DECISION.csv']:shutil.copyfile(ROOT/n,ROOT/('PRE_DELIVERY_'+n))
 metrics=read(ROOT/'METRICS.csv');decisions=read(ROOT/'DECISION.csv')
 for row in decisions:row['interpretation']=request['rationale'];row['decision']=request['outcome']+'; no automatic nextphase/seed/view/loss'
 table(ROOT/'DECISION.csv',decisions);text=(ROOT/'RESULT.md').read_text();first=text.index('\n\n');second=text.index('\n\n',first+2);text=text[:first+2]+request['rationale']+text[second:];(ROOT/'RESULT.md').write_text(text,encoding='utf-8')
 with (ROOT/'phase318.md').open('a',encoding='utf-8') as f:f.write('\n- 实际结果复核决定：'+request['outcome']+'。'+request['rationale']+' 无新增训练/重推理/真值重选，未启动下一phase。\n')
 code=ROOT/'delivery_code';code.mkdir()
 for p in (R/'src/tools').glob('*phase318*.py'):shutil.copyfile(p,code/p.name)
 write(ROOT/'DELIVERY_REVIEW.json',dict(reviewed=True,request=request,metrics_sha256=sha(ROOT/'METRICS.csv'),selector_quality_sha256=sha(ROOT/'SELECTOR_QUALITY.csv'),all_scientific_outputs_unchanged=True,original_reports_preserved=True,no_new_label_or_neural_reads=True,code={p.name:sha(p) for p in code.iterdir()},epoch=time.time()))
 assert budget_bytes()<=12000000000;print('Phase318 actual-effect review sealed; scientific outputs unchanged')
if __name__=='__main__':main()
