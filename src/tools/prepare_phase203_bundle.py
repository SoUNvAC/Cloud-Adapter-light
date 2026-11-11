"""Build immutable small no-label execution bundle, never weights or references."""
import ast,hashlib,json,shutil,sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase203/phase203_20261018';B=O/'remote_bundle'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
 assert not B.exists();B.mkdir();(B/'code').mkdir();script=R/'src/tools/execute_phase203.py';ast.parse(script.read_text())
 sys.path.insert(0,str(R/'src/tools'));from execute_phase203 import adapted_source
 original=(R/'src/tools/infer_phase122.py').read_text();adapt=adapted_source(original);ast.parse(adapt);assert '3500' not in adapt and "assert stage in ('Catalogue','PixBox')" in adapt and 'configure_model(models[0])' in adapt
 shutil.copyfile(script,B/'code/execute_phase203.py');shutil.copyfile(O/'EVALUATION_INPUT_PLAN.json',B/'EVALUATION_INPUT_PLAN.json');shutil.copyfile(R/'src/research_results/phase122_20261015/C100_PARAMETER_WHITELIST.json',B/'WHITELIST.json')
 accepted=['infer_phase122.py','phase118_common.py','phase119_common.py','phase120_common.py','install_phase111_runtime.py']
 models={'/home/scv/shared/phase122_20261015/train_C/decoder_initial.safetensors':'a0e2f32b41acd15404f412981cee2243c019cd14b8ab8143ebbd848b938b3f52','/home/scv/shared/phase122_20261015/C_decoder_final.safetensors':'59f88f462b85c6a2eb8725781085efbc474962e862dd0a2401f4b344d88b5b9e','/home/scv/shared/phase123_20261016/C_decoder_final.safetensors':'2ad9b8dea32e8cb06e33656fc0a4518a412a85722432bcc21abf604d463238dd'}
 cfg=dict(phase=203,source_sha256={str(p.relative_to(B)).replace('\\','/'):sha(p) for p in B.rglob('*') if p.is_file()},accepted_source_sha256={'src/tools/'+p:sha(R/'src/tools'/p) for p in accepted},model_source_sha256=models,spot_products=load(O/'PRE_EXECUTION_BUDGET.json')['spot_products'],protocol_sha256=sha(R/'src/research_plans/PHASE203_DECODER_PARAMETER_BN_MECHANISM_20261018.md'),authorization_sha256=sha(O/'AUTHORIZATION_5000.md'),budget=load(O/'PRE_EXECUTION_BUDGET.json'),adapter_source_sha256=hashlib.sha256(adapt.encode()).hexdigest(),no_training=True,no_fitting=True,no_calibration_inputs=True,reference_join_local_only=True)
 for path in (B/'EXECUTION_LOCK.json',O/'EXECUTION_LOCK.json'):
  with path.open('x',encoding='utf8') as f:json.dump(cfg,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(status='syntax_and_adapter_verified_bundle_sealed',bytes=sum(p.stat().st_size for p in B.rglob('*') if p.is_file()),planned_calls=4851,cap=5000)))
if __name__=='__main__':main()
