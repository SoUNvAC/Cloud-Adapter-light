"""Accept a Phase207 engineering receipt; never imports or calls a model."""
import ast,base64,csv,hashlib,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase207/phase207_20261020'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(n,v):
 with (O/n).open('x',encoding='utf8') as f:f.write(v+'\n')
def save(n,v):put(n,json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def main():
 snap=load(O/'GATE_SNAPSHOT.json');assert snap['exit_code']=='0';dest=O/'receipts';dest.mkdir(exist_ok=True)
 for n,q in snap['files'].items():
  b=base64.b64decode(q['base64']);assert hashlib.sha256(b).hexdigest()==q['sha256'];assert not n.startswith('/') and '..' not in Path(n).parts and '\\' not in n
  (dest/n).parent.mkdir(parents=True,exist_ok=True)
  if (dest/n).exists():assert (dest/n).read_bytes()==b
  else:
   with (dest/n).open('xb') as f:f.write(b)
 g=load(dest/'IMPLEMENTATION_GATE.json');gr=load(dest/'GRADIENT_RECEIPT.json');arch=load(dest/'ARCHITECTURE.json');frozen=load(dest/'FROZEN_ORIGINALS.json');pre=load(O/'PRE_EXECUTION_LOCK.json');stat=load(dest/'RUN_STATUS.json');static=load(O/'STATIC_METHOD_AUDIT.json');ledger=[json.loads(t) for t in (dest/'CALL_LEDGER.jsonl').read_text().splitlines()]
 assert g['status']=='passed' and g['neural_calls']==len(ledger)<=24 and [x['call'] for x in ledger]==list(range(1,len(ledger)+1));assert g['optimizer_steps']==0 and g['backward_calls']==1
 assert stat['status']=='completed' and stat['stage_seconds']<=900 and stat['CPU_peak_bytes']<=16*1024**3 and stat['GPU_peak_reserved_bytes']<=20*1024**3
 assert g['disabled_output_max_error']<=1e-5 and not g['strict_zero_output_init'] and g['scale_init']==.001 and g['all_originals_unchanged']
 assert not gr['new_module_has_BN'] and gr['new_buffer_count']==0 and gr['original_buffers_and_parameters_unchanged'] and gr['original_parameters_all_no_grad'] and gr['original_R4_N4_all_eval']
 node=next(n for n in ast.parse((R/'src/cloud_adapter/models/backbones/target_msre_dinov2.py').read_text(encoding='utf8')).body if isinstance(n,ast.ClassDef) and n.name=='SparseMsREResidual');normalized_AST=ast.dump(node,include_attributes=False).replace(', type_params=[]','');normalized_sha=hashlib.sha256(normalized_AST.encode()).hexdigest()
 assert g['core_class_source_sha256']==sha(R/'src/cloud_adapter/models/backbones/target_msre_dinov2.py')
 assert g['core_class_AST_sha256']==normalized_sha and g['new_parameters']==970369 and g['core_parameters']==354433 and g['input_projection_parameters']==308736 and g['output_projection_parameters']==307200
 assert all(not p['requires_grad'] for p in frozen['parameters']) and len(frozen['identities'])==2
 assert [r['channels'] for r in arch['encoder_features'][-4:]]==[48,104,208,440]
 for prefix in ('core.','input_projections.','output_projections.'):
  assert any(r['has_gradient'] and r['finite'] and r['nonzero'] for r in gr['rows'] if r['name'].startswith(prefix))
 assert all(r['finite'] for r in gr['rows'] if r['has_gradient']);assert sha(O/'PRE_EXECUTION_LOCK.json')==load(dest/'INPUT_RECEIPT.json')['prelock_sha256']
 assert load(dest/'PRE_EXECUTION_LOCK.json')==pre
 for name,h in load(dest/'CODE_LOCK.json')['files'].items():assert sha(R/'src/tools'/name)==h
 for row in frozen['identities']:
  assert any(row['sha256']==w['sha256'] and Path(row['path']).name==w['filename'] for w in load(dest/'V4_WEIGHT_LOCK.json')['weights'])
 save('AST_PORTABILITY_ACCEPTANCE.json',dict(status='passed',local_raw_AST_hash=static['original_core_class_AST_sha256'],normalized_AST_hash=normalized_sha,remote_Python310_AST_hash=g['core_class_AST_sha256'],source_bytes_SHA_exact=True,normalization='removeonlyemptytype_paramsintroducedinPython3.12; preserveallcoreoperations'))
 save('IMPLEMENTATION_GATE.json',g)
 no_grad=[r['name'] for r in gr['rows'] if not r['has_gradient']];zero=[r['name'] for r in gr['rows'] if r['has_gradient'] and not r['nonzero']]
 feature_text='；'.join(str(r['shape']) for r in arch['encoder_features']);loss=float(gr['loss_value'])
 recovery_note='首次标签数组形状错误发生于0次网络调用，失败6.635771秒已计入累计阶段墙钟并保存原记录；修复仅还原有效像元标签格网，不更换数据或结构。' if (dest/'RECOVERY_LOCK.json').exists() else '无工程恢复。'
 data=f'''# Phase207：MsRE接入V4的可行性审查

## 1．数据与工程结果

固定方案为“MsRE-inspired parallel CNN-interface adapter”。原SparseMsREResidual类通过AST原样提取：3/5/7深度卷积、尺度编码、4槽×16token×384维、token0保留softmax影响但不作value、共享投影及scale=1e-3都保留。原target_head_delta独立26144参数，本候选不含它；原核心354433参数。

R4实际encoder返回特征：{feature_text}。只取架构顺序最后四级48/104/208/440通道，在encoder输出到decoder接口并行接入；四级独立1x1输入/输出投影，核心跨级共享。新增970369参数，为原R4的{g['new_parameter_fraction_of_R4']*100:.2f}%。原R4/N4全部参数及buffer冻结/eval，新增路径无BN。

输入是原fit100按group_id排序第一组T19NBJ，固定左上512裁剪，预先锁定，未按云影量选位置。关闭新增路径最大输出差{g['disabled_output_max_error']:.9g}；开启后的初始化最大差{g['enabled_initial_output_max_error']:.9g}，平均绝对差{g['enabled_initial_output_mean_abs_error']:.9g}，明确不是零输出初始化。

仅一次固定父类CE反传，参考loss={loss:.6g}（工程量，不是精度结果）；0optimizer更新、0拟合、0全池评价。新增参数无梯度项{len(no_grad)}，有梯度但全零项{len(zero)}，完整逐项列表在receipts/GRADIENT_RECEIPT.json。原参数均无梯度，原参数与BN/其他buffer均未变；原encoder实际输出保持不变，decoder接收的特征因新增残差而变化。

总网络调用{g['neural_calls']}，阶段墙钟{stat['stage_seconds']:.3f}秒；GPU峰值reserved {stat['GPU_peak_reserved_bytes']}bytes，CPU峰值{stat['CPU_peak_bytes']}bytes。调用含构建，账本已核；并非配对部署测速。没有development/PixBox标签访问、下载、环境升级或训练效果结论。{recovery_note}

## 2．结论与适用范围

工程上可以接入，且梯度能穿过冻结decoder回到新增路径。保留了原计算核心，但不能称“原MsRE原样跨底座验证”：真实DINO CLS语义改为内部零占位、同网格逐层传播改为不同分辨率的并行CNN接口、加入独立通道投影。原初始化中的14patch公式按原代码继承，只是启发式，不代表CNN或该DINO配置的patch14语义。

这套实现足以成为以后单独预注册的小规模公平对照候选；尚无证据判断精度增量或部署价值。普通残差控制可保持同位置/同投影，以共享DW+点式MLP构成约944641参数，差25728（约2.65%），函数差异仍须披露；本轮不实现或训练第二套候选。候选不含head-delta，因此本方案不需要head-delta控制；以后若加必须补齐。

未来必须与原R4和完整E4并列，C4只作已存在的decoder适配背景；fit/采样/输入/seed/末步/评价支持统一。单seed两控制各1000步约2000训练调用+3438固定评价调用+构建余量，只有预算示例，不构成训练授权。旧4000步DINO/development选模型与新V4固定末步不可作底座因果比较。训练截止没有因本工程门延长，不自动进入下一phase。

完整对应表METHOD_MAPPING.csv；修改假设UNIQUE_PROPOSAL.md；未来控制FUTURE_CONTROLS.md；实际权重/输入/代码/冻结/梯度SHA见receipts和PRE_EXECUTION_LOCK.json。
'''
 assert len(data)<4500;put('RESULT.md',data)
 save('LOCAL_ACCEPTANCE.json',dict(status='passed',gradient_groups_verified=['core','input_projections','output_projections'],new_no_grad_parameters=no_grad,new_zero_gradient_parameters=zero,neural_calls=0,new_fits=0,source_AST_exact=True,shape_and_parameter_counts_verified=True,scope='engineeringonly; notaccuracy',validation_code_sha256=sha(Path(__file__))))
 total=sum(p.stat().st_size for p in O.rglob('*') if p.is_file());assert total<1000000000
 summary=R/'src/research_results/phase207_20261020';summary.mkdir(exist_ok=False)
 for n in ['RESULT.md','METHOD_MAPPING.csv','UNIQUE_PROPOSAL.md','FUTURE_CONTROLS.md','IMPLEMENTATION_GATE.json','LOCAL_ACCEPTANCE.json','PRE_EXECUTION_LOCK.json','STATIC_METHOD_AUDIT.json','CODE_LOCK.json']:shutil.copyfile(O/n,summary/n)
 manifest=dict(status='complete',files={p.relative_to(O).as_posix():sha(p) for p in O.rglob('*') if p.is_file()},summary={p.name:sha(p) for p in summary.iterdir()},neural_calls=g['neural_calls'],optimizer_steps=0,accuracy_evaluation=False)
 save('DELIVERY_COMPLETE.json',manifest)
 for n,h in manifest['files'].items():assert sha(O/n)==h
 print(json.dumps(dict(status='complete',calls=g['neural_calls'],disabled_error=g['disabled_output_max_error'],enabled_error=g['enabled_initial_output_max_error'],parameters=g['new_parameters'],seconds=stat['stage_seconds'],zero_new_grads=zero,no_new_grads=no_grad)))
if __name__=='__main__':main()
