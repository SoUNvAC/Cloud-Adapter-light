"""Seal Phase212's failed original-file gate, without loading any model/data."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/phase212/phase212_20261020'
SUMMARY = ROOT / 'src/research_results/phase212_20261020'


def write(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    inventory=json.loads((OUT/'REMOTE_ORIGINAL_INVENTORY.json').read_text(encoding='utf-8-sig'))
    src,l8,sp=inventory['originals']
    assert src['exists'] and src['SHA_match']
    assert not l8['exists'] and not sp['exists']
    assert not inventory['search_truncated']
    old=json.loads((ROOT/'src/work_dirs/phase64_pareto/summary.json').read_text(encoding='utf-8'))
    for target,item in [('l8',l8),('sparcs',sp)]:
        assert old['methods'][target]['shared_parent']['checkpoint_sha256']==item['expected_sha256']
    missing=[dict(target=t,restore_to=x['path'],expected_sha256=x['expected_sha256'],status='missing_original',recovery='user-controlled third-party backup; not accessed by agent') for t,x in [('L8',l8),('SPARCS',sp)]]
    with (OUT/'BLOCKING_INPUTS.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(missing[0]));w.writeheader();w.writerows(missing)
    lock={'status':'blocked_original_identity_gate','fixed_alphas':[0,0.25,0.5,0.75,1],
          'interpolation':'complete decode_head learnable floating parameters; CPU float64 then original dtype; endpoints loaded directly',
          'backbone_other_components':'must equal Source tensor-by-tensor; not yet audited because targets missing',
          'buffers':'must equal endpoints; not yet audited', 'precision':'FP32 per Phase212; no forward performed',
          'new_seed_training_fitting_threshold_search':False,
          'supports':{'CloudSEN_val_images':535,'L8_val_images':963,'L8_val_scenes':8,'SPARCS_val_actual_inputs':None,'SPARCS_manifest_status':'not yet locked; do not infer from75 scenes or old summary validation_count'},
          'cost_plan':{'SPARCS_input_symbol':'n; pending original manifest','internal_main_calls':'6099+3*n','all_endpoint_main_calls_if_no_qualified_cache':'3531+2*n','total_if_all_endpoints_missing':'9630+5*n; gate/failed forwards extra','actual_calls':0,'maximum_calls':16000,'GPU_seconds_cap':14400,'actual_GPU_seconds':0,'CPU_cap_bytes':16*1024**3,'GPU_cap_bytes':20*1024**3,'new_persistent_bytes_cap':4*1024**3,'host_physical_free_floor_bytes':4*1024**3,'Windows_physical_free_bytes':'unknown; no inference started'},
          'bootstrap':{'draws':10000,'seed':212020,'unit':'verified Source ROI/related group; target original scene','source_groups_not_newly_validated':True},
          'no_fallback_checkpoint_no_retraining':True,'gate_checks_before_any_call':['restore both exact originals and verify SHA','reconstruct original full backbone using original DINO initialization/runtime','compare all non-head params and buffers; head schema/classes','lock original input manifests/transforms/ignore/support/counts and budget','verify endpoint reproduction before full interpolation evaluation'],
          'no_timer_no_push_no_next_phase':True}
    write('EXECUTION_LOCK.json',lock)
    write('COST_AND_FAILURE_LEDGER.json',{'model_forward_calls':0,'GPU_stage_seconds':0,'fits':0,'training_steps':0,'image_label_pixels_read':False,'model_checkpoint_loads':0,'remote_mutations':0,'failure_records':[{'stage':'sandbox_SSH_alias_resolution','attempts':4,'outcome':'alias not resolved in restricted context; no remote execution/model calls'},{'stage':'escalated_one_line_readonly_probe','outcome':'remote Python quoting SyntaxError; zero model calls'},{'stage':'stdin_readonly_probe','outcome':'completed; Source hash passed, two target originals missing'}],'archive_scope':'bounded file-name inventory depth6; no archive payload opened; unrelated-phase files not consumed; no third-party host accessed','no_cleanup_or_work_dirs_sync':True})
    headers=['target','alpha_or_method','target_mIoU','source_mIoU','gain_vs_Source','source_loss','target_Shadow_IoU','source_Shadow_IoU','nondominated','status']
    with (OUT/'MAIN_TABLE_NOT_RUN.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=headers);w.writeheader()
        for t in ['L8','SPARCS']:
            for a in ['0','0.25','0.5','0.75','1','historical_MsRE_seed64','historical_LoRA_seed64']:
                w.writerow({'target':t,'alpha_or_method':a,'nondominated':'not_assessed','status':'not_run_missing_target_originals; no new performance or dominance claim'})
    report='''# Phase212：原件门检受阻，未启动推理

## 1．数据结果

已执行协议第2节的远端原件核验。三父类Source原件存在（95,875,704 bytes），SHA与预注册值一致。两个目标头原件缺失：

| 原件 | 原路径（远端仓库相对） | 结果 |
|---|---|---|
| Source | src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth | SHA匹配64dd9a…859dc9 |
| L8 shared-parent | src/work_dirs/phase64_target_parent/l8/shared_parent/seed64/best_mIoU_iter_3500.pth | 不存在；期望2d9dfa…576fe |
| SPARCS shared-parent | src/work_dirs/phase64_target_parent/sparcs/shared_parent/seed64/best_mIoU_iter_3000.pth | 不存在；期望69a49f…38827 |

允许的repo/shared目录内，深度≤6的同名文件／指定归档文件名盘点未找到这两个原件的替代保存路径（列举165515个文件；非全深度、非逐归档内容搜索）。同名其他phase权重不作为替代。第三方备份未访问或独立验证；没有自动同步work_dirs、读取旧34GB备份或恢复历史大文件。

**本轮实际：0训练、0拟合、0模型加载、0前向、0GPU阶段、0影像／标签像元读取。** 不生成插值头、性能结果、bootstrap或非支配结论。主表占位均为未运行，不能把历史参考值当本轮复现。原摘要仅核对两个目标头SHA，未复用其旧四类Source基线和旧遗忘结论。

固定α=0.25／0.5／0.75及原端点。源535张、L8目标963输入／8scene；SPARCS实际输入数n仍待原manifest核验，不能用scene数代入。三内部点主调用计划6099+3n；若端点均无合格缓存，总主调用9630+5n，另计门检／失败。均未批准启动，仍须满足16000调用／4h GPU等原预算。

## 2．结论与恢复条件

这是**原件缺失导致的执行障碍，不是插值方法的正／负实验结果**。尚不能判断普通插值的保源折中，也不能判断MsRE／LoRA是否被覆盖。

恢复只需先从用户已确认的第三方备份找回上述两个原checkpoint，放回原远端路径，或放在/home/scv/shared内可核验的明确路径；完整期望SHA见BLOCKING_INPUTS.csv。代理未尝试连接第三方主机、下载新权重或重训。恢复后先核验SHA，再检查完整模型恢复、backbone／其他组件及buffer逐张量一致性，锁定实际支持与预算并复现端点，之后才执行固定插值。原DINO初始化、runtime或输入如有进一步缺项，须如实列出，不能随机补齐。

本轮状态：**BLOCKED_INPUTS**，Phase212实验尚未完成。未建定时任务、未push、未启动其他phase。
'''
    (OUT/'RESULT.md').write_text(report,encoding='utf-8')
    recovery='''# Phase212 恢复所需原件

第三方备份由用户管理，本轮没有访问该主机。无需恢复整个work_dirs；优先找回以下两个精确文件。

1. L8：恢复至`/home/scv/Cloud-Adapter-light/src/work_dirs/phase64_target_parent/l8/shared_parent/seed64/best_mIoU_iter_3500.pth`。
   SHA256：`2d9dfac9a8182445e52cc284cbd475f3c1ed413873305e2215565e50d3c576fe`。
2. SPARCS：恢复至`/home/scv/Cloud-Adapter-light/src/work_dirs/phase64_target_parent/sparcs/shared_parent/seed64/best_mIoU_iter_3000.pth`。
   SHA256：`69a49f09718a7e0a132a05ea36fb8d9150f90087233a3b1b3475a86c43b38827`。

也可将精确原件放在`/home/scv/shared/`下明确的新目录，随后按上述SHA核验身份。不要覆盖现有其他phase文件，不改用最后一步／相近迭代checkpoint。Source已保留且SHA匹配，当前无需重复恢复它。

找回后尚需完成原完整backbone恢复与参数／buffer同一性、支持与资源、端点复现门检；这两文件恢复不等于所有门检已通过。没有请求新训练授权或新的α搜索。
'''
    (OUT/'RECOVERY_REQUIREMENTS.md').write_text(recovery,encoding='utf-8')
    hashes={}
    for rel in ['src/research_plans/PHASE212_VFM_HEAD_WEIGHT_INTERPOLATION_20261020.md','src/research_plans/EXPLORATION_POLICY_20261014.md','src/research_plans/PHASE64_FINAL_RESULT_AUDIT.md','src/work_dirs/phase64_pareto/summary.json','src/tools/evaluate_phase64_rgb_recheck.py','src/tools/evaluate_phase64_source_only_parent.py','src/tools/export_phase12_onnx.py','src/tools/probe_phase212_originals.py','src/tools/finalize_phase212_preflight.py']:
        b=(ROOT/rel).read_bytes();hashes[rel]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    write('INPUT_CODE_HASHES.json',hashes)
    write('PREFLIGHT_ACCEPTANCE.json',{'status':'verified_blocked_inputs_not_experiment_complete','Source_SHA_pass':src['SHA_match'],'missing_originals':2,'both_target_hashes_match_original_summary_metadata':True,'interpolation_gate_passed':False,'backbone_buffer_identity':'not_checked_missing_originals','endpoint_reproduction':'not_run','metrics_bootstrap_dominance':'not_computed','model_calls':0})
    SUMMARY.mkdir(parents=True,exist_ok=True)
    for n in ['RESULT.md','RECOVERY_REQUIREMENTS.md','BLOCKING_INPUTS.csv','EXECUTION_LOCK.json','COST_AND_FAILURE_LEDGER.json','PREFLIGHT_ACCEPTANCE.json','INPUT_CODE_HASHES.json']:
        shutil.copyfile(OUT/n,SUMMARY/n)
    (SUMMARY/'README.md').write_text('# Phase212\n\n原件门检已执行，实验因L8／SPARCS两个原checkpoint缺失而受阻。Source SHA匹配；零模型调用。\n\n数据结果与结论见RESULT.md；两个精确恢复路径与SHA见RECOVERY_REQUIREMENTS.md。未完成插值／评价，不把门检交付称实验完成。\n',encoding='utf-8')
    files={str(p.relative_to(OUT)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in OUT.rglob('*') if p.is_file() and p.name!='PREFLIGHT_DELIVERY.json'}
    write('PREFLIGHT_DELIVERY.json',{'status':'blocked_inputs','experiment_complete':False,'preflight_report_delivered':True,'files':files})
    print(json.dumps({'status':'blocked_inputs','missing_originals':missing,'model_calls':0,'report':str(OUT/'RESULT.md')},ensure_ascii=False))


if __name__=='__main__':
    main()
