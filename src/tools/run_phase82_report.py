"""Fixed four-row report, screening outcome and training/deployment cost."""
import json
from phase82_common import *
from run_phase80_analysis import csvwrite

def main():
    report=read(ROOT/'phase82_report.json');train=read(ROOT/'training_completed.json');pre=read(ROOT/'preflight.json')
    trace=[json.loads(s) for s in (ROOT/'loss_gradient_curve.jsonl').read_text().splitlines()]
    assert len(trace)==4000 and [x['iteration'] for x in trace]==list(range(1,4001))
    assert all(np.isfinite(x['kd']) and np.isfinite(x['supervised']) and len(x['gradient_norms'])==96 for x in trace)
    cost=dict(training_seconds=train['seconds'],training_peak_allocated_bytes=max(x['cuda_peak_allocated_bytes'] for x in trace),
        training_peak_reserved_bytes=max(x['cuda_peak_reserved_bytes'] for x in trace),
        memory_definition='maximum of after_train_iter interval peak readings over all 4000 steps; MMEngine resets counters each log interval',
        student_trainable_parameters=pre['trainable_parameters'],student_total_parameters=pre['total_parameters'],
        teacher_parameters=pre['teacher_parameters'],teacher_training_only=True,student_checkpoint_excludes_teacher=train['teacher_not_in_student_checkpoint'],
        fit_student_inference_seconds=read(ROOT/'fit_representatives/index.json')['inference_seconds'],
        development_student_inference_seconds=read(ROOT/'development_val/index.json')['inference_seconds'],
        inference_definition='cohort wall time includes prepared file SHA/read, forward/postprocessing and cache SHA/write',
        student_forward_budget=dict(fit=153,development=32),deployment_networks=1,
        training_teacher_forwards=4000,student_training_backbone_forwards=4000,
        student_checkpoint_selection_development_forwards=8*32,teacher_development_training_targets=0,
        student_decode_passes_per_training_batch=2,preflight_extra_forwards_excluded_from_4000=True,
        checkpoint_bytes=next(x['bytes'] for x in train['checkpoints'] if x['path']==train['best_checkpoint']))
    write(ROOT/'cost_audit.json',cost)
    table=[]
    for n,p in report['policies'].items():
        table.append(dict(policy=n,macro_AP_percent=p['macro_ap']['mean']*100,pooled_AP_percent=p['pooled_ap']*100,
            macro_recall_percent=p['macro_recall']['mean']*100,macro_FPR_percent=p['macro_fpr']['mean']*100,
            FPR_upper95_percent=p['macro_fpr']['upper95_one_sided']*100,
            shadow_FPR_percent=p['shadow_fpr']['mean']*100,no_shadow_FPR_percent=p['no_shadow_fpr']['mean']*100,
            macro_mIoU_percent=p['macro_miou_percent']['mean'],pooled_mIoU_percent=p['miou_percent'],
            Surface_IoU_percent=p['iou_percent'][0],Cloud_IoU_percent=p['iou_percent'][1],Shadow_IoU_percent=p['iou_percent'][2],
            Surface_to_Cloud=p['confusion'][0][1],Cloud_to_Surface=p['confusion'][1][0],
            Shadow_to_Surface=p['shadow_to_surface'],Shadow_to_Cloud=p['shadow_to_cloud'],
            zero_recall_groups=len(p['zero_recall_groups']),groups_FPR_above_one_percent=p['fpr_above_one_percent_groups']))
    csvwrite(ROOT/'summary.csv',table)
    csvwrite(ROOT/'training_curve.csv',[dict(iteration=x['iteration'],**x['metrics']) for x in train['curve']])
    csvwrite(ROOT/'loss_curve.csv',[{k:v for k,v in x.items() if k!='gradient_norms'} for x in trace])
    text=['# Phase82：固定Surface／Cloud条件蒸馏','', '## 1 数据结果','',
        '一个seed65／4000步LoRA训练作业，一个D＋S浅层拟合；λ=1、T=1。155产品/153fit组训练，153fit代表拟合与阈值校准；32development代表评价，其中7组有参考云影。未读取任何确认池或ALCD。','',
        f"最佳checkpoint仅按每500步development池化mIoU选择（平手取最早）：{train['best_iteration']}步，SHA `{train['best_checkpoint_sha256']}`。全部8个定期checkpoint保留在远端新work_dirs；学生从原Source重新初始化，没有续训Phase80。",'',
        '|组合|宏AP %|池化AP %|宏召回 %|宏FPR %|FPR单侧95%上界 %|宏mIoU %|池化mIoU %|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in table:text.append('|'+x['policy']+'|'+'|'.join(f'{x[k]:.4f}' for k in ['macro_AP_percent','pooled_AP_percent','macro_recall_percent','macro_FPR_percent','FPR_upper95_percent','macro_mIoU_percent','pooled_mIoU_percent'])+'|')
    text+=['','M=MsRE＋光谱，L=原LoRA＋光谱。MM／LL使用各自非云影读出，LM是LoRA云影决策＋MsRE非云影读出的双模型参考。D＋S只运行本轮蒸馏学生，使用本轮新fit的标准化、系数与阈值。','',
           '固定配对差，单位百分点；组bootstrap 10000次、seed82011，AP／召回7组，mIoU32组，均为描述性区间：','']
    for key,c in report['comparisons'].items():
        for field in ['AP_pp','recall_pp','mIoU_pp']:
            v=c[field];text.append(f"- {key} {field}: {v['mean']:+.4f}，95%区间[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]。")
        text.append(f"- {key} 池化mIoU {c['pooled_mIoU_pp']:+.4f}；最差逐组AP {c['worst_AP_pp']:+.4f}、召回 {c['worst_recall_pp']:+.4f}、mIoU {c['worst_mIoU_pp']:+.4f}。")
    text+=['','五项固定探索晋级门：','']
    for k,v in report['continuation_gate']['checks'].items():text.append(f"- {k}: {'满足' if v else '不满足'}。")
    text+=['',f"训练耗时{cost['training_seconds']/60:.3f}分钟；学生可训练{cost['student_trainable_parameters']:,}参数，总参数{cost['student_total_parameters']:,}；教师{cost['teacher_parameters']:,}参数仅训练使用。训练逐步区间峰值最大分配显存{cost['training_peak_allocated_bytes']/1024**3:.3f}GiB。153fit／32development一次学生缓存推理总墙时分别{cost['fit_student_inference_seconds']:.2f}／{cost['development_student_inference_seconds']:.2f}秒（含文件校验与存储）。",'',
        '训练中同一预处理后的增强RGB张量同时输入教师与学生；教师eval/no_grad、严格加载且前后所有权重一致。KD只用参考Surface／Cloud像元，对两类logits单独log_softmax；忽略Shadow及255，空mask为0。原监督损失全部保留。训练每步监督/KD/96张量梯度与显存记录完整，数值有限；Source基础张量始终不变，学生checkpoint不含教师。','',
        '## 2 结论探讨','']
    text.append('五项同时满足，保留本固定λ=1、T=1方案供后续必要研究。' if report['continuation_gate']['retain_fixed_scheme'] else '至少一项固定条件失败，关闭本轮λ=1、T=1方案；不搜索权重、不重选checkpoint。这不能否定所有蒸馏方案。')
    text+=['', '这是一次训练及既有development选择下的探索筛查，不是正式非劣检验或独立确认，也不构成方法创新证明。LM仅是双模型参考而非可保证达到的上界；本轮D的云影指标允许改变，不适用Phase81的不变量。','',
        '宏／池化mIoU与逐组支持完整保留；旧union>0缺类平均实现不改写，宏改善不能掩盖单景损伤。FPR点估计≤1%不能替代单侧上界≤1%，也不表示每景≤1%。全32组分层误报率、流向、零召回和三类IoU见summary.csv与development_groups.csv；没有参考云影的组AP保持NA。','',
        '训练需要冻结教师额外前向以及学生两次decode_head读出（原监督与KD）；部署只需学生一套网络＋四输入浅层评分器。单模型部署成本与训练教师成本分开，不宣称训练免费。权重和大缓存留远端，不同步work_dirs。旧确认／失败结论保持，不追加训练或启封。']
    (ROOT/'README.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    names=['README.md','summary.csv','development_groups.csv','paired_differences.csv','confusion_matrices.json',
        'training_curve.csv','training_curve.jsonl','loss_curve.csv','loss_gradient_curve.jsonl','training_completed.json',
        'training_lock.json','preflight.json','resolved_config.py','fit_input_audit.json','calibration_lock.json',
        'development_input_audit.json','phase82_report.json','SHARED_PIXEL_VERIFIED.json','COMPLETED.json','cost_audit.json',
        'fit_representatives/index.json','development_val/index.json']
    write(ROOT/'artifact_hashes.json',{n:digest(ROOT/n) for n in names})

if __name__=='__main__':main()
