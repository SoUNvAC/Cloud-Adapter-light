"""Publish only backed-up measurements; keep pending local analysis explicit."""
import json
from pathlib import Path
import shutil
from transfer_phase302 import R,load,sha,write

ROOT=R/'outputs/phase302/phase302_20261020'
D=ROOT/'remote_backup'
S=R/'src/research_results/phase302_20261020'


def main():
    receipt=load(ROOT/'BACKUP_RECEIPT.json');assert receipt['all_archive_members_verified']
    manifest=load(D/'BACKUP_MANIFEST.json')
    for n,v in manifest.items():assert sha(D/n)==v['sha256'],n
    assert (D/'EXIT_CODE').read_text().strip()=='0'
    endpoint=load(D/'ENDPOINT_LOCK.json');model=load(D/'MODEL_ACCEPTANCE.json')
    assert sha(D/'final_lora.pth')==endpoint['final_sha256']
    train=load(D/'train_RECEIPT.json');evaluation=load(D/'evaluate_RECEIPT.json')
    calls=[json.loads(s) for s in (D/'CALLS.jsonl').read_text().splitlines()]
    kinds={k:sum(r['kind']==k for r in calls) for k in sorted({r['kind'] for r in calls})}
    assert len(calls)<=4000
    S.mkdir(parents=True,exist_ok=True)
    names=('SOURCE_ADMISSION.json','MODEL_ACCEPTANCE.json','PREPROCESS_ACCEPTANCE.json','ENDPOINT_LOCK.json',
        'EXECUTION_LOCK.json','EVALUATION_LOCK.json','train_RECEIPT.json','evaluate_RECEIPT.json','PSEUDO_LOCK.json',
        'PROTOCOL_LOCK.json','CONTROLLER.json','CALLS.jsonl','STAGE_LEDGER.jsonl')
    for name in names:shutil.copyfile(D/name,S/name)
    for name in ('BACKUP_RECEIPT.json','LOCAL_ANALYSIS_WAIT.json','LOCAL_ANALYSIS_RESOURCE.json','CPU_ANALYSIS_RESOURCE.json','CPU_BACKUP_RECEIPT.json'):
        if (ROOT/name).exists():shutil.copyfile(ROOT/name,S/name)
    complete=(ROOT/'analysis/CPU_ACCEPTANCE.json').exists()
    if complete:
        assert load(ROOT/'analysis/CPU_ACCEPTANCE.json')['status']=='passed'
        cpu_resource=load(ROOT/('CPU_ANALYSIS_RESOURCE.json' if (ROOT/'CPU_ANALYSIS_RESOURCE.json').exists() else 'LOCAL_ANALYSIS_RESOURCE.json'))
        assert cpu_resource['status']=='completed'
        for p in (ROOT/'analysis').iterdir():
            if p.is_file():shutil.copyfile(p,S/p.name)
    pseudo=load(D/'PSEUDO_LOCK.json')['rows'];accepted=sum(r['accepted'] for r in pseudo);valid=sum(r['valid'] for r in pseudo)
    metrics=dict(status='completed' if complete else 'gpu_completed_local_cpu_waiting',
        calls=len(calls),call_kinds=kinds,updates=endpoint['updates'],skipped=endpoint['skipped'],
        accepted_pseudo_pixels=accepted,total_valid_fit_pixels=valid,coverage_fraction=accepted/valid,
        gpu_pipeline_seconds=load(D/'CONTROLLER.json')['completed_seconds'],
        peak_gpu_allocated=max(train['gpu_peak_allocated'],evaluation['gpu_peak_allocated']),
        peak_gpu_reserved=max(train['gpu_peak_reserved'],evaluation['gpu_peak_reserved']),
        peak_cpu_bytes=max(train['cpu_peak_bytes'],evaluation['cpu_peak_bytes']),
        initial_lora_sha256=endpoint['initial_sha256'],final_lora_sha256=endpoint['final_sha256'],
        frozen_base_identical=endpoint['all_base_tensors_unchanged'],backup_sha256=receipt['sha256'])
    write(S/'EXECUTION_SUMMARY.json',metrics)
    lines=['# Phase302 实验记录','',
        '- 目标：固定DINOv2-S＋Cloud-Adapter Source，检验LoRA固定伪标签适配是否同时减少Cloud→Shadow并增加Surface→Cloud，以及同512视野的实际纠错/损伤。',
        '- 改动：src/tools下新增Source身份/空间准入、纯RGB影像加载与访问护栏、FP32固定9窗语义得分拼接、零LoRA/梯度/冻结验收、1000尝试单次训练、末步封存评价、全量备份、独立CPU统计及本地内存准入脚本。未使用L8、PixBox、源样本或新增下载。',
        '- 网络：本地修改后git push、gzs指定仓库git pull；已明确获得研究结果/哈希/日志上传现有GitHub仓库的授权；SCP回传全部本轮产物。Phase301此前未推送提交已同步并远程验收。',
        '- 环境：按计划使用cloud-lite-pt210（PyTorch2.1+cu121，MMCV2.1，MMseg1.2.2），未切换Omni runtime。Source权重SHA与历史模型配置、PNG映射及空间审计匹配；132个本轮产品无已知源footprint重叠，DINO预训练交集未知，历史开发暴露保留。',
        '- 止损线：网络前向≤4000、GPU阶段≤4小时、GPU≤20GiB、CPU≤16GiB、新持久文件≤6GB、宿主可用内存/磁盘≥4GB；每次前向及每2秒监控。固定阈值0.90，不因覆盖稀疏补步或降阈值。',
        f"- 实际执行：{len(calls)}次前向；GPU管线{metrics['gpu_pipeline_seconds']:.2f}秒；GPU峰值allocated/reserved={metrics['peak_gpu_allocated']}/{metrics['peak_gpu_reserved']}字节，CPU峰值{metrics['peak_cpu_bytes']}字节。1000尝试，{endpoint['updates']}更新、{endpoint['skipped']}跳过。伪标签接受{accepted}/{valid}像元（{100*accepted/valid:.6f}%）；初末基础参数/buffer逐张量不变。LoRA参数{model['trainable_parameters']}，零初始化/关闭LoRA复现Source及可微读出/梯度验收通过。",
        f"- 封存与备份：末步SHA={endpoint['final_sha256']}；{receipt['archived_files']}个归档成员、{receipt['bytes']}字节压缩包，SHA256={receipt['sha256']}。全部新结果（输入锁、初末增量、教师缓存、32组双策略得分、16探针、日志及代码）位于outputs/phase302/phase302_20261020/remote_backup，已逐文件SHA验收。",
        '- 随机性限制：CUDA日志记录了部分算子的非确定性警告，固定seed不等于位级确定性；不抹去警告，不据此追加seed。']
    if complete:
        lines += [f"- CPU执行变更：用户因本机硬件/内存不足，明确同意改在gzs宿主机执行独立CPU统计；此前本地等待记录作为历史保留。实际位置={cpu_resource['execution']}，耗时{cpu_resource['seconds']:.3f}秒、峰值工作集{cpu_resource['peak_working_set_bytes']}字节、最低可用内存{cpu_resource['minimum_available_physical_bytes']}字节；零新增模型前向。",
            '- 结果：独立CPU验收通过；全部评价指标、27格错误流、边界与逐组支持及四探针全部时点见RESULT.md与CSV。统计新增文件另包回传并逐文件验SHA，见CPU_BACKUP_RECEIPT.json。最终文件由SUMMARY_SHA256.json锁定。定时检查完成后暂停，不启动下一phase。']
    else:
        wait=load(ROOT/'LOCAL_ANALYSIS_WAIT.json')
        lines+=[f"- 当前结果状态：远程训练和固定评价已完成，等待本地CPU统计。本地最近可用物理内存{wait['available_physical_bytes']}字节，未满足4GB最低空闲要求；另为预计2.5GB统计工作集预留空间（估算不是测量）。用户明确要求保留本地分析，未在远程连接真值。当前不报告mIoU、AP或错误流结论。",
            '- 定时检查：每5分钟观察本地可用内存；满足预计工作集加4GB空闲后才启动本地CPU统计，运行时继续检查最低空闲。检查状态不变时保持安静，不重训、不扩预算。']
    (S/'phase302.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase302*.py') if p.is_file()}
    write(S/'SUMMARY_SHA256.json',dict(status=metrics['status'],code=code,
        files={p.name:sha(p) for p in S.iterdir() if p.is_file() and p.name!='SUMMARY_SHA256.json'}))
    print(json.dumps(metrics))


if __name__=='__main__':main()
