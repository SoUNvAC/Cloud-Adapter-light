"""Publish actual SHA-verified Phase304 findings, ledgers and concise experiment record."""
from pathlib import Path
import shutil
import json
from phase304_common import R,sha,load,write,read
from transfer_phase304 import LOCAL

S=R/'src/research_results/phase304_20261024'

def main():
    receipt=load(LOCAL/'BACKUP_RECEIPT.json');assert receipt['all_stream_members_verified']
    root=LOCAL/'remote_backup'
    for n,v in load(root/'BACKUP_FILES.json').items():assert sha(root/n)==v['sha256'],n
    S.mkdir(parents=True,exist_ok=True)
    for folder in (root,root/'analysis'):
        if folder.exists():
            for p in folder.iterdir():
                if p.is_file() and p.suffix in ('.json','.csv','.log','.jsonl','.md'):shutil.copyfile(p,S/p.name)
    shutil.copyfile(LOCAL/'BACKUP_RECEIPT.json',S/'BACKUP_RECEIPT.json')
    status=load(root/'STATUS.json');stages=load(root/'STAGES.json')
    calls=sum(1 for _ in (root/'CALLS.jsonl').open()) if (root/'CALLS.jsonl').exists() else 0
    notes=['# Phase304 实验记录','',
        '- 目标：固定Q90-PL，检验充分三类自训练支持下Cloud→Shadow和Surface→Cloud的方向取舍；不增加机制、seed或阈值。',
        '- 改动：本地新增准备/训练/评价/独立CPU审计、资源护栏、全量流式SHA备份和报告复核脚本；原Phase302/303只读。',
        '- 网络：本地代码git push后gzs pull；全部实验在cloud-lite-pt210。实际网络事件见OPERATIONAL_LEDGER.json（若有）；不得假定同步成功。',
        '- 止损线：全新增前向≤3000、GPU管线≤4小时、CPU≤16GiB、新持久含备份≤6GB、宿主空闲内存/磁盘≥4GB；冲突时等待，失败保留不自动重跑。',
        '- 计算位置：gzs；本机仅代码编辑、文件哈希与备份。教师/训练仅影像；末步封存后独立CPU审计才读参考。',
        f"- 状态：{status['status']}；实际前向{calls}；逐阶段退出码/时间/峰值内存见STAGES.json与资源日志。",
        f"- 备份：{receipt['files']}成员，流{receipt['stream_bytes']}字节，SHA256={receipt['stream_sha256']}，全部文件逐项SHA核验；无持久归档副本。"]
    if (root/'ENDPOINT_LOCK.json').exists():
        end=load(root/'ENDPOINT_LOCK.json');notes.append(f"- 训练：{end['attempt']}尝试、{end['updates']}更新、{end['skipped']}跳过；Source全部原张量不变={end['all_base_tensors_unchanged']}，末步SHA={end['final_sha256']}。")
    if status['status']=='completed':
        results=read(S/'FIXED_RESULTS.csv');d=load(S/'RESULT_DIFFERENCES_PP.json')
        audit=load(S/'PSEUDO_AUDIT_ACCEPTANCE.json');assert audit['fit_products']==100
        notes.append(f"- 评价：32组；Source/Q90组均mIoU={float(results[0]['group_mean_mIoU_pct']):.6f}/{float(results[1]['group_mean_mIoU_pct']):.6f}%，差{d['group_mean_mIoU_pp']:+.6f}pp；完整真实结果见RESULT.md及各CSV。")
    (S/'phase304.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    write(S/'DELIVERY_STATUS.json',dict(status=status['status'],actual_calls=calls,complete_backup=True,
        backup_stream_sha256=receipt['stream_sha256'],raw_bytes=receipt['raw_bytes'],archive_persisted=False))
    write(S/'SUMMARY_SHA256.json',dict(files={p.name:sha(p) for p in S.iterdir() if p.is_file() and p.name!='SUMMARY_SHA256.json'},
        code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase304*.py')}))
    print(json.dumps(load(S/'DELIVERY_STATUS.json')))

if __name__=='__main__':main()
