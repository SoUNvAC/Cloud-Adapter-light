"""Publish verified Phase310 outputs without local model/statistical computation."""
import shutil
from phase310_common import *
def main():
    b=load(LOCAL/'BACKUP_RECEIPT.json');assert b['all_stream_members_verified'];root=LOCAL/'remote_backup'
    for n,v in load(root/'BACKUP_FILES.json').items():assert sha(root/n)==v['sha256'],n
    S.mkdir(parents=True,exist_ok=True)
    for p in root.iterdir():
        if p.is_file() and p.suffix in ('.csv','.json','.jsonl','.md','.log','.png','.svg'):shutil.copyfile(p,S/p.name)
    shutil.copyfile(LOCAL/'BACKUP_RECEIPT.json',S/'BACKUP_RECEIPT.json')
    status=load(root/'STATUS.json');end=load(root/'ENDPOINT_LOCK.json');support=read(root/'TRAINING_SUPPORT.csv')[0]
    notes=['# Phase310 实验记录','',
        '- 目标：Q90-PL只增加未接纳Source-Surface区域的同512冻结Source软预测保持，固定lambda1/T1，检验损伤修复、转移及原Cloud纠正保留。',
        '- 改动：src/tools新增纯影像A/P复现、独立Source教师、原304接受步AST lambda0逐张量门检、一次LoRA训练、32组固定末步、27格/六向错误及P/A/其余分层和完整流式SHA备份。',
        '- 网络：本地代码push后gzs pull；无下载或升级，完整流回传逐文件及整流SHA通过，最终同步后双机复验。',
        '- 环境：gzs cloud-lite-pt210/PyTorch2.1.0+cu121、FP32单GPU；全部CPU分析/曲线在gzs，本机仅编辑/文件哈希/备份。',
        '- 止损：新增前向≤2400、GPU管线≤4小时、CPU≤16GiB、新持久含双机备份≤4GB、宿主空闲内存/磁盘≥4GB；无seed/lambda/阈值扫描，不重训或补步。',
        f"- 结果：{status['status']}，实际新增前向{status['neural_calls']}，1000尝试/{end['updates']}更新/{end['skipped']}跳过；全部Source原张量、buffer及teacher不变。",
        f"- 支持：A=75469158，三类[40622307,33554491,1292360]；P唯一产品像元{support['P_unique_product_pixels']}，正式更新裁剪累计P{support['P_cumulative_updated_crops']}。",
        f"- 备份：{b['files']}成员，流{b['stream_bytes']}字节，SHA256={b['stream_sha256']}，逐文件/整流均通过，无持久大归档；包括初末LoRA、评价/探针缓存、训练日志及全部验收。",
        '- 结论：真实指标与损伤/纠正取舍见RESULT；历史development反复暴露、单seed和参考/最终地理仿射局限保留，不把本轮称独立盲测或云特有创新；不启动下一phase。']
    (S/'phase310.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    write(S/'SUMMARY_SHA256.json',dict(status=status['status'],files={p.name:sha(p) for p in S.iterdir() if p.is_file() and p.name!='SUMMARY_SHA256.json'},code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase310*.py')}));print(status['status'])
if __name__=='__main__':main()
