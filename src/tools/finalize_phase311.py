"""Publish verified files only; no local scientific computation."""
import shutil
from phase311_common import *

def main():
    b=load(LOCAL/'BACKUP_RECEIPT.json');assert b['all_archive_members_verified'] and b['archive_SHA_verified_before_extraction']
    root=LOCAL/'remote_backup'
    for n,v in load(root/'BACKUP_FILES.json').items():assert sha(root/n)==v['sha256'],n
    assert sha(LOCAL/b['archive'])==b['archive_sha256']
    S.mkdir(parents=True,exist_ok=True)
    for p in root.iterdir():
        if p.is_file() and p.suffix in ('.csv','.json','.jsonl','.md','.log','.png','.svg'):shutil.copyfile(p,S/p.name)
    shutil.copyfile(LOCAL/'BACKUP_RECEIPT.json',S/'BACKUP_RECEIPT.json')
    state=load(root/'STATUS.json');end=load(root/'ENDPOINT_LOCK.json')
    notes=['# Phase311 实验记录','',
        '- 目标：Q90-CM单向Cloud−Surface得分差约束，相对PL/SP检验原损伤修复、Cloud纠正保留与Shadow误报代价。',
        '- 改动：全新30201 LoRA；A/P只读复用310并逐掩膜SHA复现；独立Source同512，mean_A CE + mean_P ReLU(mS−mT)²，lambda1；合成梯度门检与原304 AST lambda0门检。',
        '- 环境：gzs cloud-lite-pt210/PyTorch2.1.0+cu121，所有模型、CPU分析和曲线在gzs，本机仅编辑/文件哈希/备份。',
        '- 网络：本地commit/push后gzs pull；无下载/升级；归档先整包SHA、再全成员SHA；最终代码/结果双机复验。',
        '- 止损：前向≤2400、GPU管线≤4h、CPU≤16GiB、含双机原始结果与双份归档/摘要的新持久≤4GB、宿主空闲内存/磁盘≥4GB；不重训、不扫lambda/seed/阈值。',
        f"- 结果：{state['status']}；1000尝试/{end['updates']}更新/{end['skipped']}跳过，新增前向{state['neural_calls']}；Source与teacher张量/buffer不变。真实四策略指标、完整27格、固定D/B及P误保护见RESULT.md与CSV。",
        f"- 备份：{b['files']}成员；原始文件{b['raw_bytes']}字节、归档{b['archive_bytes']}字节；SHA256={b['archive_sha256']}。全量初末LoRA/缓存/日志/输入代码锁均本地保留。",
        '- 局限：单seed、历史development反复暴露；CM/KL同lambda不等强度，Shadow输出直接偏导0不排除共享参数影响；雪冰参考和最终逐景仿射未决。无自动下一phase。']
    (S/'phase311.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    write(S/'SUMMARY_SHA256.json',dict(status=state['status'],files={p.name:sha(p) for p in S.iterdir() if p.is_file() and p.name!='SUMMARY_SHA256.json'},code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase311*.py')}))
    print(state['status'])
if __name__=='__main__':main()
