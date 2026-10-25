"""Publish verified files and model/scaler JSON; file operations only on local host."""
import shutil
from phase312_common import *

def main():
    b=load(LOCAL/'BACKUP_RECEIPT.json');assert b['all_archive_members_verified'] and b['archive_SHA_verified_before_extraction']
    root=LOCAL/'remote_backup'
    for n,v in load(root/'BACKUP_FILES.json').items():assert sha(root/n)==v['sha256'],n
    assert sha(LOCAL/b['archive'])==b['archive_sha256']
    S.mkdir(parents=True,exist_ok=True);(S/'models').mkdir(exist_ok=True)
    for p in root.iterdir():
        if p.is_file() and p.suffix in ('.csv','.json','.jsonl','.md','.log'):shutil.copyfile(p,S/p.name)
    for p in (root/'models').glob('*.json'):shutil.copyfile(p,S/'models'/p.name)
    shutil.copyfile(LOCAL/'BACKUP_RECEIPT.json',S/'BACKUP_RECEIPT.json')
    state=load(root/'STATUS.json');fit=load(root/'FIT_LOCK.json')
    notes=['# Phase312 实验记录','',
        '- 目标：纯Source伪标签训练RGB3/RGB6Q/MS6固定光谱选择器，核对额外波段的同参数及等数量增量，完整报告Surface修复与Cloud撤销。',
        '- 改动：100 fit影像、PCG64(31201)每产品≤8192等产品权重；3次零初始固定L-BFGS-B逻辑回归，训练scaler；32组概率与0.90/等数量掩膜全部封存后独立真值评价。',
        '- 环境：所有拟合、预测及CPU分析均在gzs cloud-lite-pt210；本机仅编辑、文件哈希和完整备份。神经训练/前向/下载均0，无CV/重启/阈值搜索。',
        '- 网络：本地commit/push、gzs指定仓库pull；无下载/升级。首次全量归档回传，整包SHA先验、再全部成员SHA。',
        '- 止损：CPU总执行≤2小时、BLAS≤4线程、内存≤16GiB、新持久含双机原始结果与归档/摘要≤2GB、宿主空闲内存及磁盘≥4GB。不收敛模型保留参数/梯度，其他模型继续，不追加拟合。',
        f"- 结果：管线{state['status']}，科学状态{state['scientific_status']}；实际拟合{fit['fit_attempts']}次，成功模型{fit['successful_models']}，32组/固定D506798与B784478核验；实际指标、D/K/Z与AUC见RESULT.md。",
        f"- 备份：{b['files']}成员，原始{b['raw_bytes']}字节、归档{b['archive_bytes']}字节，SHA256={b['archive_sha256']}，全部SHA通过；采样索引/组权、模型、概率/掩膜与访问日志均完整本地保留。",
        '- 局限：Source伪标签可能复制亮地表错误；单类组AUC为NA；原B与P天然不交，全保留不是学得的新增收益；scene snow/ice非像元真值，历史development反复暴露。无定时检查或自动下一phase。']
    (S/'phase312.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    write(S/'SUMMARY_SHA256.json',dict(status=state['status'],scientific_status=state['scientific_status'],files={p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file() and p.name!='SUMMARY_SHA256.json'},code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase312*.py')}))
    print(state['status'],state['scientific_status'])
if __name__=='__main__':main()
