"""Publish SHA-verified descriptions and fixed alignment panels, without local analysis."""
import shutil
from phase308_common import *
def main():
    receipt=load(LOCAL/'BACKUP_RECEIPT.json');assert receipt['all_members_SHA_verified']
    root=LOCAL/'remote_backup'
    for n,v in load(root/'BACKUP_FILES.json').items():assert sha(root/n)==v['sha256']
    S.mkdir(parents=True,exist_ok=True)
    for p in root.iterdir():
        if p.is_file() and p.suffix in ('.csv','.json','.md','.log','.jsonl'):shutil.copyfile(p,S/p.name)
    shutil.copytree(root/'previews',S/'previews',dirs_exist_ok=True)
    shutil.copyfile(LOCAL/'BACKUP_RECEIPT.json',S/'BACKUP_RECEIPT.json')
    resource=load(root/'RESOURCE.json');accept=load(root/'CPU_ACCEPTANCE.json')
    notes=['# Phase308 实验记录','',
        '- 目标：复现Phase307全部32折既有探针并诊断最大损伤组，固定历史前三组作训练支持、概率贡献和图像/参考有限核对。',
        '- 改动：src/tools新增原采样/权重/输入身份锁恢复、固定系数概率复算、条件组权分位数、双类范围/重叠、有符号贡献和固定坐标预览；历史304—307只读。',
        '- 网络：本地代码commit/push后gzs指定仓库pull；完整归档回传逐文件SHA核验，交付后再双机复验。下载0；任何实际传输异常单独保留。',
        '- 环境：计算/渲染全部在gzs cloud-lite-pt210独立CPU、BLAS4线程；本机仅编辑、文件哈希与完整备份。',
        '- 止损线：拟合/训练/神经前向/下载0；CPU≤16GiB/1小时；新持久含备份≤1GB，宿主空闲内存/磁盘≥4GB；失败保留，不调阈值/环境/容差或重试训练。',
        f"- 资源：状态{resource['status']}，耗时{resource['seconds']:.3f}秒，峰值CPU{resource['peak_CPU_bytes']}字节，最低宿主空闲{resource['minimum_available_bytes']}字节。",
        f"- 结果：32折×3模型共96行完整复现，概率最大误差{accept['probability_max_error']}；原修复/撤销/Brier/AUC一致，最大纯D组三模型仍修复0。",
        '- 核查缺口：主参考身份一致不等于真值正确；README固定运行时无现成解析器，内容未解析且SHA保留；alt_masks编码/标注独立性未确认，unknown不替换。',
        f"- 备份：{receipt['files']}成员，归档{receipt['bytes']}字节，SHA256={receipt['sha256']}；全成员SHA通过，包含全部描述统计、锁、资源/失败记录及12张固定预览。",
        '- 结论边界：有限多光谱增量仍不能解决最大损伤；原始标注语义未决，单变量支持不证明联合覆盖或物理因果；本轮结束，不启动下一phase。']
    (S/'phase308.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    write(S/'SUMMARY_SHA256.json',dict(status=resource['status'],files={p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file() and p.name!='SUMMARY_SHA256.json'},code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase308*.py')}))
    print(resource['status'])
if __name__=='__main__':main()
