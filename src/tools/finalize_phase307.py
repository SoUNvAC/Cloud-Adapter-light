"""Publish verified supervised offline LOGO information-probe findings."""
import shutil
from phase307_common import *

def main():
    receipt=load(LOCAL/'BACKUP_RECEIPT.json');assert receipt['all_members_SHA_verified']
    root=LOCAL/'remote_backup'
    for n,v in load(root/'BACKUP_FILES.json').items():assert sha(root/n)==v['sha256'],n
    S.mkdir(parents=True,exist_ok=True)
    for p in root.iterdir():
        if p.is_file() and p.suffix in ('.csv','.json','.md','.log','.jsonl'):shutil.copyfile(p,S/p.name)
    shutil.copyfile(LOCAL/'BACKUP_RECEIPT.json',S/'BACKUP_RECEIPT.json')
    resource=load(root/'RESOURCE.json');status=resource['status']
    notes=['# Phase307 实验记录','',
        '- 目标：固定按组留出的RGB3、等参数RGB6Q与MS6比较，诊断额外观测是否支持跨组D/K区分；这是目标参考标签监督式离线信息诊断，非SFDA性能。',
        '- 改动：src/tools新增统一每组随机采样锁、32折互斥训练/权重/标准化、固定逻辑回归、逐像元折外概率、三项成对比较、固定分层、独立数值/泄漏验收、资源护栏和全量SHA备份。历史302—306只读。',
        '- 网络：本地代码git push后gzs指定仓库pull；同步及完整回传以收据和双机SHA复验为准。',
        '- 环境：gzs cloud-lite-pt210独立CPU，BLAS≤4线程；分割训练/神经前向/下载0，本机仅编辑、文件哈希及备份。',
        '- 止损线：浅层拟合≤96次、CPU≤16GiB/1小时、新持久含备份≤1GB、宿主空闲内存/磁盘≥4GB；不收敛留痕、不重试/调参/删组。',
        f"- 实际状态：{status}，拟合{resource['classifier_fits']}次、耗时{resource['seconds']:.3f}秒、峰值CPU{resource['peak_CPU_bytes']}字节、最低宿主空闲{resource['minimum_available_bytes']}字节。",
        f"- 备份：{receipt['files']}成员，归档{receipt['bytes']}字节，SHA256={receipt['sha256']}，逐成员SHA全部通过；包括每折系数/标准化、固定采样ID和完整折外概率。"]
    if status=='completed':
        accept=load(root/'CPU_ACCEPTANCE.json');pair=read(S/'PAIRED_COMPARISONS.csv')[0]
        notes += [f"- 拟合/覆盖：32折×3模型完整交表；拟合失败/不可识别{accept['fit_failures']}，全部C折外覆盖={accept['all_C_coverage']}。单类AUC为NA，空测试不拟合但保留，Brier与0.5读出保留纯D最大损伤组。",
            f"- 主比较MS6−RGB6Q：双方成功{pair['both_success_C_groups']}组、双类AUC{pair['AUC_groups']}组，覆盖D/K={pair['D_coverage']}/{pair['K_coverage']}；组均AUC差{pair['group_mean_AUC_delta']}，Brier差{pair['group_mean_Brier_delta']}，修复D/撤销K增量={pair['repaired_D_delta']}/{pair['revoked_K_delta']}。",
            '- 限制：概率来自未含本测试组的固定训练折，仍是使用目标标签的监督探针；历史开发暴露、单seed、RGB6Q函数类限制与原始cube溯源限制保留，不部署、不回写Phase304、不启动下一phase。']
    (S/'phase307.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    write(S/'SUMMARY_SHA256.json',dict(status=status,files={p.name:sha(p) for p in S.iterdir() if p.is_file() and p.name!='SUMMARY_SHA256.json'},
        code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase307*.py')}))
    print(status)

if __name__=='__main__':main()
