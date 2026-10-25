"""Publish only verified Phase305 artifacts and actual experiment record."""
import shutil
from phase305_common import *

def main():
    receipt=load(LOCAL/'BACKUP_RECEIPT.json');assert receipt['all_members_SHA_verified']
    root=LOCAL/'remote_backup'
    for n,v in load(root/'BACKUP_FILES.json').items():assert sha(root/n)==v['sha256'],n
    S.mkdir(parents=True,exist_ok=True)
    for p in root.iterdir():
        if p.is_file() and p.suffix in ('.csv','.json','.md','.log'):shutil.copyfile(p,S/p.name)
    shutil.copyfile(LOCAL/'BACKUP_RECEIPT.json',S/'BACKUP_RECEIPT.json')
    resource=load(root/'RESOURCE.json');status=resource['status']
    note=['# Phase305 实验记录','',
        '- 目标：定位高置信原正确Surface损伤，评估一个固定Source优先读出的保护空间和必要纠错代价。',
        '- 改动：src/tools新增流式CPU统计、固定G90读出、全部32组/参考边界/11场景字段分层、独立计数验收、资源护栏和全量SHA回传脚本。Phase302/304原文件只读。',
        '- 网络：本地修改并push后gzs pull；准入阶段两次SSH kex断连，恢复后执行；同步和回传以真实收据为准。',
        '- 环境：gzs cloud-lite-pt210独立CPU，零训练、零模型前向、零下载；本机仅编辑、文件哈希及备份。',
        '- 止损线：CPU≤16GiB、耗时≤1小时、新持久含备份≤1GB、宿主空闲内存/磁盘≥4GB；缺缓存报缺项，不补神经推理。',
        f"- 实际状态：{status}，{resource['seconds']:.3f}秒，峰值CPU{resource['peak_CPU_bytes']}字节，最低宿主空闲{resource['minimum_available_bytes']}字节，前向0/更新0。",
        f"- 备份：{receipt['files']}成员，归档{receipt['bytes']}字节，SHA256={receipt['sha256']}，全量逐项SHA通过。"]
    if status=='completed':
        a=read(S/'G_RELIABILITY.csv')[0];changes=read(S/'READOUT_CHANGES.csv')[0];metrics=read(S/'FIXED_READOUT_RESULTS.csv')
        note += [f"- 损伤：506798原正确Surface损伤中G内{a['D_correct_G']}（{float(a['damage_concentration_in_G_pct']):.3f}%）；G支持{a['G']}，H支持{a['H']}，H损伤率{float(a['G_correct_Surface_damage_pct']):.3f}%。另19553为原错误Shadow转Cloud，不混入该分母。",
            f"- 固定读出：修复真Surface{changes['repaired_true_Surface']}、撤销真Cloud纠正{changes['revoked_true_Cloud_correction']}、真Shadow Cloud→Surface变化{changes['true_Shadow_Cloud_to_Surface']}；组均mIoU Q90/G90={float(metrics[1]['group_mean_mIoU_pct']):.6f}/{float(metrics[2]['group_mean_mIoU_pct']):.6f}%。Shadow决策和原连续评分不变。",
            '- 解释限制：G不用真值，H只离线审计；场景标签不是像元地物，参考边界不是物理云边界。历史开发暴露、单seed和预训练交集未知均保留；此读出未部署，不证明训练margin可达到同效，不启动下一phase。']
    (S/'phase305.md').write_text('\n'.join(note)+'\n',encoding='utf-8')
    write(S/'SUMMARY_SHA256.json',dict(status=status,files={p.name:sha(p) for p in S.iterdir() if p.is_file() and p.name!='SUMMARY_SHA256.json'},
        code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase305*.py')}))
    print(status)

if __name__=='__main__':main()
