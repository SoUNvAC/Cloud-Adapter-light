"""Publish SHA-verified Phase306 descriptive evidence and actual execution record."""
import shutil
from phase306_common import *

def main():
    receipt=load(LOCAL/'BACKUP_RECEIPT.json');assert receipt['all_members_SHA_verified']
    root=LOCAL/'remote_backup'
    for n,v in load(root/'BACKUP_FILES.json').items():assert sha(root/n)==v['sha256'],n
    S.mkdir(parents=True,exist_ok=True)
    for p in root.iterdir():
        if p.is_file() and p.suffix in ('.csv','.json','.md','.log'):shutil.copyfile(p,S/p.name)
    shutil.copyfile(LOCAL/'BACKUP_RECEIPT.json',S/'BACKUP_RECEIPT.json')
    resource=load(root/'RESOURCE.json');status=resource['status']
    notes=['# Phase306 实验记录','',
        '- 目标：复核非G Surface→Cloud损伤的场景集中度，并在同场景、相近Source q内检查六个既有原始TOA波段的分布和D/K排序线索。',
        '- 改动：src/tools新增固定事件/分层/支持、六通道位置抽查、精确分位数、并列分数AUC与独立秩和验收、资源护栏及全量SHA备份；Phase302/304/305只读。',
        '- 网络：本地修改代码后git push，gzs指定仓库pull执行；全部同步及回传以实际收据/SHA复验为准。',
        '- 环境：gzs cloud-lite-pt210独立CPU；训练/分类器拟合/模型前向/下载均0，本机仅编辑、文件哈希与备份。',
        '- 止损线：CPU≤16GiB、运行≤1小时、新持久含备份≤1GB、宿主空闲内存/磁盘≥4GB；输入缺失交缺项，不补推理、配准搜索或新波段。',
        f"- 实际状态：{status}，耗时{resource['seconds']:.3f}秒，峰值CPU{resource['peak_CPU_bytes']}字节，最低宿主空闲{resource['minimum_available_bytes']}字节。",
        f"- 备份：{receipt['files']}成员，归档{receipt['bytes']}字节，SHA256={receipt['sha256']}，逐成员SHA全部通过；含32产品原始TOA事件取值及索引/共同支持NPZ。"]
    if status=='completed':
        t=read(S/'EVENTS_POOLED.csv')[0];top=read(S/'TOP_1_3_5_CONCENTRATION.csv');snow=next(r for r in read(S/'SCENE_EVENT_STRATA.csv') if r['field']=='snow/ice' and r['value']=='1')
        notes += [f"- 原事件复现：原正确Surface损伤506798=G内56740+非G450058；R中D/U/K/O={t['D']}/{t['U']}/{t['K']}/{t['O']}；非G正确Surface支持2812248，tiny-D={t['tiny_D']}。",
            f"- 场景集中度：snow/ice=1的7场景全损伤375385、非G366352；非G D排序前1/3/5组贡献="+'/'.join(f"{float(r['D_pool_share_pct']):.3f}%" for r in top)+f"；雪冰标签层同时有D/K为{snow['both_groups']}组。",
            '- 口径：six-band交集及原事件数分开，六波段共支持；固定原始TOA及q三层，Cloud正方向，AUC不翻转，单类NA。全部32组、零支持行和11字段0/1/缺失保留。',
            '- 限制：场景标签不是像元地物/物理原因；池化不能替代组内，零拟合不能证明B11/B12对RGB有独立增量。历史开发反复使用、单seed/预训练独立性及cube原生重采样溯源限制均保留，不启动下一phase。']
    (S/'phase306.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    write(S/'SUMMARY_SHA256.json',dict(status=status,files={p.name:sha(p) for p in S.iterdir() if p.is_file() and p.name!='SUMMARY_SHA256.json'},
        code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase306*.py')}))
    print(status)

if __name__=='__main__':main()
