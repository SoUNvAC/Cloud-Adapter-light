"""Local report-only explanation of crossed readout losses; no pixel access."""
import argparse
from pathlib import Path
from phase80_common import read,write,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
    assert read(root/'LOCAL_VERIFIED.json')['status']=='passed'
    report=read(root/'phase81_report.json');policies=report['policies']
    rows={n:{r['product']:r for r in v['rows']} for n,v in policies.items()}
    counts=[r['valid_pixels'] for r in rows['LL'].values()]
    assert len(counts)==32 and len(set(counts))==1 and counts[0]==1044484
    lines=['# Phase81：逐组损伤与汇总差异补充','', '## 1 数据结果','',
       '本文件只复算已经SHA校验的报告，不访问像元或增加任何实验。32组每组均1,044,484有效像元，总计33,423,488；本轮宏／池化差异并不是各景有效像元总数不同造成的。','',
       '固定LoRA云影决策时，LM−LL有2组mIoU上升、8组下降、22组完全相同。以下完整列出所有非零差：','',
       '|产品|ΔmIoU百分点|LL有效IoU类数|LM有效IoU类数|ΔSurface→Cloud|ΔCloud→Surface|',
       '|---|---:|---:|---:|---:|---:|']
    for product,x in rows['LM'].items():
        y=rows['LL'][product];d=x['miou_percent']-y['miou_percent']
        if d:
            lines.append(f"|{product}|{d:+.4f}|{y['defined_iou_classes']}|{x['defined_iou_classes']}|{x['surface_to_cloud']-y['surface_to_cloud']:+d}|{x['cloud_to_surface']-y['cloud_to_surface']:+d}|")
    lines+=['', '相对原MsRE＋光谱MM，LM有11组mIoU改善、2组下降、19组相同；两下降组为T38TNN −1.4597百分点、T02LKH −0.2120百分点。完整原始值在paired_differences.csv中。','',
       'LM−LL池化Surface→Cloud少1,116,735像元，Cloud→Surface多242,703像元；净少874,032个Surface／Cloud互相误分。云影二值决策完全相同，因此这些变化只来自同一预测非云影区域的读出。漏影分流也改变：Shadow→Surface多35,566，Shadow→Cloud少35,566，总漏影量不变。','',
       '反向交叉ML−MM池化mIoU −3.2820百分点，与LM−LL +3.2651的方向相反。主要严重损伤位于T25XEG及T40XDR，完整表未排除任何景。','',
       '## 2 结论探讨','',
       '池化mIoU改善主要伴随Surface／Cloud读出的混淆结构变化。T25XEG和T40XDR的改善与若干其他景下降同时存在；这一固定输出交换不能证明模型内部的机制，也没有挑景切换读出。','',
       'T20WNV参考全部是Cloud（1,044,484像元）。LL全对，Surface与Shadow的union均为0，旧实现只平均Cloud IoU，mIoU=100%。LM把254像元判为Surface，使Surface IoU=0开始参与平均，Cloud IoU仍99.9757%，mIoU变为49.9878%。因此−50.0122百分点主要是旧指标在类别出现时的分母跳变，不能说一半像元突然错了；254个误分仍是真实损失，旧指标未改写。','',
       'T32NKF则是三类均有参考且两方案均平均三类，LM−LL mIoU下降12.0171百分点并非缺类分母跳变。其Cloud→Surface从56,997增到205,548，Surface→Cloud从53,587降到4,497；必须作为真实读出取舍保留。','',
       '即使每景有效像元总数相同，逐景类别组成、每类union和缺类定义仍不同。先计算逐景IoU再等权平均，与先汇总各类混淆再算IoU不等价；这些非线性与类别支持共同解释宏／池化方向可以不同。没有事后换指标重新判成功。','',
       'FPR上界按本轮协议seed81011为1.3353%，与Phase80 seed80011的1.3304%存在bootstrap重抽样差异；点估计和逐组误报率精确相同，不能把区间小变动当模型变化。预算稳定满足仍未获支持。','',
       '本轮到此收口：固定交叉读出消除这批development的池化mIoU差距，同时保留逐组损伤、两模型部署代价与独立验证缺口；不追加模型、特征、阈值或确认评价。']
    path=root/'READOUT_INTERPRETATION.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Separate local reporting supplement; preserve original shared delivery/hash manifest.
    write(root/'local_supplement_hashes.json',dict(source_report_sha256=digest(root/'phase81_report.json'),
        local_supplement_sha256=digest(path),shared_artifacts_unchanged=True,report_only=True))
    print('Phase81 report-only supplement written')

if __name__=='__main__':main()
