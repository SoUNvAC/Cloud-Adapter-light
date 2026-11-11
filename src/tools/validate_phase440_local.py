"""Independent stdlib acceptance and compact numerical readout."""
from pathlib import Path
import csv,json,math
R=Path(__file__).resolve().parents[2]/'outputs/phase440/phase440_20261103'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def metrics(cm):
    support=[sum(row) for row in cm];pred=[sum(row[i] for row in cm) for i in range(3)]
    return [100*cm[i][i]/(support[i]+pred[i]-cm[i][i]) if support[i]+pred[i]-cm[i][i] else None for i in range(3)]
def main():
    lines=['# Phase440：结果解读',''];facts={};checks=0
    for weight in ['source','adapted']:
        summary=js(R/weight/'SUMMARY.json');data=[js(p) for p in sorted((R/weight/'paired').glob('*.json'))]
        assert len(data)==32
        assert js(R/weight/'IDENTITY_CHECK.json')['images']==32
        assert js(R/weight/'COMPLETE.json')['forward_calls']==1152
        for mode,values in summary.items():
            for j,v in enumerate(values):
                cm=[[0]*3 for _ in range(3)]
                for d in data:
                    e=d['records'][mode]['entries'][j];tr=e['transition']
                    assert [[sum(tr[y][a][b] for a in range(3)) for b in range(3)] for y in range(3)]==e['metrics']['confusion']
                    for a in range(3):
                        for b in range(3):cm[a][b]+=e['metrics']['confusion'][a][b]
                    assert e['appearance']['newly_zero_pixels']==0
                assert cm==v['confusion']
                ious=metrics(cm);assert max(abs(a-b) for a,b in zip(ious,v['IoU']))<1e-10
                assert abs(sum(ious)/3-v['mIoU'])<1e-10
                checks+=1
        values=summary['baseline'];dense=values[1:]
        drops=[dict(start=a['ratio'],end=b['ratio'],drop=a['mIoU']-b['mIoU']) for a,b in zip(dense,dense[1:])]
        largest=max(drops,key=lambda x:x['drop']);total=dense[0]['mIoU']-dense[-1]['mIoU']
        top=sorted([(d['product'],d['cohort_counts']['flippers']) for d in data],key=lambda x:-x[1])
        flip=sum(n for _,n in top)
        facts[weight]=dict(drops=drops,largest_drop=largest,dense_total_drop=total,
            largest_share=largest['drop']/total,flippers=flip,top_flipper_scenes=top[:5],
            top3_flipper_share=sum(n for _,n in top[:3])/flip if flip else None,
            baseline_cloud_to_noncloud_reversals=sum(d['records']['baseline']['crossing']['cloud_to_noncloud_reversals'] for d in data))
        lines += [f"## {weight}",'',f"0.70→0.50累计下降{total:.4f}点；最大相邻下降在{largest['start']:.2f}→{largest['end']:.2f}，下降{largest['drop']:.4f}点，占该段净下降{largest['drop']/total:.1%}。原图正确地表在0.50变云共{flip:,}像素。",'',
            '|路径|原图mIoU|0.50 mIoU|相对自身原图变化|落差改善|偏蓝绝对收益|','|---|---|---|---|---|---|']
        base_delta=values[-1]['mIoU']-values[0]['mIoU']
        for mode,vs in summary.items():
            delta=vs[-1]['mIoU']-vs[0]['mIoU']
            lines.append(f"|{mode}|{vs[0]['mIoU']:.4f}|{vs[-1]['mIoU']:.4f}|{delta:+.4f}|{delta-base_delta:+.4f}|{vs[-1]['mIoU']-values[-1]['mIoU']:+.4f}|")
        lines.append('')
    lines+=['此次干预仅覆盖第一次交叉注意力；未关闭第一阶段MsRE、局部读取和反馈。原图context替换属于同图诊断参照。翻转/稳定组由常规路径结果定义，得分差曲线描述其变化，不把分组后的差异本身当作因果证据。']
    (R/'READOUT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (R/'LOCAL_VALIDATION.json').write_text(json.dumps(dict(passed=True,pooled_metric_checks=checks,transition_conservation=True,all_64_original_replacements_exact=True,facts=facts),indent=2))
    print(json.dumps(facts,indent=2))
if __name__=='__main__':main()
