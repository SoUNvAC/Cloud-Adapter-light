from pathlib import Path
import json,csv,hashlib
R=Path(__file__).resolve().parents[2]/'outputs/phase466/phase466_20261107'
names={'msre_off':'关闭MsRE（读取结果随之不用）','read_off':'关闭局部读取','write_off':'关闭局部反馈','read_write_off':'关闭局部读取＋反馈','token_read_off':'关闭MsRE多尺度token读取','projector_off':'关闭MsRE特征投影'}
rows=[];lines=[]
for mode,label in names.items():
 d=json.loads((R/mode/'ABLATION_OFF.json').read_text());assert d['passed'] and d['extra_forwards']==535 and d['target_label_supervision_percent']==0
 r=d['results'][0];assert r['state_unchanged']
 p=json.loads((R/mode/'OFF_INDEPENDENT_6000.json').read_text())['per_image'];assert len(p)==535 and len({v['group'] for v in p})==107
 cm=r['confusion']
 assert cm==[[sum(v['confusion'][a][b] for v in p) for b in range(3)] for a in range(3)]
 assert all(abs(100*cm[c][c]/(sum(cm[c])+sum(v[c] for v in cm)-cm[c][c])-r['IoU'][c])<1e-10 for c in range(3))
 delivery=json.loads((R/mode/'DELIVERY_SHA.json').read_text())
 assert all(hashlib.sha256((R/mode/n).read_bytes()).hexdigest()==h for n,h in delivery.items())
 t=r['transition'];fn=sum(t[2][a][b] for a in range(3) for b in [0,1])-sum(t[2][a][b] for a in [0,1] for b in range(3));fp=sum(t[c][a][2] for c in [0,1] for a in range(3))-sum(t[c][2][b] for c in [0,1] for b in range(3))
 row=dict(mode=mode,surface=r['IoU'][0],cloud=r['IoU'][1],shadow=r['IoU'][2],mIoU=r['mIoU'],shadow_delta=r['IoU_delta'][2],FN_delta=fn,FP_delta=fp,shadow_CI95=r['paired_ROI_delta_CI95'][2]);rows.append(row)
 lines.append('|'+label+'|'+ '|'.join(f"{v:.4f}" for v in r['IoU'])+f"|{r['IoU_delta'][2]:+.4f}|{fn:+d}|{fp:+d}|" )
with (R/'MAIN.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,images_per_condition=535,ROI_count=107,total_forwards=3210,target_label_supervision_percent=0,states_unchanged=True),indent=2))
(R/'RESULT.md').write_text('# Phase466：固定权重的特征路径诊断\n\n目标域监督 **0%**。原始CloudSEN验证535张，6种干预共3210次前向；零训练、固定BN和argmax。基准为原431源网络＋450分支。\n\n| 干预 | 地表IoU | 云IoU | 云影IoU | 云影变化 | 漏检变化/像素 | 误报变化/像素 |\n|---|---:|---:|---:|---:|---:|---:|\n|450完整网络|89.2515|84.9714|65.5593|0|0|0|\n'+'\n'.join(lines)+'\n\n关闭读取：MsRE局部输入置零。关闭反馈：detail不更新，context也不再获得反馈差值。关闭MsRE：直接返回原token，因此局部读取结果同时失去出口。450分支仍接收原空间CNN输出。关闭token读取保留delta=x+detail及其投影；关闭投影则保留完整delta，跳过1×1 projector。\n\n这些数值测量已有权重对路径的依赖，不代表移除模块后重新训练能得到相同性能。107个ROI配对bootstrap 2000次，区间及像素错误转移见各条件JSON和NPZ。所有模型状态评测前后哈希一致。\n',encoding='utf-8')
print(json.dumps(rows))

with (R/'RESULT.md').open('a',encoding='utf-8') as f:
 f.write("\n## 结论\n\n1. 当前网络明显依赖MsRE整条路径：关闭后云影下降24.7561点；其中关闭1×1特征投影下降21.7213点，107个ROI配对95%区间为[-25.6885,-18.1004]。这两项干预不能相加或据此计算贡献占比。\n2. 多尺度token读取关闭只下降0.1670点，区间[-0.3951,0.0375]；局部读取下降0.0289点，反馈关闭变化+0.0017点，均未显示明确的云影净收益。并不说明这些模块在训练中无用。\n3. 最优模型仍是450，云影65.5593%，没有达到70%。本轮是诊断，没有训练新网络。\n4. 下一结构实验优先改MsRE投影主路径：用局部与环形邻域的特征差异调制投影，针对遮光前后地表内容相似、亮度不同的线索。保留原投影初始化，新增修正零初始化；必须同时设仅解冻同一投影的训练对照，区分新增结构与继续训练收益。采用原源域训练标签，不改阈值。\n")
files=[Path(__file__),*Path(__file__).parent.glob('*466.py'),Path(__file__).with_name('launch_components466.sh'),Path(__file__).with_name('launch_dissect466.sh'),Path(__file__).parents[1]/'research_plans/phase466.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(Path(__file__).resolve().parents[2])):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
