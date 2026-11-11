from pathlib import Path
T=Path(__file__).parent
s=(T/'report_shadow479.py').read_text(encoding='utf-8').replace('479','480').replace('4992','15265').replace("len(u['new_changed'])==6","len(u['new_changed'])==13").replace('ENVELOPE_DIAG','REDISTRIBUTION_DIAG')
start=s.index("lines=['# Phase480");end=s.index('for name,x in rows:lines.append',start)
intro=['# Phase480：云影亚patch重分配','','在第一层VFM金字塔特征、解码器前新增15265参数分支。原生1/16 VFM特征与1/4 CNN细节分别投影到16维，融合后预测云影覆盖率，软门控16维细节并投影到384维残差。A普通残差；B先减去每个4×4子格均值，保持原patch平均特征。末层无bias且零初始化，两组初始参数相同。','','CloudSEN训练8490/原始验证535；目标监督0%；431+450起点，6000步、batch4、seed45201、分支seed48001、warmup200。新分支lr1e-3，解码器lr1e-5；其余原参数、450分支、VFM与BN固定，固定argmax。','','A/B采用相同分割保持损失，并均加入0.1倍云影覆盖率BCE；覆盖率为原三类标签的4×4有效像素占比，255忽略。A/B是匹配的均值约束对照；477没有新分支和辅助损失，比较只代表整套方案增量，不能单独归因于均值约束。','','1000/3000/6000及最佳点535图独立复算；最佳点同权重关分支。107ROI配对bootstrap2000次。450地表/云为保护线，云影目标70%。','','|模型|步数|地表IoU|云IoU|云影IoU|mIoU|','|---|---:|---:|---:|---:|---:|']
s=s[:start]+'lines='+repr(intro)+'\n'+s[end:]
start=s.index("lines+=['',*details");end=s.index("(R/'RESULT.md')",start)
s=s[:start]+'''assert diag['maximum_patch_mean_residual']<1e-5
lines+=['',*details,'',f"B分支相对第一层金字塔特征RMS变化均值{diag['mean_relative_detail_rms']:.6f}；535图最大patch均值残差{diag['maximum_patch_mean_residual']:.9g}。均值保持只约束特征，不保证类别IoU不变。",'','超过476关层候选且保护450地表/云的检查点：'+json.dumps(candidates),'',(R/'INTERPRETATION.md').read_text(encoding='utf-8'),'','方法参考：[FADE细节引导上采样](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870228.pdf)。本轮测试云影覆盖率引导与patch均值保持组合，不声称上采样或零均值算子原创。','','验收：新分支13个参数张量与原解码器更新，其余原参数/BN固定；共同初始化/记录批次、独立混淆矩阵、权重和传输SHA通过。单seed验证集探索，场景区间不包含重新训练seed波动。','','部署：加载原431基座、student_patch以及infer_shadow480中的分支state_dict。off是显式enabled=False诊断条件，单独state_dict不保存该运行时开关，不能将off结果当作默认部署结果。']
'''+s[end:]
s=s.replace('保持原patch平均特征','保持分支加入前每个4×4块的平均特征')
s=s.replace('diag=read(R/', "for arm in ['A','B']:\n e=read(R/arm/'network_run/EFFECT_BOUNDARY_DIAG.json');assert e['passed']\n details.append(arm+'最佳点开启减关闭的边界错误变化：'+json.dumps(e['delta_on_minus_off']))\ndiag=read(R/",1)
(T/'report_shadow480.py').write_bytes(s.encode())
s=(T/'log_shadow479.py').read_text(encoding='utf-8').replace('479','480')
start=s.index('lines=[');end=s.index("for arm in ['A','B']:",start)
intro=['## Phase480：云影亚patch重分配（2026-11-08，完成）','','- 第一层VFM金字塔前新增15265参数分支：1/16语义与1/4 CNN细节融合，覆盖率预测软门控细节并投影384维；A普通残差，B约束每4×4子格均值为0，末层无bias零初始化。','- 原431+450起点；CloudSEN训练8490/验证535、512²、FP32、4090D、目标监督0%。6000步、batch4、seed45201、分支seed48001，warmup200；新分支lr1e-3/解码器lr1e-5，其余原参数、450分支与BN冻结。','- A/B相同分割保持损失加0.1覆盖率BCE；477没有新分支及辅助损失。固定argmax，保护450地表/云、目标影70%。1000/3000/6000及最佳点独立535图，最佳点同权重关分支、107ROI配对区间及B均值保持复核。']
s=s[:start]+'lines='+repr(intro)+'\n'+s[end:]
s=s.replace('形态学闭运算/黑顶帽为已有方法，研究本轮网络融合，不声称算子原创。','细节引导上采样参考FADE，本轮测试云影覆盖率与patch均值保持组合。').replace('shadow_envelope480.py','shadow_redistribution480.py')
(T/'log_shadow480.py').write_bytes(s.encode())
