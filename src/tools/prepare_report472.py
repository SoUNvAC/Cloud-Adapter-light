from pathlib import Path
T=Path(__file__).parent
s=(T/'report_shadow471.py').read_text(encoding='utf-8').replace('471','472')
s=s.replace("'自由二维位移' if arm=='A' else '轮廓法向位移'","'原网络联合续训' if arm=='A' else '主路径遮光对比'")
s=s.replace('ABLATION_NORMAL_OFF','ABLATION_APPEARANCE_OFF').replace('NORMAL_OFF_','APPEARANCE_OFF_').replace('B同权重去掉法向约束','B同权重关闭外观偏好')
s=s.replace('B_transport_on_minus_off','B_contrast_on_minus_off').replace('B_normal_on_minus_off','B_appearance_on_minus_off')
s=s.replace("assert all(update['new_changed'].values()) and not any(update['changed'].values())", "assert all(update['new_changed'].values())\n assert update['changed']['backbone.refinement.']>0 and update['changed']['decode_head.']>0\n assert not any(v for k,v in update['changed'].items() if k not in ['backbone.refinement.','decode_head.'])")
a=s.index("lines=['# Phase472");b=s.index('\nfor name,x in rows:',a)
s=s[:a]+'''lines=['# Phase472：MsRE主路径邻域遮光对比投影','','目标监督0%；CloudSEN训练8490/原始验证535；450起点，两组各6000步，batch4、seed45201，固定argmax。','','A：原MsRE projector与解码器联合续训。B：相同训练范围，另加6147参数参照选择器；8个间隔2-token邻居，通过语义相似、色度接近及较亮偏好形成软参照，在原MsRE projector输入使用x+0.25*(x-reference)，四阶段共享。无额外零输出投影头。','','原模块lr1e-5，新模块lr1e-4；其他原参数、450分支、DINOv3及BN固定。两组损失完全相同，参考固定450，无额外辅助监督。B强度在前2000步从0升到0.25并固定，写入权重；两组均只从2000步及以后选训练最优点，step0回退。','','RGB是外观描述，软参照不是已知同材质地物或真实无影反射率。关闭B结构仍保留已训练的原投影/解码器，不能期待恢复450；净结构收益主要看B对同条件A，开关检验共同适配后的条件作用。','','|模型|地表IoU|云IoU|云影IoU|mIoU|','|---|---:|---:|---:|---:|']'''+s[b:]
a=s.index("diag=read(R/'TRANSPORT_DIAG.json')");b=s.index("lines+=['','决策：'",a)
s=s[:a]+'''diag=read(R/'CONTRAST_DIAG.json');assert diag['independent_predictions_identical'] and diag['state_unchanged'] and diag['strength']==.25
lines+=['','B四阶段投影输入修正RMS/原输入RMS的逐图均值：'+json.dumps(diag['stage_mean_relative_input_rms'])+'；中位数：'+json.dumps(diag['stage_median_relative_input_rms'])+'。该量说明参与幅度，不等于性能收益。']
'''+s[b:]
s=s.replace('原参数与BN冻结','未开放原参数与BN冻结、原投影和解码器更新')
s=s.replace('参考：[Improving Semantic Segmentation via Decoupled Body and Edge Supervision](https://arxiv.org/abs/2007.10035)。借鉴特征位移思想，本轮检验云影距离场和法向约束，不将通用位移宣称为原创。','参考：[Adaptive Illumination Mapping for Shadow Detection in Raw Images](https://openaccess.thecvf.com/content/ICCV2023/html/Sun_Adaptive_Illumination_Mapping_for_Shadow_Detection_in_Raw_Images_ICCV_2023_paper.html)。借鉴对比信息动机，本轮不使用RAW照明映射。')
s=s.replace('ablate_normal472.py','ablate_appearance472.py')
s='\n'.join(line for line in s.splitlines() if "B的逐图特征修正RMS/原特征RMS" not in line)+'\n'
(T/'report_shadow472.py').write_bytes(s.encode('utf-8'))
