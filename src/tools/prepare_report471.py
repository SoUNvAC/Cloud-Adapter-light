from pathlib import Path
T=Path(__file__).parent
s=(T/'report_shadow470.py').read_text(encoding='utf-8').replace('470','471')
s=s.replace('A等权专家','A自由二维位移').replace("'等权专家' if arm=='A' else '错误类型竞争路由'","'自由二维位移' if arm=='A' else '轮廓法向位移'")
s=s.replace('ABLATION_ROUTING_OFF','ABLATION_NORMAL_OFF').replace('ROUTING_OFF_','NORMAL_OFF_').replace('B同权重等权混合','B同权重去掉法向约束')
s=s.replace('B_experts_on_minus_off','B_transport_on_minus_off').replace('B_routing_on_minus_off','B_normal_on_minus_off')
start=s.index("lines=['# Phase471")
end=s.index('\nfor name,x in rows:',start)
s=s[:start]+'''lines=['# Phase471：云影轮廓法向特征位移','','目标监督0%；CloudSEN训练8490/原始验证535；450起点，两组各6000步，batch4、seed45201，固定argmax。','','在解码器前1/4分辨率特征加入16348参数模块。融合空间CNN、四级语义和RGB描述，预测云影有符号距离、二维位移与混合门。A自由二维位移；B将同一向量投影到预测距离场的法向，最大位移2网格（8原图像素）。按预测距离衰减远离边界的修正。原解码器和450分支读取相同修正特征。','','原网络全部权重和BN冻结，仅新增模块lr1e-4。位移零初始化，使用同一采样网格差分保证初始精确零更新。两组辅助监督相同：0.1距离场损失，内含0.5梯度项；其余分割损失沿用450参考。','','训练距离场由源标签在1/4网格的云影占比≥50%生成，内正外负、截断±8网格；未知标签附近不监督。此二值化只生成辅助目标，不改最终分割阈值。推理不使用标签，也不额外运行参考网络。','','|模型|地表IoU|云IoU|云影IoU|mIoU|','|---|---:|---:|---:|---:|']'''+s[end:]
start=s.index("route=read(R/'ROUTING_DIAG.json')")
end=s.index("lines+=['','决策：'",start)
s=s[:start]+s[end:]
s=s.replace("参考：[Distraction-Aware Shadow Detection, CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Zheng_Distraction-Aware_Shadow_Detection_CVPR_2019_paper.html)。借鉴显式学习阴影干扰区域，实验实现为错误类型驱动的高分辨率特征专家路由。","参考：[Improving Semantic Segmentation via Decoupled Body and Edge Supervision](https://arxiv.org/abs/2007.10035)。借鉴特征位移思想，本轮检验云影距离场和法向约束，不将通用位移宣称为原创。")
s=s.replace('ablate_routing471.py','ablate_normal471.py')
(T/'report_shadow471.py').write_bytes(s.encode('utf-8'))
