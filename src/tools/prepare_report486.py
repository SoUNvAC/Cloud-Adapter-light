from pathlib import Path
T=Path(__file__).parent
s=(T/'report_shadow484.py').read_text(encoding='utf-8').replace('484','486').replace('双尺度云影修正','场景遮光变化共识')
a=s.index(" '保留482B细节支路，");b=s.index(" '两组均无辅助BCE",a)
s=s[:a]+" '保留482B细节支路，新增全图共享RGB变化方向：1/8网格比较8邻居，轴向距离32像素，以冻结VFM内容相似度、色度差和亮度差作软权重，从全图对数RGB差估计共同方向。局部沿方向变化、正交残差等6种量经邻居聚合形成12通道，轻量CNN后零投影注入第一层特征。A每图估计方向，B固定等比例RGB方向。新增8272参数，总模块23537参数。','',\n"+s[b:]
s=s.replace('区域支路开启','共识证据支路开启').replace('新增分支18个参数张量','新增分支17个参数张量').replace('regional_enabled','evidence_enabled').replace('区域支路为潜在特征修正，不是物理遮光量估计。','共识方向是图像统计代理，不是真实光照、反射率或太阳方向。REGION_OFF沿用通用文件名，实际关闭共识证据支路。方向替换干预将adaptive_mode设为False并保存在诊断权重中；该权重不是原A部署条件。')
pos=s.index("lines += ['验收：")
s=s[:pos]+"""z=comparison['486A_6000_minus_486B_6000'];lines += [f"A场景方向−B固定方向（6000步）：三类增量{z['IoU_delta']}，云影场景区间{z['CI95'][2]}。",'']
z=comparison['486Abest_minus_486Afixed'];lines += [f"A最佳点同权重，场景方向−替换固定方向：三类增量{z['IoU_delta']}，云影场景区间{z['CI95'][2]}。",'']
for arm,v in read(R/'DIRECTION_DIAGNOSTIC.json')['arms'].items():lines += [f"{arm}与等比例RGB方向夹角（度，最小/25%/中位/75%/最大）：{v['angle_to_equal_RGB_degrees_quantiles']}。",'']
lines += ['文献依据及与455/461/476/479/485的区别见LITERATURE.md；借鉴色度不变性及材质一致边缘思想，未复现原论文的阴影去除或测试时优化。','']
"""+s[pos:]
(T/'report_shadow486.py').write_bytes(s.encode('utf-8'))
s=(T/'log_shadow484.py').read_text(encoding='utf-8').replace('484','486').replace('双尺度云影修正','场景遮光变化共识')
a=s.index(" '- 基于无辅助损失");b=s.index(" '- 两组从",a)
s=s[:a]+" '- 基于482B，新增全图加权对数RGB变化方向及局部投影/正交残差。A每场景估计，B固定等比例方向；冻结VFM内容相似度等作软权重，新增8272参数，总模块23537参数。借鉴DC-ShadowNet色度不变性与WACV2025材质一致边缘思想，具体设计与文献边界见LITERATURE.md。',\n"+s[b:]
s=s.replace('区域支路单独关闭','共识证据支路单独关闭、同权重场景方向替换固定方向')
(T/'log_shadow486.py').write_bytes(s.encode('utf-8'))
