from pathlib import Path
T=Path(__file__).parent
description='保留482B细节支路，新增局部纹理结构张量与环形邻域比较。12/28像素尺度提取纹理方向、能量及邻域差异，卷积编码后零投影注入第一层特征。A梯度按局部亮度加0.02归一化，B原始梯度。新增8272参数，总模块23537参数。不增加骨干前向；显示RGB上计算，不宣称严格光照不变。'
s=(T/'report_shadow487.py').read_text(encoding='utf-8').replace('487','488').replace('交错patch采样融合','亮度归一化纹理比较')
a=s.index(" '保留482B细节支路，");b=s.index(" '两组均无辅助BCE",a)
s=s[:a]+repr(description)+",'',\n"+s[b:]
s=s.replace('交错采样支路','纹理支路').replace('半patch A−整patch B','归一化A−原始梯度B')
a=s.index('REGION_OFF沿用通用文件名');b=s.index('enabled、evidence_enabled',a)
s=s[:a]+'REGION_OFF沿用通用文件名，实际关闭新增纹理支路。归一化包含稳定常数，亮度仍作为一个上下文通道，故不是严格亮度不变网络。'+s[b:]
(T/'report_shadow488.py').write_bytes(s.encode('utf-8'))
s=(T/'log_shadow487.py').read_text(encoding='utf-8').replace('487','488').replace('交错patch采样融合','亮度归一化纹理比较')
a=s.index(" '- 保留482B，");b=s.index(" '- 两组从431",a)
s=s[:a]+repr('- '+description)+',\n'+s[b:]
s=s.replace('第二视图融合支路','纹理支路')
(T/'log_shadow488.py').write_bytes(s.encode('utf-8'))
