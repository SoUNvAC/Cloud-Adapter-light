from pathlib import Path
T=Path(__file__).parent
description='保留488B原始梯度纹理分支，新增16/32像素偏移的8方向邻域比较。读取相对亮度、纹理能量差、纹理方向差、语义差；A按冻结语义及纹理相容性加权，B均匀加权，卷积编码后零投影注入第一层特征。新增7696参数，总模块31233参数，不增加骨干前向。'
s=(T/'report_shadow488.py').read_text(encoding='utf-8').replace('488','490').replace('482B','488B').replace('phase482','phase488').replace('亮度归一化纹理比较','纹理相容方向邻域比较')
a=s.index("'保留482B细节支路，") if "'保留482B细节支路，" in s else s.index("'保留488B细节支路，")
b=s.index(" '两组均无辅助BCE",a)
s=s[:a]+repr(description)+",'',\n"+s[b:]
s=s.replace('纹理支路开启','方向邻域支路开启').replace('归一化A−原始梯度B','相容加权A−均匀B').replace('新增分支17个参数张量','新增分支21个参数张量')
a=s.index('REGION_OFF沿用通用文件名');b=s.index("运行时开关不保存在state_dict",a)
s=s[:a]+'REGION_OFF沿用通用文件名，实际只关闭新增方向邻域支路，保留488B纹理分支。相容权重是预设语义和纹理相似度函数，其效果由A/B对照检验。enabled、evidence_enabled、directional_enabled'+s[b:]
(T/'report_shadow490.py').write_bytes(s.encode('utf-8'))
s=(T/'log_shadow488.py').read_text(encoding='utf-8').replace('488','490').replace('482B','488B').replace('亮度归一化纹理比较','纹理相容方向邻域比较')
a=s.index("'- 保留488B细节支路，");b=s.index(" '- 两组从431",a)
s=s[:a]+repr('- '+description)+',\n'+s[b:]
s=s.replace('纹理支路单独关闭','新方向邻域支路单独关闭')
(T/'log_shadow490.py').write_bytes(s.encode('utf-8'))
