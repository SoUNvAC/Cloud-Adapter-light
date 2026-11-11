from pathlib import Path
import re
T=Path(__file__).parent
description='保留488B，读取第1及第4交互位置输出的同网格特征，分别LayerNorm并投影到16维。A拼接浅深差分及逐通道乘积，B直接拼接两层投影作同参数对照，经卷积与零投影注入第一层特征。新增19088参数，总模块42625参数，不增加骨干前向。没有将乘积直接解释成真实材质保持量。'
for name in ['report_shadow490.py','log_shadow490.py']:
 s=re.sub(r'(?<!\d)490(?!\d)','491',(T/name).read_text(encoding='utf-8')).replace('纹理相容方向邻域比较','浅深层关系交互').replace('方向邻域','跨层').replace('新增分支21个参数张量','新增分支22个参数张量')
 if name.startswith('report'):
  a=s.index("'保留488B原始梯度纹理分支，");b=s.index(" '两组均无辅助BCE",a);s=s[:a]+repr(description)+",'',\n"+s[b:]
  s=s.replace('相容加权A−均匀B','差分乘积A−拼接B').replace('相容权重是预设语义和纹理相似度函数，其效果由A/B对照检验。','浅深层投影共同训练，差分乘积的有效性由A/B对照检验。').replace('directional_enabled','cross_depth_enabled')
 else:
  a=s.index("'- 保留488B原始梯度纹理分支，");b=s.index(" '- 两组从431",a);s=s[:a]+repr('- '+description)+',\n'+s[b:]
 (T/name.replace('490','491')).write_bytes(s.encode('utf-8'))
