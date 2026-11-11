from pathlib import Path
import re
T=Path(__file__).parent
description='保留488B，新增颜色梯度分解支路：在12/28像素尺度上，把RGB空间梯度分成沿颜色方向的平行分量、垂直残差及能量占比。A采用局部RGB方向，B固定灰色方向，编码后零投影注入第一层。新增8272参数，总模块31809参数，不增加骨干前向；显示RGB仅作学习线索，不作严格反射率解释。'
for name in ['report_shadow490.py','log_shadow490.py']:
 s=re.sub(r'(?<!\d)490(?!\d)','492',(T/name).read_text(encoding='utf-8')).replace('phase492_20261108','phase492_20261109').replace('2026-11-08，完成','2026-11-09，完成').replace('纹理相容方向邻域比较','颜色方向遮光边缘证据').replace('方向邻域','颜色梯度')
 if name.startswith('report'):
  a=s.index("'保留488B原始梯度纹理分支，");b=s.index(" '两组均无辅助BCE",a);s=s[:a]+repr(description)+",'',\n"+s[b:]
  s=s.replace('相容加权A−均匀B','局部颜色方向A−固定灰色方向B').replace('相容权重是预设语义和纹理相似度函数，其效果由A/B对照检验。','颜色方向分解的有效性由A/B对照检验，不能视为严格光照不变量。').replace('directional_enabled','chromatic_enabled')
 else:
  a=s.index("'- 保留488B原始梯度纹理分支，");b=s.index(" '- 两组从431",a);s=s[:a]+repr('- '+description)+',\n'+s[b:]
 (T/name.replace('490','492')).write_bytes(s.encode('utf-8'))
