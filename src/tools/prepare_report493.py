from pathlib import Path
import re
T=Path(__file__).parent
description='保留488B，新增下采样前亮暗次序纹理支路：原图灰度在1/2/4像素半径比较八邻域，先编码再4倍池化，并与周围12–28像素环形邻域比较。A取亮度差符号，B取tanh(20×亮度差)，同参数对照。48通道经卷积和零投影注入第一层，新增13456参数，总模块36993参数，不增加骨干前向。次序编码参考[Census局部变换](https://woodfill.com/Papers/Census94.pdf)，本轮检验其在云影网络中的作用。'
for name in ['report_shadow492.py','log_shadow492.py']:
 s=re.sub(r'(?<!\d)492(?!\d)','493',(T/name).read_text(encoding='utf-8'))
 s=s.replace('颜色方向遮光边缘证据','下采样前亮暗次序纹理').replace('颜色梯度','次序纹理').replace('chromatic_enabled','ordinal_enabled')
 if name.startswith('report'):
  a=s.index("'保留488B，");b=s.index(" '两组均无辅助BCE",a);s=s[:a]+repr(description)+",'',\n"+s[b:]
  s=s.replace('局部颜色方向A−固定灰色方向B','亮暗次序A−亮度差幅度B')
  s=s.replace('颜色方向分解的有效性由A/B对照检验，不能视为严格光照不变量。','次序编码与幅度编码由A/B对照检验；不能将单独增加容量解释成遮光机制。空间变化光照、噪声、量化和饱和仍可改变次序。')
  s=s.replace("if n!='branch_0.pth':assert sha(R/arm/'network_run'/n)==s['sha256']","assert len(s['sha256'])==64 and v['passed']")
  s=s.replace('weight_signatures=True,','remote_weight_signatures_verified=True,local_weights_complete=False,weight_storage="remote",')
  s=s.replace('权重签名与传输通过。','远端权重验签及元数据传输通过；完整权重保留远端。')
  s=s.replace('新增分支21个参数张量和解码器实际更新','结构模块21个参数张量（保留17个、新增4个）和解码器实际更新')
 else:
  a=s.index("'- 保留488B，");b=s.index(" '- 两组从431",a);s=s[:a]+repr('- '+description)+',\n'+s[b:]
 (T/name.replace('492','493')).write_bytes(s.encode('utf-8'))
