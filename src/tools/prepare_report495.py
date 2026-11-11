from pathlib import Path
T=Path(__file__).parent
description='保留493B全部输出路径，共享已有16通道细纹理编码，池化到token网格并通过零初始化16→384投影写入MsRE输出。A在第1次交互写入并经过后续9个冻结VFM块，B在第4次交互写入；均按当前token的脱离梯度RMS匹配量纲。新增6144参数，总结构模块43137参数，无额外骨干前向，但A需要反向穿过后续冻结块。'
for stem in ['report_shadow','log_shadow']:
 s=(T/f'{stem}493.py').read_text(encoding='utf-8')
 for prefix in ['phase','Phase','shadow','redistribution','effect','region','archive','all']:s=s.replace(prefix+'493',prefix+'495')
 s=s.replace('493{','495{').replace('493A_','495A_').replace('493B_','495B_').replace("'493A'","'495A'").replace("'493B'","'495B'").replace('*493.py','*495.py').replace('*493.sh','*495.sh')
 s=s.replace('phase488_20261108','phase493_20261109').replace('outputs/phase488/','outputs/phase493/').replace('488B','493B').replace('下采样前亮暗次序纹理','细纹理写回MsRE主干token').replace('次序纹理','token写回')
 if stem=='report_shadow':
  a=s.index("'保留493B，");b=s.index(" '两组均无辅助BCE",a);s=s[:a]+repr(description)+",'',\n"+s[b:]
  s=s.replace('亮暗次序A−亮度差幅度B','第1次写入A−第4次写入B')
  s=s.replace('结构模块21个参数张量（保留17个、新增4个）','结构模块22个参数张量（保留21个、新增1个）')
  s=s.replace('次序编码与幅度编码由A/B对照检验；不能将单独增加容量解释成遮光机制。空间变化光照、噪声、量化和饱和仍可改变次序。','A/B对照改变接入位置和后续传播路径；不能把差异归因于单一的语义传播机制。')
  s=s.replace('ordinal_enabled运行时开关','ordinal_enabled、feedback_enabled运行时开关')
  s=s.replace('保留493B纹理分支','保留493B原有全部细节分支')
 else:
  a=s.index("'- 保留493B，");b=s.index(" '- 两组从431",a);s=s[:a]+repr('- '+description)+',\n'+s[b:]
 (T/f'{stem}495.py').write_bytes(s.encode('utf-8'))
