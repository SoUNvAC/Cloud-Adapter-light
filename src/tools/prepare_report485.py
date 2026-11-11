from pathlib import Path
T=Path(__file__).parent
s=(T/'report_shadow484.py').read_text(encoding='utf-8').replace('484','485')
s=s.replace('双尺度云影修正','云影过渡剖面证据')
start=s.index(" '保留482B细节支路，")
end=s.index(" '两组均无辅助BCE",start)
s=s[:start]+" '保留482B细节支路，新增RGB连续剖面支路：RGB缩至1/8，沿横、竖及两对角采样9点，轴向间隔8像素。对数RGB、对数亮度和两色度相对中心差异，经共享沿剖面卷积、位置融合、方向均值/最大聚合，再零初始化投影注入第一层特征。A正常顺序，B固定乱序；同容量、同初始参数。新增13776参数，总模块29041参数。均匀RGB证据为零。','',\n"+s[end:]
s=s.replace('区域支路开启','剖面支路开启').replace('新增分支18个参数张量','新增分支17个参数张量').replace('regional_enabled','evidence_enabled').replace('区域支路为潜在特征修正，不是物理遮光量估计。','剖面是观测RGB线索，不是物理半影或透射率测量。B保留全部采样值及固定位置映射，只破坏卷积的邻接顺序，不能视为完全删除空间信息。REGION_OFF文件沿用通用命名，实际关闭剖面支路。')
pos=s.index("lines += ['验收：")
s=s[:pos]+"z=comparison['485A_6000_minus_485B_6000'];lines += [f\"正常顺序A−固定乱序B（6000步）：三类增量{z['IoU_delta']}，云影场景区间{z['CI95'][2]}。\",'']\n"+s[pos:]
(T/'report_shadow485.py').write_bytes(s.encode('utf-8'))
s=(T/'log_shadow484.py').read_text(encoding='utf-8').replace('484','485').replace('双尺度云影修正','云影过渡剖面证据')
start=s.index(" '- 基于无辅助损失")
end=s.index(" '- 两组从",start)
s=s[:start]+" '- 基于无辅助损失482B，新增观测RGB的四方向9点连续剖面，6通道中心差分经沿剖面卷积、位置融合及方向聚合后零初始化注入第一层特征。A正常顺序，B固定乱序，同容量、同初始化；新增13776参数，总模块29041参数。均匀颜色证据为零。',\n"+s[end:]
s=s.replace('区域支路单独关闭','剖面支路单独关闭')
(T/'log_shadow485.py').write_bytes(s.encode('utf-8'))
