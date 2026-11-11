from pathlib import Path
T=Path(__file__).parent
s=(T/'report_shadow484.py').read_text(encoding='utf-8').replace('484','487').replace('双尺度云影修正','交错patch采样融合')
a=s.index(" '保留482B细节支路，");b=s.index(" '两组均无辅助BCE",a)
s=s[:a]+" '保留482B细节支路，新增第二采样网格的语义差融合：同一冻结431骨干额外读取平移图像，第一层特征逆移回原坐标，与原第一层特征作差，经低秩卷积及零投影注入。A平移8像素（半patch），B平移16像素（整patch），两组统一屏蔽16像素边缘。新增12688参数，总模块27953参数。融合参数与解码器共同训练，不是预测投票；部署新增一次骨干前向。','',\n"+s[b:]
s=s.replace('区域支路开启','交错采样支路开启').replace('新增分支18个参数张量','新增分支17个参数张量').replace('regional_enabled','evidence_enabled').replace('区域支路为潜在特征修正，不是物理遮光量估计。','REGION_OFF沿用通用文件名，实际关闭第二视图特征融合；关闭后跳过额外骨干前向。B整patch视图仍存在填充及全局上下文差异，不等于原特征的严格复制。forward_images计数包含额外骨干图像次数，主模型每图2次，冻结参考每图1次。')
s=s.replace('辅助结果：验证集最佳云影点','辅助结果：训练内验证预选点的独立重载结果')
pos=s.index("lines += ['验收：")
s=s[:pos]+"""z=comparison['487A_6000_minus_487B_6000'];lines += [f"半patch A−整patch B（6000步）：三类增量{z['IoU_delta']}，云影场景区间{z['CI95'][2]}。",'']
cost=read(R/'INFERENCE_COST.json');assert cost['passed'] and cost['state_unchanged']
lines += [f"推理成本：同一张512图、batch1、各5次预热、交替顺序各30次CUDA event计时。中位毫秒{cost['CUDA_event_median_ms']}，A/482B={cost['ratio']:.3f}倍。仅GPU前向，不含加载、IO、预处理，不代表全数据吞吐。",'']
for arm in ['A','B']:
 d=R/arm/'network_run';lines += [f"{arm}并行训练墙钟{read(d/'COMPLETE.json')['seconds']/60:.2f}分钟，峰值分配显存{read(d/'UPDATE_CHECK.json')['peak_memory_bytes']}字节。",'']
"""+s[pos:]
(T/'report_shadow487.py').write_bytes(s.encode('utf-8'))
s=(T/'log_shadow484.py').read_text(encoding='utf-8').replace('484','487').replace('双尺度云影修正','交错patch采样融合')
a=s.index(" '- 基于无辅助损失");b=s.index(" '- 两组从",a)
s=s[:a]+" '- 保留482B，新增第二采样网格语义差融合，额外读取同一冻结431骨干，逆移第一层特征后与原网格作差并低秩注入。A平移8像素，B平移16像素，统一屏蔽16像素边缘；新增12688参数，总模块27953参数。部署多一次骨干前向，不是免费增益或预测投票。',\n"+s[b:]
s=s.replace('区域支路单独关闭','第二视图融合支路单独关闭')
(T/'log_shadow487.py').write_bytes(s.encode('utf-8'))
