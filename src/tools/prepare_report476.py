from pathlib import Path
T=Path(__file__).parent
s=(T/'report_shadow474.py').read_text(encoding='utf-8').replace('474','476')
s=s.replace("'普通组合' if arm=='A' else '条件组合'","'logRGB引导' if arm=='A' else 'log色度引导'")
s=s.replace("assert all(update['new_changed'].values()) and not any(update['changed'].values())","assert all(update['new_changed'].values()) and update['changed']['decode_head.']>0\n assert all(not v for k,v in update['changed'].items() if k!='decode_head.')")
s=s.replace(" assert read(d/'REFERENCE_CACHE_CHECK.json')['passed']\n",'')
s=s.replace("('ABLATION_CONDITIONAL_OFF.json','CONDITIONAL_OFF_','B同权重普通组合'),('ABLATION_SPATIAL_MEAN.json','SPATIAL_MEAN_','B同权重全图均值修正')","('ABLATION_APPEARANCE_OFF.json','APPEARANCE_OFF_','B同权重logRGB引导')")
s=s.replace("   if prefix=='OFF_':assert x['confusion']==base['confusion']\n",'')
start=s.index("lines=['# Phase476");end=s.index('\nfor name,x in rows:lines.append',start)
s=s[:start]+'''lines=['# Phase476：材质引导空间卷积','','目标监督0%；CloudSEN训练8490/验证535。两组各6000步、batch4、seed45201、模块seed47601；新增模块lr1e-3、原解码器lr1e-5，warmup200、固定argmax。','','在空间CNN的1/4细节上，用8邻居特征差分、可学习逐通道卷积核和1×1通道混合提取特征；按RGB引导距离分配邻域权重。修正同步池化进入1/16上下文，位于VFM交互之前。两组各4609新参数。','','A使用logRGB距离；B使用去通道均值的log色度距离，弱化未截断无色增益对引导的影响。材质只是设计动机，相似色度不等于已知同材质或物理反射率。','','原参数仅开放解码器，其他原参数、450分支、BN固定。零差分核初始复现450；新层的梯度穿过冻结VFM回传。两组采用相同不加权CE、KL/softIoU保持约束，初始化与记录批次相同。','','|模型|地表IoU|云IoU|云影IoU|mIoU|','|---|---:|---:|---:|---:|']'''+s[end:]
s=s.replace("'B_decoder_on_minus_off','B_conditional_on_minus_off','B_spatial_minus_mean'","'B_contrast_on_minus_off','B_appearance_on_minus_off'")
s=s.replace("lines+=['','全图均值干预将两头输出各自替换为每图平均值，保留每图偏置，移除学习修正的空间变化。该干预用于区分空间解码作用与整体偏置作用。']","lines+=['','B_contrast_on_minus_off表示空间卷积开关；B_appearance_on_minus_off表示色度引导对同权重logRGB引导。关闭新层保留已训练解码器，不应期待恢复450。']")
s=s.replace("name.startswith(('A','B条件','B受保护'))","name.startswith(('Alog','A受保护','Blog','B受保护'))")
s=s.replace('B同权重切换组合仅为条件干预；架构比较以独立训练A/B为准。','同权重切换引导仅为条件干预；结构增量以独立训练A/B为准。')
(T/'report_shadow476.py').write_bytes(s.encode())
s=(T/'verify_archive475.py').read_text().replace('475','476');(T/'verify_archive476.py').write_bytes(s.encode())
