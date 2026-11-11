from pathlib import Path
T=Path(__file__).parent
s=(T/'compare_shadow474.py').read_text().replace('474','475')
start=s.index("cls=json.loads");end=s.index("cls=json.loads",start+1);s=s[:start]+s[end:]
s+='''
oldroot=Path('/home/scv/shared/phase474_20261107/A/network_run')
oldfinal=json.loads((oldroot/'FINAL_RESULTS.json').read_text());oldstep=oldfinal['choices']['best_shadow']
old=json.loads((oldroot/f'INDEPENDENT_{oldstep}.json').read_text())
for arm in ['A','B']:
 cms=np.array([[sum((np.array(r['confusion']) for r in source['per_image'] if r['group']==g),start=np.zeros((3,3),np.int64)) for source in [old,data[arm]]] for g in groups])
 m=np.einsum('bn,nkij->bkij',draws,cms);diag=np.diagonal(m,axis1=-2,axis2=-1);iou=100*diag/(m.sum(-1)+m.sum(-2)-diag);diff=iou[:,1]-iou[:,0]
 result[arm+'_minus_474A']=dict(IoU_delta=(np.array(data[arm]['result']['IoU'])-np.array(old['result']['IoU'])).tolist(),CI95=np.quantile(diff,[.025,.975],axis=0).T.tolist())
oh={r['step']:r for r in json.loads((oldroot/'HISTORY.json').read_text())}
result['same_step_old_control']=[dict(step=s,old474A=oh[s]['IoU'],A=h['A'][s]['IoU'],B=h['B'][s]['IoU']) for s in sorted(oh)]
(R/'MATCHED_COMPARISON.json').write_text(json.dumps(result,indent=2))
'''
(T/'compare_shadow475.py').write_bytes(s.encode())
s=(T/'report_shadow474.py').read_text(encoding='utf-8').replace('474','475')
s=s.replace("rows=[('450完整网络',base)]", "oldfinal=read(ROOT/'outputs/phase474/phase474_20261107/A/network_run/FINAL_RESULTS.json');old=next(x for x in oldfinal['results'] if 'best_shadow' in x['roles'])\nassert accept['matches_474A_initial_state_and_recorded_batches']\nrows=[('450完整网络',base),('474A原损失',old)]")
s=s.replace("'普通组合' if arm=='A' else '条件组合'", "'取消正确影加权' if arm=='A' else '全部像素权重1'")
s=s.replace("('ABLATION_CONDITIONAL_OFF.json','CONDITIONAL_OFF_','B同权重普通组合'),",'')
start=s.index("lines=['# Phase475");end=s.index('\nfor name,x in rows:lines.append',start)
s=s[:start]+'''lines=['# Phase475：云影误报的损失权重对照','','目标监督0%；CloudSEN训练8490/原始验证535。新A/B各6000步，batch4、seed45201、模块seed47401，lr1e-4、warmup200，固定argmax。','','全部使用474A普通组合网络，原450完整网络冻结，仅训练31106个新参数；网络与初始化相同。本轮没有提出新架构。','','474A原对照：云影相关难例权重4、正确影1.5，其余1。新A：难例4，其余1。新B：全部1。其余KL、softIoU保护及修正正则相同。已验证三组初始权重和记录批次相同。','','新A对旧对照检验正确影加权；新B对新A检验难例加权。结论仅覆盖当前网络和训练设置。','','|模型|地表IoU|云IoU|云影IoU|mIoU|','|---|---:|---:|---:|---:|']'''+s[end:]
s=s.replace("'B_conditional_on_minus_off',", "'A_minus_474A','B_minus_474A',")
s=s.replace("name.startswith(('A','B条件','B受保护'))", "name.startswith(('A取消','A受保护','B全部','B受保护'))")
s=s.replace('B同权重切换组合仅为条件干预；架构比较以独立训练A/B为准。','同权重开关仅为条件干预；损失权重比较以匹配训练为准。')
(T/'report_shadow475.py').write_bytes(s.encode())
