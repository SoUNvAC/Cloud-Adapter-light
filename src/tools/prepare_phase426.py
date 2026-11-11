"""Generate isolated experiment orchestration from the accepted Phase425 harness."""
from pathlib import Path
import ast
T=Path(__file__).parent
s=(T/'controller_phase425.py').read_text().replace('425','426')
s=s.replace('phase426_relation','phase426_competition').replace('PhysicsMsRE as SparseMsREResidual','CompetitiveMsRE as SparseMsREResidual')
s=s.replace("('local_projection.','local_output.')","('router.',)")
s=s.replace("cfg.model.type='Phase413Segmentor'","cfg.model.type='Phase426Segmentor'")
start=s.index("   t=t.replace('return cfg',\"for loader")
end=s.index('   t=t.replace("return cfg",f',start)
s=s[:start]+s[end:]
s=s.replace("'phase411_msre.py','phase421_material.py','phase426_competition.py'","'phase411_msre.py','phase426_competition.py'")
s=s.replace("'PHASE413_MODE':mode","'PHASE426_MODE':mode")
(T/'controller_phase426.py').write_text(s,encoding='utf-8')
s=(T/'launch_phase425.py').read_text().replace('425','426').replace("'phase421_material.py',",'').replace('phase426_relation','phase426_competition')
(T/'launch_phase426.py').write_text(s,encoding='utf-8')
s=(T/'collect_phase425.py').read_text().replace('425','426')
(T/'collect_phase426.py').write_text(s,encoding='utf-8')
# Reuse independent prediction/AP/freeze audit; replace only experiment-specific comparisons.
s=(T/'report_phase425.py').read_text()
start=s.index(' historical=list(')
end=s.index(" save('REPORT_COMPLETE.json'",start)
s=s[:start]+''' a,b=endpoint['competitive'],endpoint['baseline']
 delta=a['mIoU']-b['mIoU']
 accepted={'passed':True,'checks':checks,'calls':sum(c['calls'] for c in costs),'updates':sum(c['steps'] for c in costs),'delta_mIoU':delta,'CPU_report_seconds':time.monotonic()-start}
 save('INDEPENDENT_ACCEPTANCE.json',accepted)
 writecsv(R/'COMPARISONS.csv',[{'comparison':'competitive - baseline','delta_mIoU':delta,'delta_shadow_IoU':a['Shadow_IoU']-b['Shadow_IoU'],'delta_macro_AP':a['macro_AP']-b['macro_AP']}])
 save('MECHANISM_SCREEN.json',{'delta_mIoU':delta,'clears_0_30pp':delta>=.30,'scope':'full package, no architecture-only attribution'})
 lines=['# Phase426 结果','','153 fit / 完整32 development，seed40901，固定1000步为主。','', '|方案|步数|mIoU|地表IoU|云IoU|阴影IoU|阴影宏AP|','|---|---:|---:|---:|---:|---:|---:|']
 for row in rows:
  m=js(R/row['mode']/f"evaluations/M/step_{row['step']}/METRICS.json")
  lines.append(f"|{row['mode']}|{row['step']}|{row['mIoU']:.4f}|{m['IoU'][0]:.4f}|{m['IoU'][1]:.4f}|{row['Shadow_IoU']:.4f}|{row['macro_AP']:.4f}|")
 lines+=['',f'固定1000步完整改造相对同轮基线：{delta:+.4f}个百分点。',f"共{accepted['calls']}次前向、{accepted['updates']}次更新，GPU阶段累计{sum(c['seconds'] for c in costs)/60:.2f}分钟。",'', '新增15参数；三组语义token＋竞争路由＋0.05辅助监督及在线困难度加权。共享优化器/梯度裁剪，收益或损伤不能单独归因架构。', '共同初始权重、逐步输入/增强/LR、冻结Source、完整预测混淆矩阵与AP复算通过。单seed开发集探索，未触及确认集。']
 (R/'RESULT.md').write_text('\\n'.join(lines)+'\\n',encoding='utf-8')
''' +s[end:]
(T/'report_phase426.py').write_text(s,encoding='utf-8')
for p in T.glob('*phase426*.py'):ast.parse(p.read_text(encoding='utf-8'))
print('Phase426 scripts parsed')
