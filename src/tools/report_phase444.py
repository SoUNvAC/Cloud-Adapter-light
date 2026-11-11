from pathlib import Path
import json,csv,numpy as np
R=Path('/home/scv/shared/phase444_retrain_20261103')
def main():
    rows=[]
    for arm in ['baseline429','source431']:
        v=json.loads((R/arm/'EVAL_COMPLETE.json').read_text());assert v['passed'];old,new=v['summary']
        for d in ['A','B']:
            a=old['metrics'][d];b=new['metrics'][d];row=dict(model=arm,dataset='CloudSEN535' if d=='A' else 'Catalogue32',target_label_supervision_percent=0,original_mIoU=a['mIoU'],aligned_retrained_mIoU=b['mIoU'],delta_mIoU=b['mIoU']-a['mIoU'])
            for i,c in enumerate(['surface','cloud','shadow']):row[c+'_original']=a['IoU'][i];row[c+'_retrained']=b['IoU'][i]
            rows.append(row)
    with (R/'MAIN.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Phase444：颜色对齐后源域40k重训','', '429与431均为0%目标标签监督；全新任务分支、官方预训练冻结DINO。两者均使用源训练原生BN，不做目标BN拟合。新权重仅按对齐源验证选择。','', '|模型|数据|原训练原图mIoU|对齐重训对齐图mIoU|差/点|影IoU原→新|','|---|---|---:|---:|---:|---:|']
    for v in rows:lines.append(f"|{v['model']}|{v['dataset']}|{v['original_mIoU']:.4f}|{v['aligned_retrained_mIoU']:.4f}|{v['delta_mIoU']:+.4f}|{v['shadow_original']:.4f}→{v['shadow_retrained']:.4f}|")
    lines+=['','映射复用Phase443两域无标签图像统计，属于使用目标图像外观信息的零目标标签实验。每个模型40k源训练；DINO主干保持冻结。完整三类指标见MAIN.csv，各模型原生BN；不混用历史目标BN校准结果。']
    (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(R/'COMPLETE.json').write_text(json.dumps(dict(passed=True,training_iters_per_model=40000,target_label_supervision_percent=0,rows=rows),indent=2))
if __name__=='__main__':main()
