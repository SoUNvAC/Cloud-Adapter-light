"""Phase301 independent integer audit and data-driven result delivery."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
import os

R=Path(__file__).resolve().parents[2]
REMOTE=os.environ.get('PHASE301_REMOTE')=='1'
O=Path('/home/scv/shared/phase301_20261020/verified_run') if REMOTE else R/'outputs/phase301/phase301_20261020/verified_run'
S=R/'src/research_results/phase301_20261020/verified_run'
CLASSES=('Surface','Cloud','Shadow')

def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024**2),b''): h.update(part)
    return h.hexdigest()
def rows(name):
    with (O/name).open(encoding='utf-8-sig') as stream: return list(csv.DictReader(stream))
def write(name,data): (O/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def table(name,data):
    with (O/name).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
def number(row,key): return int(row[key])

def main():
    started=time.monotonic()
    assert not (O/'DELIVERY_SHA256.json').exists()
    acceptance=load(O/'CPU_ACCEPTANCE.json')
    resources=load(O/'RESOURCE_RECEIPT.json')
    assert acceptance['status']=='passed' and resources['status']=='completed'
    trans=rows('TABLE1_TRANSITIONS.csv');event=rows('TABLE2_ERROR_EVENTS.csv')
    lookup={(r['pool'],r['student'],r['truth'],r['old_prediction'],r['new_prediction']):number(r,'count') for r in trans}
    assert len(trans)==len(lookup)==108
    checks=0
    for row in event:
        pool,arm=row['pool'],row['student'];truth,wrong=row['error'].split('->')
        old=sum(lookup[pool,arm,truth,wrong,n] for n in CLASSES)
        new=sum(lookup[pool,arm,truth,p,wrong] for p in CLASSES)
        removed=sum(lookup[pool,arm,truth,wrong,n] for n in CLASSES if n!=wrong)
        added=sum(lookup[pool,arm,truth,p,wrong] for p in CLASSES if p!=wrong)
        corrected=lookup[pool,arm,truth,wrong,truth]
        for key,value in [('R4_errors',old),('student_errors',new),('removed',removed),('added',added),
                          ('net_reduction',old-new),('corrected_among_removed',corrected),('changed_to_other_error',removed-corrected)]:
            assert number(row,key)==value;checks+=1
        assert removed-added==old-new
    for filename in ('PER_GROUP_TRANSITIONS.csv','PER_PRODUCT_TRANSITIONS.csv'):
        totals=collections.Counter()
        for row in rows(filename):
            totals[row['pool'],row['student'],row['truth'],row['old_prediction'],row['new_prediction']]+=number(row,'count')
        assert dict(totals)==lookup
        checks+=108
    with (R/'src/research_results/phase300_20261020/TABLE1_FIXED_RESULTS.csv').open(encoding='utf-8-sig') as stream:
        original={(r['pool'],r['strategy']):json.loads(r['confusion']) for r in csv.DictReader(stream)}
    for pool in ('Catalogue','PixBox'):
        for arm in ('B','M'):
            for yi,y in enumerate(CLASSES):
                for pi,p in enumerate(CLASSES):
                    assert sum(lookup[pool,arm,y,p,n] for n in CLASSES)==original[pool,'R4'][yi][pi]
                    assert sum(lookup[pool,arm,y,n,p] for n in CLASSES)==original[pool,arm+'-PL'][yi][pi]
                    checks+=2
    boundary_support=rows('BOUNDARY_SUPPORT.csv')
    assert len(boundary_support)==32
    for row in boundary_support:
        assert number(row,'total')==number(row,'boundary')+number(row,'nonboundary')+number(row,'excluded');checks+=1
    strata_support=rows('PIXBOX_STRATA_SUPPORT.csv')
    for field in {r['field'] for r in strata_support}:
        assert sum(number(r,'reference_points') for r in strata_support if r['field']==field)==8569;checks+=1
    strata_cells=rows('PIXBOX_STRATA_TRANSITIONS.csv')
    for field in {r['field'] for r in strata_cells}:
        totals=collections.Counter()
        for r in strata_cells:
            if r['field']==field: totals['PixBox',r['student'],r['truth'],r['old_prediction'],r['new_prediction']]+=number(r,'count')
        assert dict(totals)=={k:v for k,v in lookup.items() if k[0]=='PixBox'};checks+=54
    train=rows('TABLE4_SAME512_TRAINING.csv');same=rows('SAME512_TRANSITIONS.csv')
    assert len(train)==96 and len(same)==864
    same_lookup={(r['product'],r['student'],r['attempt'],r['truth'],r['old_prediction'],r['new_prediction']):number(r,'count') for r in same}
    for r in train:
        p,a,step,y=r['product'],r['student'],r['attempt'],r['truth']
        old=sum(same_lookup[p,a,step,y,y,n] for n in CLASSES)
        new=sum(same_lookup[p,a,step,y,n,y] for n in CLASSES)
        kept=same_lookup[p,a,step,y,y,y]
        assert old==number(r,'initial_correct') and new==number(r,'current_correct')
        assert old-kept==number(r,'initial_correct_current_wrong') and new-kept==number(r,'initial_wrong_current_correct')
        assert new-old==number(r,'net_correct_change');checks+=5
        if step=='0': assert old==new==kept and number(r,'net_correct_change')==0
    group_events=[]
    for key in sorted({(r['pool'],r['student'],r['group'],r['products']) for r in rows('PER_GROUP_TRANSITIONS.csv')}):
        pool,arm,gid,products=key
        part=[r for r in rows('PER_GROUP_TRANSITIONS.csv') if (r['pool'],r['student'],r['group'])==(pool,arm,gid) and r['truth']=='Surface']
        added=sum(number(r,'count') for r in part if r['old_prediction']!='Cloud' and r['new_prediction']=='Cloud')
        removed=sum(number(r,'count') for r in part if r['old_prediction']=='Cloud' and r['new_prediction']!='Cloud')
        correct_to_cloud=sum(number(r,'count') for r in part if r['old_prediction']=='Surface' and r['new_prediction']=='Cloud')
        group_events.append(dict(pool=pool,student=arm,group=gid,products=products,net_new_Surface_to_Cloud=added-removed,correct_Surface_to_Cloud=correct_to_cloud))
    table('GROUP_EVENT_SUMMARY.csv',group_events)
    # Check every input against its actual frozen bytes after analysis.
    inputs=load(O/'INPUT_SHA256.json')
    for path,expected in inputs.items(): assert sha(path)==expected,path
    events={(r['pool'],r['student'],r['error']):r for r in event}
    boundary={(r['student'],r['region'],r['error']):r for r in rows('TABLE3_BOUNDARY.csv')}
    def cell(pool,a,y,p,n): return lookup[pool,a,y,p,n]
    lines=['# Phase301：错误转移与相同512上下文对照','','## 数据结果','',
      f"本次复核零训练、零神经前向、零下载；保留原301交付。Catalogue32组、PixBox16产品/10组；27格计数经{acceptance['independent_boolean_cell_checks']}次布尔交集复核，原混淆边缘和产品/组之和精确一致。CPU分析{acceptance['seconds']:.2f}秒、监测峰值{resources['peak_bytes']/1024**3:.3f}GiB。",'',
      '下表消除包含纠正及转成另一错误；新增包含原正确和另一错误进入该错误单元。净减少=消除−新增，不能当成纠错数。','',
      '|池|学生|错误类型|消除|新增|净减少|消除中真正改正|', '|---|---|---|---:|---:|---:|---:|']
    for pool in ('Catalogue','PixBox'):
        for error in ('Cloud->Shadow','Surface->Cloud','Surface->Shadow'):
            for arm in ('B','M'):
                r=events[pool,arm,error]
                lines.append('|'+ '|'.join([pool,arm,error]+[r[k] for k in ('removed','added','net_reduction','corrected_among_removed')])+'|')
    lines+=['','全部六类有向错误及27格零项见CSV；所有率以相应真值类支持为分母，零支持为NA。','']
    for arm in ('B','M'):
        r=events['Catalogue',arm,'Surface->Cloud']
        added=number(r,'added');correct=cell('Catalogue',arm,'Surface','Surface','Cloud')
        away=number(boundary[arm,'away_from_cloud_boundary','Surface->Cloud'],'added')
        edge=number(boundary[arm,'cloud_boundary_neighborhood','Surface->Cloud'],'added')
        g=[v for v in group_events if v['pool']=='Catalogue' and v['student']==arm]
        worst=max(g,key=lambda v:v['net_new_Surface_to_Cloud'])
        net=sum(v['net_new_Surface_to_Cloud'] for v in g)
        lines.append(f"- {arm}：新增Surface→Cloud中{correct:,}/{added:,}原本为正确Surface；非参考边界区{away:,}（{100*away/added:.1f}%），边界邻域{edge:,}，排除区{added-away-edge:,}。净增长出现在{sum(v['net_new_Surface_to_Cloud']>0 for v in g)}/32组，最大组{worst['products'].split('_T')[-1][:5]}占全池净增{100*worst['net_new_Surface_to_Cloud']/net:.1f}%。")
    lines+=['',f"参考边界分析有效{sum(number(r,'boundary')+number(r,'nonboundary') for r in boundary_support):,}像元，另排除{sum(number(r,'excluded') for r in boundary_support):,}；不改全体错误流。边界仅表示现有Cloud/非Cloud标签的邻接与半径2邻域，不代表薄云、物理云边界或投影。PixBox不做空间边界；全部原始代码/缺失项均保留，字段分层不可相加。",'',
      '|学生|同512第0步错误→末步正确|第0步正确→末步错误|净正确变化|', '|---|---:|---:|---:|']
    totals={}
    class_notes=[]
    for arm in ('B','M'):
        selected=[r for r in train if r['student']==arm and r['attempt']=='1000']
        repaired=sum(number(r,'initial_wrong_current_correct') for r in selected)
        damaged=sum(number(r,'initial_correct_current_wrong') for r in selected)
        net=sum(number(r,'net_correct_change') for r in selected)
        totals[arm]=net
        lines.append(f'|{arm}|{repaired}|{damaged}|{net}|')
        by_class=[sum(number(r,'net_correct_change') for r in selected if r['truth']==c) for c in CLASSES]
        class_notes.append(f"{arm}按Surface/Cloud/Shadow净正确变化：{by_class[0]:+d}/{by_class[1]:+d}/{by_class[2]:+d}。")
    lines+=['']+class_notes+['','## 结论分析','',
      '1. 实际转移同时包含Cloud→Shadow纠正和原正确Surface→Cloud恶化；后者多在参考非边界区，且集中于少数组。变化与Cloud判定扩大相容，不能归结为边界微调或指认为某一地物成因。',
      '2. 四固定裁剪同512视野的第0步→末步净正确变化见上表。完整教师→第0步同时含视野和非零初始化差异；第0步不是严格原R4裁剪。完整教师→末步的净纠错不能替代同视野训练收益。',
      '3. 唯一后续VFM假设：固定伪标签适配在纠正Cloud/Shadow混淆时，是否扩大Cloud判定、损伤原可靠Surface；限制此有向错误转移能否保留云影收益。先验证受控VFM Source的单seed基础对照，再决定约束或补seed，本次不启动下一phase。', '',
      '历史开发支持、单seed、V4预训练交集未知；继承Phase300的参考记录提前预览披露。结果不作为新盲测、VFM机制、跨传感器或标注口径偏移的因果证据；未新增地物判读，不改变Phase300指标、阈值、分组或缺类规则。', '',
      '完整27格、全部六类错误、逐产品/组、边界排除、原始字段分层、32探针与完整教师对照、SHA及验收均在同目录。']
    (O/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if REMOTE:
        import resource as os_resource
        peak=os_resource.getrusage(os_resource.RUSAGE_SELF).ru_maxrss*1024
    else:
        from run_phase300_local import usage
        peak=usage(os.getpid())
    assert peak<=16*1024**3 and time.monotonic()-started+resources['seconds']<=3600
    write('INDEPENDENT_ACCEPTANCE.json',dict(status='passed',checks=checks,complete_108_cells=True,
        all_product_group_sums_exact=True,phase300_both_confusions_reconstructed=True,
        boundary_partition_exact=True,metadata_field_supports_exact=True,zero_step_identity_verified=True,
        input_hashes_rechecked=len(inputs),seconds=time.monotonic()-started,peak_bytes=peak,
        finalizer_sha256=sha(Path(__file__)),neural_calls=0))
    write('EXECUTION_NOTES.json',dict(analysis='fresh verified_run; original Phase301 output and source snapshots retained',
        network='git synchronization only; CPU analysis reads existing local backups',training=0,neural_calls=0,downloads=0,
        old_phase300_unchanged=True,resource='16GiB / one hour / new1GB / free4GB; see resource receipt'))
    files={p.name:sha(p) for p in O.iterdir() if p.is_file()}
    write('DELIVERY_SHA256.json',files)
    if not REMOTE:
        S.mkdir(parents=True,exist_ok=True)
        for p in O.iterdir():
            if p.is_file(): shutil.copyfile(p,S/p.name)
        shutil.copytree(O/'logs',S/'logs',dirs_exist_ok=True)
        for name,digest in files.items(): assert sha(S/name)==digest,name
    print(json.dumps(dict(status='complete',checks=checks,input_hashes=len(inputs),files=len(files)+1,neural_calls=0)))

if __name__=='__main__': main()
