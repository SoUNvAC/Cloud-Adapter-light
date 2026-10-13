"""Local-only reference join and aggregate readiness; never scores or predictions."""
import collections
import shutil
from phase104_common import *

OUT=REPO/'outputs/phase104/phase104_20261013'
DATA=REPO/'data/pixbox_s2_cmix_5036991/phase104_20261013'
REFERENCE=REPO/'data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json'
REFERENCE_SHA='b1448f3fbd05c9e77af7ed0ea50dbde4be7480a35a4c532a2fec45789af25f96'

def main():
    assert sha(REFERENCE)==REFERENCE_SHA
    with (DATA/'point_validity_no_labels.csv').open(encoding='utf-8-sig',newline='') as f:validity=list(csv.DictReader(f))
    assert len(validity)==9500
    acceptance=read(OUT/'input_acceptance.json');cpu=read(OUT/'CPU_48_verification.json')
    downloads=read(OUT/'download_status.json');budget=read(OUT/'transfer_budget.json')
    manifest=read(RESULT/'object_manifest.json');grid=read(OUT/'grid_acceptance.json')
    runtime=read(OUT/'runtime_chain_acceptance.json')
    assert downloads['status']=='all_files_verified' and len(downloads['files'])==96
    assert budget['actual_received_bytes']<=budget['reserved_bytes']<=CAP
    for obj in manifest['objects']:
        received=downloads['files'][obj['path']]
        assert all(received[k]==v for k,v in obj.items()) and received['verified'] and len(received['sha256'])==64
    assert grid['all_grids_match_original_XML'] and len(grid['rows'])==96 and runtime['status']=='passed'
    for r in grid['rows']:
        assert r['sha256']==downloads['files'][r['product_id']+'/'+r['band']+'.jp2']['sha256']
    assert sha(DATA/'point_validity_no_labels.csv')==acceptance['point_validity_sha256']
    assert cpu['status']=='passed' and len(cpu['rows'])==48
    assert cpu['point_inputs_sha256']==acceptance['point_inputs_sha256']
    assert [r['point_id'] for r in cpu['rows']]==read(RESULT/'task_lock.json')['CPU_point_ids']
    assert sha(OUT/'window_index.json')==acceptance['window_index_sha256']
    assert sha(OUT/'grid_acceptance.json')==acceptance['grid_acceptance_sha256']
    reference=read(REFERENCE);byid={}
    for r in reference:
        assert len(r['point_ids'])==1
        assert r['point_ids'][0] not in byid;byid[r['point_ids'][0]]=r
    assert set(byid)=={r['point_id'] for r in validity}
    groups=collections.defaultdict(list);products=collections.defaultdict(list);joined=[]
    for r in validity:
        ref=byid[r['point_id']];assert r['product_id']==ref['product_id'] and r['group_id']==ref['group_id']
        assert (int(r['row20']),int(r['col20']))==(int(ref['row20']),int(ref['col20']))
        assert r['input_valid'] in ('True','False')
        p=dict(**r,reference=ref['reference'],accepted=(r['input_valid']=='True' and ref['reference']!='ignore'))
        joined.append(p);groups[r['group_id']].append(p);products[r['product_id']].append(p)
    def aggregate(key,rr):
        count=lambda name,after:sum(r['reference']==name and (not after or r['input_valid']=='True') for r in rr)
        cells=collections.Counter((r['product_id'],r['row20'],r['col20']) for r in rr)
        retained=collections.Counter((r['product_id'],r['row20'],r['col20']) for r in rr if r['accepted'])
        loss=collections.Counter(reason for r in rr if r['input_valid']=='False' for reason in json.loads(r['invalid_reasons']))
        pos=count('positive_ground_cloud_shadow',True);sn=count('negative_surface',True);cn=count('negative_cloud',True)
        return dict(key=key,tiles=';'.join(sorted({r['product_id'].split('_T')[1][:5] for r in rr})),
            products=len({r['product_id'] for r in rr}),raw_points=len(rr),
            positive_before=count('positive_ground_cloud_shadow',False),positive_after=pos,
            surface_negative_before=count('negative_surface',False),surface_negative_after=sn,
            cloud_negative_before=count('negative_cloud',False),cloud_negative_after=cn,
            negative_after=sn+cn,reference_ignored=sum(r['reference']=='ignore' for r in rr),
            input_invalid=sum(r['input_valid']=='False' for r in rr),
            reference_valid_input_lost=sum(r['reference']!='ignore' and r['input_valid']=='False' for r in rr),
            loss_reasons_nonexclusive=json.dumps(dict(loss),sort_keys=True,separators=(',',':')),
            all_20m_cells=len(cells),all_multisample20m_cells=sum(n>1 for n in cells.values()),
            retained_20m_cells=len(retained),retained_multisample20m_cells=sum(n>1 for n in retained.values()),
            minimum_support_met=pos>=5 and sn+cn>=20,AP_defined=pos>0 and sn+cn>0)
    table=[aggregate(k,v) for k,v in sorted(groups.items())];product_table=[aggregate(k,v) for k,v in sorted(products.items())]
    assert len(table)==10 and len(product_table)==16
    csvwrite(RESULT/'per_group_input_support.csv',table);csvwrite(RESULT/'per_product_input_support.csv',product_table)
    csvwrite(DATA/'reference_input_join_private.csv',joined)
    eligible=sum(r['minimum_support_met'] for r in table)
    summary=dict(status='input_acceptance_complete' if eligible>=8 else 'stopped_insufficient_input_support',
        groups=10,products=16,points=9500,positive_before=379,negative_before=8190,
        positive_after=sum(r['positive_after'] for r in table),negative_after=sum(r['negative_after'] for r in table),
        input_invalid=sum(r['input_invalid'] for r in table),reference_valid_input_lost=sum(r['reference_valid_input_lost'] for r in table),
        minimum_support_groups=eligible,minimum_required_groups=8,minimum_positive=5,minimum_negative=20,
        future_primary_eligible=eligible>=8,Phase105_authorized=False,
        all_multisample20m_cells=sum(r['all_multisample20m_cells'] for r in table),
        retained_multisample20m_cells=sum(r['retained_multisample20m_cells'] for r in table),
        local_reference_sha256=REFERENCE_SHA,no_point_labels_transferred=True,
        local_join_sha256=sha(DATA/'reference_input_join_private.csv'),
        group_support_sha256=sha(RESULT/'per_group_input_support.csv'),
        point_inputs_remote_sha256=acceptance['point_inputs_sha256'],
        new_fits=0,network_forwards=0,performance_evaluations=0)
    write(RESULT/'support_acceptance.json',summary,True)
    write(RESULT/'transfer_and_disk_cost.json',dict(expected_payload_bytes=EXPECTED_BYTES,
          observed_received_bytes=budget['actual_received_bytes'],charged_upper_bound_bytes=budget['reserved_bytes'],
          cumulative_cap_bytes=CAP,one_remote_imagery_copy=True,no_dense_float_cache=True,
          disk_before=read(OUT/'disk_before.json'),disk_after_download=read(OUT/'disk_after_download.json'),
          host_free_is_user_report_and_remaining_is_estimate=True),True)
    table_md='\n'.join(f"|{r['tiles']}|{r['positive_before']}→{r['positive_after']}|{r['surface_negative_before']+r['cloud_negative_before']}→{r['negative_after']}|{r['input_invalid']}|{'是' if r['minimum_support_met'] else '否'}|" for r in sorted(table,key=lambda r:r['tiles']))
    text=f'''# Phase104 输入验收\n\n## 1 数据结果\n\n16原产品、96六波段对象：8,344,418,736bytes；固定generation、长度、镜像MD5及实际SHA256逐件通过。全部96栅格与原XML格网一致；9500点逐点检查，输入无效{summary['input_invalid']}点，其中参考有效点损失{summary['reference_valid_input_lost']}点。48固定哈希抽样点独立CPU原DN、TOA、坐标及窗口复算通过。\n\n|相关组tile|正类验收前→后|负类验收前→后|输入无效点|≥5正/20负|\n|---|---:|---:|---:|---|\n{table_md}\n\n最低支持门：{eligible}/10组满足，要求至少8组。19GBQ/48NUG保留，未填零AP。原9500点共有9484个20m单元、14个多点单元；同单元后续共享分数，仍是采样点参考，不扩展成20m真值。没有载入模型、拟合、网络前向或AP/FPR/mIoU评价。\n\n## 2 结论与边界\n\n{'输入验收及预定支持门通过；可另立Phase105协议。' if eligible>=8 else '输入损失后预定支持门不足，暂停后续主要评价资格。'} 本轮不自动启封性能评价。后续唯一主要比较为原Phase93 G7_153−G6_153逐组平均采样点AP；10000次配对相关组bootstrap、seed105013，95%描述区间下界>0且宽度≤20pp。\n\n该比较只回答Source分数的条件增量；不能替代光谱增量方向，不能恢复旧失败分支。选择性采样、保守参考语义及未知基础预训练独立性限制保留；不推断全场景FPR、尾部安全性或密集mIoU。不用这10组做校准。\n\n影像远端唯一保存；本地只接收共享数据目录的无标签索引和小型验收报告。逐点参考与交集仍仅在本地受控data目录，不进Git。上下文按1020窗口冻结为可重复惰性读取索引，避免宿主物理盘重复保存整景浮点缓存；尚未进行512网络推理。\n'''
    (RESULT/'EVALUATION_READINESS.md').write_text(text,encoding='utf-8',newline='\n')
    (RESULT/'README.md').write_text('# Phase104\n\n已完成固定PixBox原影像输入验收，零拟合、零前向、零性能评价。见EVALUATION_READINESS.md；逐组原始支持见per_group_input_support.csv。影像与逐点标签均不进入Git。Phase105需另立协议与授权。\n',encoding='utf-8',newline='\n')
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in ('.json','.md','.csv') and not (RESULT/p.name).exists():shutil.copyfile(p,RESULT/p.name)
    for p in RESULT.iterdir():
        if p.is_file():shutil.copyfile(p,OUT/p.name)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
