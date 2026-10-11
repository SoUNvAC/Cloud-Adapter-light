"""Accept the user's explicitly final draft, without fabricating missing fields/B."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from prepare_phase88 import ROOT, OUT, read, sha, write, csvwrite

FIELDS=['judgment','reason','small_sufficient','large_sufficient','needs_clear_date','needs_DEM','review_date']
DEST=OUT/'final_submission_20261011'


def validate(packet, lock):
    assert packet['stage']=='A' and packet['reviewer_type']=='human'
    assert packet['sampling_lock_sha256']==sha(OUT/'sample_input_lock.json')
    ids=[s['review_id'] for s in lock['samples']];rows=packet['rows']
    assert len(rows)==len(ids) and {r['review_id'] for r in rows}==set(ids)
    for r in rows:
        assert r.get('judgment','') in ['', 'Cloud','Shadow','Surface','混合','无法判定']
        for field in ['small_sufficient','large_sufficient','needs_clear_date','needs_DEM']:
            assert r.get(field,'') in ['', 'yes','no']
    return rows


def main(path, authorized):
    assert authorized, 'Requires explicit user final-submission designation'
    assert path.resolve().is_relative_to(ROOT)
    packet=read(path);lock=read(OUT/'sample_input_lock.json');rows=validate(packet,lock)
    assert not DEST.exists(), 'Never overwrite final human submission'
    DEST.mkdir()
    original=DEST/'expert_A_final_original.json'
    with original.open('xb') as f:f.write(path.read_bytes())
    assert sha(original)==sha(path)
    # Blank stays blank, including date/evidence flags; export time is not review date.
    csvwrite(DEST/'expert_A_final.csv',[dict(review_id=r['review_id'],**{k:r.get(k,'') for k in FIELDS}) for r in rows])
    a_lock=dict(accepted_utc=datetime.now(timezone.utc).isoformat(),
        authorization='2026-10-11 user explicitly designated expert_A_draft.json as final due to limited human resources',
        original_sha256=sha(original),csv_sha256=sha(DEST/'expert_A_final.csv'),
        sampling_lock_sha256=sha(OUT/'sample_input_lock.json'),browser_complete_flag=packet.get('complete'),
        reviewer_type=packet['reviewer_type'],reviewer_id=packet.get('reviewer_id',''),
        experience_as_submitted=packet.get('experience',''),exported_utc=packet.get('exported_utc',''),
        qualification='self-reported; remote-sensing expertise not independently established by role description',
        no_missing_fields_filled=True,stage_B_human_adjudication='not_executed',confirmation_ready_groups=0)
    # Lock A before any potential later reference access; this task does not reveal references.
    write(DEST/'FINAL_A_LOCK.json',a_lock)
    index={r['review_id']:r for r in rows};summaries=[]
    for alias in lock['aliases']:
        for stratum in ['P1','regular_control']:
            selected=[s for s in lock['samples'] if s['product']==alias['representative'] and s['phase88_stratum']==stratum]
            subset=[index[s['review_id']] for s in selected];counts=Counter(r.get('judgment','') for r in subset)
            summaries.append(dict(tile=alias['tile'],product=alias['representative'],stratum=stratum,selected=len(subset),
                reviewer_submissions=1,submitted_judgments=sum(v for k,v in counts.items() if k),unreviewed=counts[''],
                Cloud=counts['Cloud'],Shadow=counts['Shadow'],Surface=counts['Surface'],mixed=counts['混合'],unknown=counts['无法判定'],
                missing_reason=sum(not r.get('reason','').strip() for r in subset),
                missing_small_sufficient=sum(not r.get('small_sufficient','') for r in subset),
                missing_large_sufficient=sum(not r.get('large_sufficient','') for r in subset),
                missing_clear_date_flag=sum(not r.get('needs_clear_date','') for r in subset),
                missing_DEM_flag=sum(not r.get('needs_DEM','') for r in subset),missing_review_date=sum(not r.get('review_date','') for r in subset),
                possible_conflicts='NA_B_not_executed',confirmation_ready=0))
    csvwrite(DEST/'group_human_summary.csv',summaries)
    totals={t:{k:sum(s[k] for s in summaries if s['stratum']==t) for k in ['selected','submitted_judgments','unreviewed','Cloud','Shadow','Surface','mixed','unknown','missing_reason','missing_small_sufficient','missing_large_sufficient','missing_clear_date_flag','missing_DEM_flag','missing_review_date']} for t in ['P1','regular_control']}
    write(DEST/'summary.json',dict(strata=totals,unique_reviewers=1,qualification=a_lock['qualification'],
        submitted_judgments=sum(bool(r.get('judgment')) for r in rows),
        field_missing_counts={k:sum(not r.get(k,'').strip() for r in rows) for k in FIELDS},
        stage_B='not_executed; no original/revised conflict conclusions',confirmation_ready_groups=0))
    concerns=[dict(review_id=r['review_id'],human_judgment=r.get('judgment',''),human_comment=r.get('reason',''),
              disposition='原文保留；图像/传感器问题未被独立证实，未改标签或删除位置') for r in rows
              if any(w in r.get('reason','') for w in ['图有问题','伪影','烂图','图像质量','坏的','传感器'])]
    csvwrite(DEST/'human_image_concerns.csv',concerns)
    lines=['# Phase88最终人工A提交\n','## 1．数据结果\n',
        '用户因人力限制明确指定expert_A_draft.json为最终提交；原JSON字节及所有原始空项保留，不改浏览器complete=false。姓名／编号及经验均为自报，不将“主要负责人”自动认证为遥感专家资历。\n',
        '| 来源 | 判断提交 | Cloud | Shadow | Surface | 混合 | 无法判定 | 理由缺失 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for t in ['P1','regular_control']:
        s=totals[t];lines.append('|'+t+'|'+ '|'.join(str(s[k]) for k in ['submitted_judgments','Cloud','Shadow','Surface','mixed','unknown','missing_reason'])+'|')
    missing={k:sum(not r.get(k,'').strip() for r in rows) for k in FIELDS}
    lines += ['\n空项统计：'+json.dumps(missing,ensure_ascii=False)+'。空白不等于否；导出时间不作为逐点判读日期。实际看图行为无法从导出文件独立验证，这里报告的是提交的判断数。',
        '\n## 2．结论分析\n',
        '本轮按用户最终提交收口，不要求补字段或继续人工B。已取得人工A意见，但部分证据链与日期缺失，B阶段标签对照／裁决未执行，不能报告参考冲突率、专家复核通过或标签正确率。定向抽查不能估计全景准确率，57点不是57个独立场景。确认就绪仍0。',
        '\n对图像质量／疑似伪影／疑似传感器问题的原话单列保存，仅是主审提出的疑点，未证明成因。保留混合、未知和文字中的犹豫，不把单类选择自动解释成确定无疑。不改原标签、mask、成员或参考版本。',
        '\nFINAL_A_LOCK.json保存原始提交和CSV的SHA；group_human_summary.csv按九代表及两类来源分层。summary.json为字段缺失及计数，human_image_concerns.csv为原文疑点。旧材料包、空模板与Phase87记录不覆盖；不自动启动后续训练或评价。']
    with (DEST/'README.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    assert sha(original)==a_lock['original_sha256'] and sha(DEST/'expert_A_final.csv')==a_lock['csv_sha256']
    write(DEST/'DELIVERY_SHA256.json',{p.name:sha(p) for p in sorted(DEST.iterdir()) if p.is_file()})
    print(json.dumps(read(DEST/'summary.json'),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--file',type=Path,required=True);p.add_argument('--user-final',action='store_true');a=p.parse_args();main(a.file,a.user_final)
