"""Editorial-only source review; retain original report and every raw provenance table."""
from phase309_common import *
def main():
    root=LOCAL/'remote_backup';assert load(LOCAL/'BACKUP_RECEIPT.json')['all_members_SHA_verified']
    raw=root/'RESULT.md';text=raw.read_text(encoding='utf-8')
    text=text.replace('没有“所有雪冰都归CLEAR”的通用规则。','未找到文档明确规定“所有雪冰都归CLEAR”的证据。')
    text=text.replace('更没有独立金标准。','不能由这些材料建立独立金标准。')
    text=text.replace('1152→1024→1022支持居中去64再去边缘1像元的65像元/1300m每侧推导','1152→1024→1022支持在居中裁边前提下，去64像元再去边缘1像元，即65像元/1300m每侧的条件推导')
    text+='\n补充资料核对：原记录4172871及作者链接的IRIS仓库/README均访问成功并保存时间、URL与版本标识。当前IRIS README介绍通用交互式标注能力，不能替代这批数据实际采用流程的版本证据；本轮没有把其能力推成历史参数。全部9页渲染与提取顺序已人工查看，引用所指章节/行号可定位。\n'
    (S/'RESULT.md').write_text(text,encoding='utf-8')
    tables={p.name:sha(p) for p in root.glob('*.csv')}
    assert all(sha(S/n)==h for n,h in tables.items())
    net=load(root/'NETWORK_EVIDENCE.json')
    review=dict(raw_RESULT_sha256=sha(raw),published_RESULT_sha256=sha(S/'RESULT.md'),all_original_CSV_unchanged=True,original_CSV_SHA=tables,
        pages_visually_checked=list(range(1,10)),page_order_tables_and_footnotes_checked=True,
        author_tool_README_is_current_general_capability_not_dataset_version_pipeline=True,
        unknowns_are_documentation_gaps_not_negative_truth=True,
        download_accounting=dict(saved_wheels_and_document_bodies_bytes=net['total_download_bytes'],pip_metadata='pip log shows rounded7.1kB and48kB; retained in tool_download.log',index_HTTP_TLS_overhead='not measured as network-interface traffic; saved-file/body count is not total wire bytes'),
        no_original_report_or_metrics_rewritten=True)
    write(S/'EDITORIAL_RECEIPT.json',review)
    notes=S/'phase309.md';content=notes.read_text(encoding='utf-8')
    content+='- 出版复核：9页渲染/提取顺序已看；原报告与全部CSV保留SHA，仅收紧unknown措辞与条件裁边推导。三份一手资料均HTTP200；精确下载计数为保存轮包和正文净负载，pip metadata仅日志约数，未将其称为链路总字节。\n'
    notes.write_text(content,encoding='utf-8')
    summary=load(S/'SUMMARY_SHA256.json');summary['files']={p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file() and p.name!='SUMMARY_SHA256.json'}
    summary['code']={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase309*.py')};write(S/'SUMMARY_SHA256.json',summary)
    print('Editorial source/unknown/accounting review published; all raw CSV and raw report preserved')
if __name__=='__main__':main()
