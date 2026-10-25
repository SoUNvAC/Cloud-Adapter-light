"""Editorial clarification only: retain raw report, all numerical tables, and previews."""
from phase308_common import *
def main():
    root=LOCAL/'remote_backup';assert load(LOCAL/'BACKUP_RECEIPT.json')['all_members_SHA_verified']
    raw=root/'RESULT.md';text=raw.read_text(encoding='utf-8')
    tables={p.name:sha(p) for p in S.glob('*.csv')}
    old='2. 支持：留出最大组后仍有6个D训练组，其中雪冰场景D仅1组。六波段D/K双方区间及重叠、逐维/任一超界在表中完整保留；这些是同类支持不足或外推线索，不能把单变量在范围内当作联合分布无偏移，也不能断言不可分。'
    new='2. 支持：最大组全部D落在训练D和K的逐维p1–p99内；包括RGB6Q平方特征及三模型标准化坐标，任一超界比例均0。因此没有本轮单变量范围证据支持“缺少可比D范围”。留出后训练D仍来自6组，但雪冰场景D仅1组/7595样本，场景代表性有限。D/K区间重叠与单变量在界内不能排除联合偏移、证明可分或作因果归因。'
    assert old in text;text=text.replace(old,new)
    old='3. 分数迁移：最大组全部D在三种固定模型中均p(K)>=0.5，因此固定读出保留Cloud、修复为0；SCORE_DISTRIBUTIONS和SIGNED_CONTRIBUTIONS保存逐特征有符号均值/分位数与截距。训练内分数只作对照，不当泛化成绩；贡献相关性不解释为物理因果。纯D组没有组内AUC证据，也未证明调阈值/校准能够解决。'
    new='3. 分数迁移：最大组D的MS6平均logit=1.615363，最小=1.112331；B04/B03/B02/B08/B11/B12有符号均值贡献依次为−6.440670/+5.455900/+5.432952/−3.887347/−0.089885/−1.323588，截距+2.468001。净分数仍为正，所以全部保留Cloud。RGB3/RGB6Q平均logit分别2.140926/2.219216，完整贡献均交表。MS6训练内D/K的加权平均logit为0.912818/2.657653，只作训练内描述。负SWIR贡献不能独自抵消其他项，也不证明因果或新阈值能解决；纯D仍无组内排序证据。'
    assert old in text;text=text.replace(old,new)
    text=text.replace('图像核查不能证明云雪像元真值，结论为无法判定。','三个原shapefile范围均为23040m×23040m，与record的1022像元/20m描述之间的裁剪或边缘关系未从README核验；这是原始地理溯源待核查疑点，不能据此断言缓存图像与mask错配。图像核查不能证明云雪像元真值，结论为无法判定。')
    (S/'RESULT.md').write_text(text,encoding='utf-8')
    review=dict(raw_report_sha256=sha(raw),published_report_sha256=sha(S/'RESULT.md'),
        purpose='Use actual zero-outside-bounds result and signed means, distinguish scene representation from univariate support; retain geographic provenance question.',
        numerical_CSV_unchanged=all(sha(S/n)==h for n,h in tables.items()),CSV_sha256=tables,
        original_image_audit_retained=True,preview_outcome='Panels and fixed NA crop inspected; no obvious cache alignment anomaly found; no new pixel ground truth inferred.',
        coordinate_outcome='存在待核查疑点：原shapefile范围与record像元尺度关系未核验；不是已证实错配',
        documentation_gap='README content unavailable in fixed environment; alternate definition/independence unknown; no downloads or environment changes.')
    assert review['numerical_CSV_unchanged'];write(S/'EDITORIAL_RECEIPT.json',review)
    notes=S/'phase308.md'
    original=notes.read_text(encoding='utf-8')
    if '出版核对' not in original:
        notes.write_text(original+'- 出版核对：所有数值CSV及原备份保持原SHA；澄清逐维超界均0、场景代表性与有符号净分数，记录原shapefile/像元尺度溯源疑点；原始RESULT与IMAGE_REFERENCE_AUDIT留存。\n',encoding='utf-8')
    summary=load(S/'SUMMARY_SHA256.json')
    summary['files']={p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file() and p.name!='SUMMARY_SHA256.json'}
    summary['code']={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase308*.py')}
    write(S/'SUMMARY_SHA256.json',summary)
    print('Editorial clarification published; raw report and all numerical tables preserved')
if __name__=='__main__':main()
