"""Evidence-led documentary closure; original raster labels and history remain read-only."""
import re
import sys
import traceback
import numpy as np
from phase309_common import *
from fetch_phase309 import fetch
DIRECT='有直接证据已澄清';CONFLICT='证据有冲突';UNKNOWN='缺少证据仍未知'
def main():
    assert 'torch' not in sys.modules
    assert load(ROOT/'STATUS.json')['status']=='running'
    inputs=load(ROOT/'INPUT_SHA256.json')
    for n,h in inputs.items():assert sha(n)==h,n
    lock=load(ROOT/'analysis_LOCK.json')
    for n,h in lock['code'].items():assert sha(R/n)==h,n
    network=fetch();pages=load(ROOT/'PDF_EXTRACTION.json')['pages'];assert len(pages)==9
    text={i:(ROOT/'evidence/pages'/f'README_page{i:02d}.txt').read_text() for i in range(1,10)}
    evidence=[];readmehash=sha(D/'README.pdf')
    def item(question,page,lines,meaning,status,impact,terms=()):
        for term in terms:assert term.lower() in text[page].lower(),term
        evidence.append(dict(question=question,source_role='published version README',record_id=4172871,doi='10.5281/zenodo.4172871',source_SHA256=readmehash,page=page,lines=lines,evidence_path=f'evidence/pages/README_page{page:02d}.txt',source_URL='https://zenodo.org/records/4172871',meaning=meaning,status=status,not_stated_or_limit=impact))
    item('主参考类顺序与one-hot',4,'L0020-L0022','bool one-hot，最后维依次CLEAR/CLOUD/CLOUD_SHADOW；不能标阴影时仍保留第三通道',DIRECT,'Surface为本项目对CLEAR的名称；不把第三通道为空当无真实阴影',('exactly one True','CLEAR, CLOUD, CLOUD_SHADOW'))
    item('无效/混合/未标像元',4,'L0004-L0008,L0020-L0022','选取无no-data区域；参考为严格单标签one-hot，未定义混合/unknown第四类',DIRECT,'源影像无效值编码与填充值未逐一规定；项目额外finite/nonzero/fill过滤属于项目处理，非作者像元真值政策',('no-data','one-hot'))
    item('雪冰是否一律CLEAR',6,'L0042-L0047','作者承认高反射冰雪场景中clear冰架与冰云可能难区分，可能整景误判',UNKNOWN,'未给出所有雪/冰强制归CLEAR的通用规则；不能证明T25XEG全CLEAR正确或错误',('ice shelves','incorrectly'))
    item('snow/ice标签含义',2,'L0049-L0054 + p3 L0011-L0012','非互斥场景属性，视觉判断；snow/ice表示存在覆盖陆地或水面的雪冰',DIRECT,'不是逐像元地物或云雪真值，不能把标签扩展到每个损伤像元',('non-exclusive','subjective'))
    item('薄云与半透明边界',6,'L0033-L0038 + p3 L0025-L0029','thin标签指可透见地表信号；薄云边界不确定时倾向标CLOUD以求召回',DIRECT,'这是作者标注偏好，不是可恢复的统一光学厚度阈值',('if in doubt','cloud'))
    item('阴影缺标与地形阴影',1,'L0048-L0052 + p2 L0069','高起伏/地形阴影或边界模糊会导致无法标注；shadows_marked=0表示不能标，=1表示存在处已标',DIRECT,'未给出所有地形阴影的一律像元归类规则；0不能当真实无阴影',('terrain','89'))
    item('阴影边缘已知标注伪影',6,'L0022-L0029','RF可能在阴影与clear边缘产生细云线，人工修正但承认有遗漏',DIRECT,'不能将此总体已知问题未经逐景证据归因到三固定产品',('thin line','some were missed'))
    item('主/备mask来源与用途',4,'L0042-L0048 + p2 L0067-L0068','CALIBRATION10和VALIDATION50的双标场景保留另一个标注者mask；可核验标注一致性或备用',DIRECT,'是同源场景、同一类定义下的另一标注者结果，不是新的地面真值；MAIN不承诺双标',('both annotators','other annotators mask'))
    item('主/备mask编码可比性',4,'L0020-L0022,L0045-L0048 + p7 L0009-L0011','另一个标注者的同类mask用于逐像元类一致性统计，支持按同三类语义比较',DIRECT,'固定三产品无alt成员，实际交叉表NA；不通过数值相似性猜映射',('mask','statistical tests'))
    item('标注流程、人员与独立性',9,'L0017-L0034,L0048','IRIS半自动RF、人工检查与修正；CALIBRATION共同调整，VALIDATION平行且不相互咨询；Alistair Francis与John Mrziglod',DIRECT,'CALIBRATION非独立；VALIDATION无咨询说明作业分离，但共享工具/定义且不等于独立金标准；固定三组均MAIN',('without consulting','CALIBRATION'))
    item('标注一致性不是实际准确率',7,'L0004-L0006','作者明确一致性测量不是实际准确率',DIRECT,'备用标签不自动成为独立真值验证',('actual accuracy','relative consistency'))
    item('波段与TOA缩放',4,'L0004-L0013 + p8 L0026-L0030','13波段按数字序，B8A在8/9之间；整数/10000；非20m波段双线性重采样到20m；反射率可超过1',DIRECT,'确认既有索引[3,2,1,7,11,12]与RGB[3,2,1]，本轮不重采样/裁剪特征',('bilinear interpolation','10\'000'))
    item('1152至1022裁边链',9,'L0004-L0015 + p6 L0014-L0017','1152像元20m抽取窗，64像元每侧注释上下文，原定1024输出因边缘bug裁到1022',DIRECT,'支持尺寸来源及中心裁边推导，不等于提供逐景发布仿射、像元中心约定或原始L1C格网核验',('1152','64 pixels','1024','1022'))
    item('shapefile定义与实际范围',4,'L0026-L0030 + p9 L0004-L0015','正文称提取子场景多边形；固定三组实测均23040m=1152×20m，符合原始上下文窗',DIRECT,'不要把1152上下文外包框直接当1022输出边界；正文简述需结合附录限定',('polygon','subscene'))
    item('逐景原点/仿射/中心约定',9,'L0004-L0015（全文检索与DBF核查）','文档给裁边过程，shapefile给多边形/CRS，DBF仅FID；未发布可直接复核的逐景栅格仿射',UNKNOWN,'最终1022像元的四角/中心地理残差NA；不从边界尺寸倒推为已知地理配准参数')
    table(ROOT/'SEMANTIC_EVIDENCE.csv',evidence)
    shapes=load(ROOT/'SHAPEFILE_METADATA.json');members=load(ROOT/'FIXED_MEMBERS.json');listing=load(ROOT/'ARCHIVE_LISTINGS.json');audit=load(H/'IMAGE_REFERENCE_AUDIT.json')
    tags={r['scene']:r for r in read(D/'classification_tags.csv')};spatial=[];relation=[]
    for scene in shapes:
        pid=scene['product'];shp=next(m for m in scene['members'] if 'geometry' in m);prj=next(m for m in scene['members'] if 'WKT' in m);dbf=next(m for m in scene['members'] if 'attributes' in m)
        loX,loY,hiX,hiY=shp['bbox_xy_m'];width=hiX-loX;height=hiY-loY
        assert width==height==1152*20==23040 and 1022*20==20440 and 23040-20440==2600
        pts=np.asarray(shp['geometry'][0]['xy_m']);assert np.array_equal(pts.min(0),[loX,loY]) and np.array_equal(pts.max(0),[hiX,hiY])
        assert list(dbf['attributes']['rows'][0])==['deleted','FID']
        im=members['subscenes/'+pid+'.npy'];main=members['masks/'+pid+'.npy'];assert im['shape']==[1022,1022,13] and main['shape']==[1022,1022,3] and main['one_hot_all']
        spatial.append(dict(product=pid,CRS_WKT=prj['WKT'],units='metre',shape_SHA=shp['sha256'],bbox_xy_m=shp['bbox_xy_m'],width_m=width,height_m=height,array_shape=im['shape'],nominal_pixel_m=20,nominal_array_boundary_width_m=20440,arithmetic_difference_m=2600,original_window_pixels=1152,annotation_context_each_side_px=64,intended_output_px=1024,final_output_px=1022,origin=None,affine=None,pixel_center_convention=None,transform_source='not explicitly provided; DBF FID only',crop_evidence='README p9 L0004-L0015; p6 L0014-L0017',conditional_crop_inference='if centered64-pixel context and one-pixel edge removal:65px/1300m per side; documented-chain inference, not independently verified affine',four_corner_center_residuals=None,remaining_gap='no authoritative final-raster origin/affine or independent original-L1C coordinate test; array cache correspondence remains separate',status=DIRECT))
        alts=[n for n in listing['alt_masks.zip'] if pid in n];assert not alts
        assert tags[pid]['dataset']=='MAIN'
        relation.append(dict(product=pid,main_member='masks/'+pid+'.npy',main_SHA=main['sha256'],main_shape=main['shape'],main_encoding='bool one-hot CLEAR,CLOUD,CLOUD_SHADOW; README p4 L0020-L0022',main_valid_count=1022**2,alt_member=None,alt_SHA=None,alt_purpose='other annotator on CALIBRATION/VALIDATION; README p4 L0045-L0048',dataset_stage=tags[pid]['dataset'],comparable=False,reason='fixed MAIN product has no alternate member',common_valid=None,main_only_valid=None,alt_only_valid=None,cross_3x3_path=None,independence='NA: no alt for this product; shared annotation process is not independent gold',conclusion='no per-product alternative-label corroboration; retain primary reference'))
    table(ROOT/'SPATIAL_PROVENANCE.csv',spatial);table(ROOT/'REFERENCE_RELATION.csv',relation)
    table(ROOT/'REFERENCE_CROSS_COUNTS.csv',[dict(product=r['product'],main_class=None,alt_class=None,count=None,status='NA',reason=r['reason']) for r in relation])
    issues=[
        ('snow_ice_classification','场景标签不证明像元真值；政策未读','README p2/p3明确标签属性，p6明确冰雪/云语义模糊',UNKNOWN,'不改历史数值；明确无一律雪冰CLEAR规则，最大组全CLEAR可属于冰雪地表解释但未获像元真值证明'),
        ('shadows_not_marked','阴影缺标口径待补','README p1及p2 shadows_marked定义',DIRECT,'0=不能标，不是没有阴影；地形阴影详细类分配仍未知，不改支持'),
        ('alt_purpose','编码/用途unknown','README p4另标注者、p9分阶段流程',DIRECT,'用途已澄清；固定三组均MAIN且无alt，不追加有利产品'),
        ('alt_independence','独立性unknown','CALIBRATION共同协商，VALIDATION不咨询；p7一致性不是真实准确率',DIRECT,'作业分离不等于独立金标准；固定三组无独立复核，参考正确性仍未知'),
        ('spatial_size_difference','23040m与1022×20的关系待查','README p9原始1152/64边缘/1024至1022，实测三组边界23040m',DIRECT,'上下文窗解释尺寸来源，不能再把2600m算术差直接叫错位；未恢复逐景最终仿射'),
        ('cache_vs_geography','同索引缓存对应通过，地理溯源未定','项目读取代码无地理重采样；DBF仅FID，未含最终栅格仿射',UNKNOWN,'维持同数组索引结论；不宣称已独立复核原始L1C四角/中心'),
        ('target_all_CLEAR','T25XEG共同支持全Surface','README主三类定义、原tags MAIN/snow1/clear100；冰雪场景不确定性',UNKNOWN,'编码及元数据自洽，未证明每像元正确；不得凭RGB改标签或性能'),
    ]
    table(ROOT/'ISSUE_DISPOSITION.csv',[dict(issue=i,phase308_status=old,new_evidence=new,current_decision=status,historical_values_changed=False,impact_and_remaining_limit=impact) for i,old,new,status,impact in issues])
    # Bibliographic facts and original project source remain distinct from direct publication evidence.
    index=[dict(evidence_id=f'README_p{p["page"]}',role='published file',version='4172871; local SHA exact',URL='https://zenodo.org/records/4172871',path=p['text'],SHA=p['text_sha256'],page=p['page']) for p in pages]
    index += [dict(evidence_id='record',role='publication metadata',version='4172871',URL='https://doi.org/10.5281/zenodo.4172871',path='evidence/record.json',SHA=sha(ROOT/'evidence/record.json'),page=None),dict(evidence_id='project_reader',role='project processing code, not author annotation policy',version=sha(R/'src/tools/phase302_images.py'),URL=None,path='evidence/project_readers/phase302_images.py',SHA=sha(R/'src/tools/phase302_images.py'),page=None)]
    table(ROOT/'EVIDENCE_INDEX.csv',index)
    for n,h in inputs.items():assert sha(n)==h,n
    assert (ROOT/'PRODUCTION_PACKAGES_BEFORE.json').read_bytes()==(ROOT/'PRODUCTION_PACKAGES_AFTER.json').read_bytes()
    write(ROOT/'CPU_ACCEPTANCE.json',dict(products=3,pages=9,semantic_questions=len(evidence),issues=len(issues),all_reference_rows_including_missing=True,all_space_dimensions_in_metres=True,unknown_not_negation=True,no_hypothesis_written_as_affine=True,four_corner_center_residuals_NA_no_complete_transform=True,main_alt_fixed3_missing=True,historical_SHA_unchanged=True,production_packages_unchanged=True,fits=0,training_updates=0,neural_calls=0,performance_recalculations=0,network_documents=len(network),download_bytes=load(ROOT/'NETWORK_EVIDENCE.json')['total_download_bytes']))
    write(ROOT/'FAILURES.json',[dict(kind='optional_primary_source_network',**r) for r in network if 'failure' in r])
    report='''# 数据结果

原版本README（SHA fe30cf507184ee3f3cf16e16bce271b0701763d00144b138e0b3edf075e50ce6）9页全文已用隔离pypdf提取并逐页渲染。15项语义问题、3产品空间表、3产品主备关系和7项闭环表均附页码/行号；原Phase308文件与项目读取代码SHA保持不变。文档工具仅安装在本轮toolenv，训练环境包清单前后一致；零拟合、训练、神经前向与历史性能重算。

1. **主参考与场景语义。** p4明确1022×1022×3、bool严格one-hot，通道为CLEAR/CLOUD/CLOUD_SHADOW。项目Surface对应CLEAR。p2–3明确snow/ice是非互斥、主观视觉场景属性；没有“所有雪冰都归CLEAR”的通用规则。p6承认高反射冰雪与冰云可能难辨，甚至可能整景错标。T25XEG原全CLEAR与snow/ice=1在编码和元数据上自洽，仍不能证明像元真值正确。
2. **阴影与薄云。** p2的shadows_marked=0表示无法标，=1表示存在处已标；保留第三通道不代表真实无阴影。p1指出地形/边界歧义造成缺标，但未逐一规定地形阴影的像元归类。p3/p6给薄云可透见地表的标签描述，并说明边界犹疑时倾向标云以求召回；不能据此恢复统一物理阈值。
3. **alt_masks。** p4与附录B说明其为CALIBRATION10及VALIDATION50双标场景的另一位标注者结果；CALIBRATION共同调整，VALIDATION无相互咨询，但共享工具和类定义，p7明确一致性并非实际准确率。固定T25XEG/T40XDR/T35RPM均MAIN，归档清单均无对应alt成员，因此共同/独有有效支持和3×3交叉数全部NA；没有改挑其他场景。用途已澄清，这3组仍没有备用标注复核，更没有独立金标准。
4. **空间范围。** 三shapefile均为23040m×23040m的轴对齐多边形，CRS/WKT与几何/DBF原样保留；DBF仅FID。p9直接记载1152×1152、20m抽取上下文窗，注释保留每侧64像元，原定1024输出因边缘bug成为1022；p6也说明末端裁边。23040=1152×20，最终名义宽20440=1022×20，2600m差值因而有文档支持的窗口链解释。p4笼统“提取子场景边界”须结合附录，不能直接将原窗口多边形当作最终输出边界。

# 结论分析

Phase308的三项资料缺口中，主类顺序/阴影缺标/薄云偏好、备用标注的用途，以及空间尺寸差异的上下文窗口来源已获得直接证据。雪冰逐像元政策与T25XEG参考实际正确性仍未知。1152→1024→1022支持居中去64再去边缘1像元的65像元/1300m每侧推导，但这不是发布的逐景最终栅格仿射；文档未明确给出最终原点、行列方向/像元中心约定，亦未独立读取原始L1C核对，因此四角/中心坐标残差NA，不尝试候选平移或重新配准。

可以继续使用：既有数组同索引对应、固定通道和原参考/历史数值。须披露：上下文边界与最终输出范围有别，最终地理变换未独立核验；场景标签不是像元真值；半自动标注有已知歧义且三固定组无alt复核。没有发现需在本轮修改历史文件的确切错误。Phase307有限多光谱增量、最大损伤仍未解决的数值边界不变，本轮不产出SFDA增益。未决事项在此封存为限制，不自动重标、训练、部署或开启下一phase。
'''
    (ROOT/'RESULT.md').write_text(report,encoding='utf-8')
if __name__=='__main__':
    try:main()
    except BaseException:write(ROOT/'FAILURES_ANALYSIS.json',dict(traceback=traceback.format_exc(),no_blind_retry=True));raise
