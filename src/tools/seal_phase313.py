"""Freeze metadata-only decision and conditional future protocol. Never train or project."""
from datetime import datetime,timezone
import resource
import shutil
import subprocess
import sys
import time
from phase313_common import *

def main():
    assert sys.platform=='linux';start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    assert not (ROOT/'DECISION_LOCK.json').exists()
    state=load(ROOT/'STATUS.json');a=load(ROOT/'INVENTORY_ACCEPTANCE.json');assert (ROOT/'REVIEW_LOCK.json').exists()
    assert a['products']==132 and a['B11_verified']==3 and a['ready']==0 and a['approximate_only']==3 and a['missing']==129
    before=load(ROOT/'INPUT_SHA256.json')
    for path,h in before.items():assert sha(path)==h,path
    rows=read(ROOT/'INPUT_READINESS.csv');assert len(rows)==len({r['product'] for r in rows})==132
    assert all(r['Source_checkpoint_sha256']==SOURCE for r in rows)
    prior70=load(S/'prior_inputs/phase70_report.json');assert prior70['promotion']['selected']==[]
    version=(ROOT/'literature/ESA_ATBD_page001.txt').read_text()
    (ROOT/'SOURCE_VERSIONS.txt').write_text('ESA acquired PDF first-page version evidence:\n'+version+'\nFmask: Zhu, Wang, Woodcock, RSE159 (2015)269-277, DOI10.1016/j.rse.2014.12.014; author-hosted PDF SHA in NETWORK.json.\nCopernicus processor webpage: TLS handshake timeout, content/version unavailable; no claims attributed.\n',encoding='utf-8')
    prior=[
      dict(source='Phase70',verified='153fit/32dev; 61x61 local contrast plus interaction/band ratios; matched shuffle controls',result='A/B/C all ineligible; B recall below L3; selected=[]; metadata D not_ready',reusable='original report promotion and metadata locks',new='current VFM caches provide model assets, not a reversal of failed gate',sufficient='no'),
      dict(source='Phase74',verified='SourceCloud exclusion from61x61 B08/B11/B12 mean; <16neighbors fallback; global count-matched randomR',result='E refit FPR0.889% vsB/R0.734%; E-R AP+0.513pp CI[-1.707,2.939]; continue condition failed',reusable='old failure and E/R support limitations retained',new='direction/height/context hypothesis differs from nondirectional exclusion; no benefit measured',sufficient='no; do not reopen exclusion rule'),
      dict(source='Phase75',verified='exact3original L1C XML; tile mean sun/coarse angle grids and native10/20/60m grids',result='metadata ready; full cube mapping pending; reparse corrected long granule/compact folder link',reusable='6XML plus original/reparse locks; all3intersect currentfit',new='explicit132pool intersection and cache inventory',sufficient='no; old public-positive-shadow/hash selection biased'),
      dict(source='Phase76/77',verified='3B11 coordinate chains independently tested; Phase76 B04 fails; Phase77 0/12 sampling candidates qualify',result='B11_verified_B04_pending / RGB unresolved; no extra candidate search permitted',reusable='phase76_20261010/reports/summary.json and mapping_lock; old77 summary/candidate_lock',new='recover existing evidence missed by phase75-only inventory; not new registration',sufficient='B11 only approximate subset; not RGB/all-band geometry ready'),
      dict(source='Phase309',verified='README1152context -> intended1024 -> actual1022; footprint23040m',result='65inset remains conditional from documentation alone; no universal affine',reusable='RESULT,SPATIAL_PROVENANCE,SHAPEFILE_METADATA and documentSHA',new='reconciles footprint discrepancy; does not erase narrower oldB11 pixel evidence',sufficient='no; preserve product/band scope'),
      dict(source='Phase304/current132',verified='Source caches132; fullQ90-PL student scores32; fitteacher pseudo100; exact SourceSHA and ninewindow means',result='metadata/header availability only;100fit student fullmaps absent; not a method comparison',reusable='FIT/EVAL/PSEUDO/EVALUATION/ENDPOINT locks and headers',new='VFM/SFDA assets plus testable directional placebo design',sufficient='no; backbone change/genericloss alone insufficient contribution evidence')]
    table(ROOT/'PRIOR_WORK_DELTA.csv',prior)
    network=load(ROOT/'NETWORK.json');urls={r['object']:r['original_url'] for r in network['requests']};ver='acquired PDF version/header in SOURCE_VERSIONS.txt; fileSHA/UTC in NETWORK.json'
    physical=[
      dict(component='geometry plus radiometry',source=urls['ESA_ATBD'],section='3.4.1-3.4.5 PDF38-40',version=ver,support='Sen2Cor combines radiometric dark candidates with cloud/sun/height geometry',assets='Source masks available; no validated all-band mapping or height prior',limits='water/terrain dark ambiguity; unobserved cloud context',forbidden='neighbor/morphology is not physical pairing'),
      dict(component='sun / height / view / context',source=urls['ESA_ATBD'],section='3.4.2 and3.4.5 PDF39-40',version=ver,support='tile sun and height distribution; viewing/parallax and outside-tile cloud restrict approximation',assets='3tile mean angles/coarse grids; no cube interpolation or north conversion',limits='tile means are not pixel angles; geographicnorth differs from UTMgridnorth and row direction',forbidden='assign solar-to-rowcol displacement before validated orientation'),
      dict(component='horizontal-plane dependency',source='geometric derivation; consistency with ATBD3.4.2, not a quoted executable formula',section='right-triangle approximation',version='Phase313 analytic assumption only',support='distance approximately h*tan(solar_zenith), opposite solar direction; view and terrain neglected',assets='no admissible current height/context prior frozen',limits='must correct cloud parallax/terrain, local vs gridnorth; no projection executed',forbidden='fixedheight is not trueheight; no rowcol formula or mask frozen'),
      dict(component='Fmask matching and Sentinel2 height assumption',source=urls['FMASK2015'],section='2.1 cloud matching /2.2.2 PDF5-6; discussionPDF8',version='Zhu Wang Woodcock2015 DOI10.1016/j.rse.2014.12.014',support='object cloud-shadow matching; Sentinel2 prototype uses flatcloud and a fixed height search assumption',assets='no thermal-derived height; published historical prior is not validated for current132',limits='prototype not evaluated on realSentinel2 in this paper; matching may miss thickcloud shadow shape',forbidden='do not import prototype height range or matching gains as current truth'),
      dict(component='Copernicus processor version gap',source=urls['COPERNICUS_PROCESSOR'],section='not retrieved',version='NA TLS handshake timeout preserved',support='NA',assets='no content read, zero responsebody bytes',limits='cannot verify present implementation/version',forbidden='no fabricated current processor algorithm claim'),
      dict(component='unpaired and dark ambiguity',source=urls['ESA_ATBD'],section='3.4.1/3.4.5; own identifiability consequence',version=ver,support='cloud outside tile and dark background can defeat local matching',assets='published cropped1022context only; externalregistered clouds absent',limits='missedcloud/occlusion/contextmissing are alternative explanations',forbidden='unpairedShadow cannot become Surface/Cloud negative; retain unknown')]
    table(ROOT/'PHYSICAL_ASSUMPTIONS.csv',physical)
    missing=[r['product'] for r in rows if r['exact_original_XML_status']=='missing'];write(ROOT/'MISSING_XML_PRODUCTS.json',dict(products=missing,count=len(missing),no_requests_executed=True))
    exact=[r['product'] for r in rows if r['B11_mapping_status']=='ready']
    actions=[
      dict(gap='RGB preprocessing unresolved for exact3fit',objects=exact,existing=str(BASE/'phase76_20261010/reports/summary.json')+'; '+str(BASE/'phase77_20261011/summary.json'),minimal_action='First obtain author preprocessing implementation/grid definition if available; compare to frozen old evidence. Do not download same sixJP2 or repeat12failed variants. Stop if provenance not obtainable.',estimated_requests='0 now; future at most1author/code provenance request after separate authorization',estimated_bytes='0 now; future<=1MB textual code/config estimate, not measured',estimated_CPU='<=10min metadata review estimate; no new registration now',acceptance='independent originalRGB grid/pixelcenter/radiometry chain with prespecified checks; no residual-driven search or tolerance change',control='required before correct/opposite direction controls',executable_now='no'),
      dict(gap='exact XML/native grids for remaining129',objects='MISSING_XML_PRODUCTS.json exact129 identifiers; INPUT_READINESS preserves fit/dev',existing=str(PL/'FIT_IMAGES.json')+'; '+str(PL/'EVAL_IMAGES.json'),minimal_action='Future exact original product/granule/baseline public metadata only; reject same-day/tile or L2A replacement; no existingXML repeats',estimated_requests='0 now; future up to258XML GET plus bounded object-list metadata lookups, separately capped',estimated_bytes='0 now; future planning estimate10-250MB, not measured; must inspect object sizes and freeze total before authorization',estimated_CPU='<=30min parsing estimate',acceptance='full product/granule/time/baseline identity links and sourcegrids; keep missing products in132list',control='allpool readiness, no selection by outcomes',executable_now='no; plan only'),
      dict(gap='north/grid convention, cloudheight, viewing and terrain',objects=exact,existing='Phase75 exactXML coarse sun/view grid; old76B11 transform only; literature/ESA_ATBD_0.pdf',minimal_action='Document geographic-to-grid direction convention and independent height/context prior; decide bounded approximation from image/metadata only; no targettruth optimization',estimated_requests='0 now; no DEM acquisition proposed until necessity and separate budget established',estimated_bytes='0 now; future metadata/code<=1MB estimate; raster/DEM size unknown and not authorized',estimated_CPU='<=15min assumption review estimate; no projection',acceptance='explicit signs/units/gridnorth, admissibleheight bounds, parallax/terrain failure cases and physical contextcoverage; unresolved means stop',control='correctdirection vs fixed180degree placebo',executable_now='no'),
      dict(gap='externalcloudcontext / observability',objects='all132 exactproducts in INPUT_READINESS.csv',existing='published1022cube; Phase3091152documentary extent; sixoldJP2 exist but are not opened here',minimal_action='Future freeze ray-support and availablecontext requirement; audit exact observable footprint without truth. If absent, mark unpaired unknown and retain baseline; do not fetch fulltiles automatically.',estimated_requests='0 now; future data requests/bytes not budgetable until height and footprint frozen',estimated_bytes='0 now; NA futurecontext size, must freeze before acquisition',estimated_CPU='<=10min footprint specification estimate',acceptance='coverage and cloud-detection uncertainty accounted for; no negative assignment outside observed context',control='nondirectionalcloud-neighborhood vs direction; missingproducts retained',executable_now='no'),
      dict(gap='comparable training controls and missingfitstudentmaps',objects='100fit+32development fixed304 lists; Source caches existing132, studentfullcache32',existing='all cache paths and inheritedSHA in INPUT_READINESS.csv',minimal_action='No missingfitstudent map forward needed for presentaudit; future freeze5arms, equalclasssupport and actualupdates feasibility before any training; fullfitstudentmaps only if separately justified/budgeted',estimated_requests='0',estimated_bytes='0 now; future training/prediction budget not granted or frozen',estimated_CPU='<=10min protocol review estimate; train/forward0',acceptance='same architecture/LoRA/optimizer/seed/step schedule; no geometry signal confounded by fewerShadow samples; selection no targettruth',control='originalPL, equalcountconfidence, nondirectionalneighborhood, correctdirection, fixedplacebo',executable_now='no')]
    table(ROOT/'NEXT_ACTIONS.csv',actions)
    protocol='''# Conditional minimal experiment — NOT EXECUTABLE / NOT RUN

唯一假设：在可信照明几何及可观察上下文成立时，方向一致性是否减少错误Shadow监督，同时保留真实Shadow学习。Source Shadow、Source Cloud与真实Shadow混淆都需报告；不预设仅是暗地表问题。

先决条件：INPUT_READINESS每项按产品/波段闭合，尤其RGB发布预处理、地理北到格网/行列、可辩护云高和上下文。旧3景B11证据仅限该波段，不传播到RGB/132池。高度范围、投影公式、支持掩膜目前未冻结，因此不得开训练。论文历史固定高度不是当前先验。不要重复Phase75取XML、Phase76原六JP2或Phase77有限搜索；先做NEXT_ACTIONS的单次可验收补齐，无法闭合即停止。

冻结五臂：原Q90-PL；等接纳数量和类别支持的无方向置信度选择；无方向云邻域；正确照明方向关联；固定180度反方向安慰剂（不再挑旋转角）。原PL可保留其原支持作描述参照；四个机制臂必须按每产品/每裁剪/每类匹配接纳数和权重总量，方可互比。必须在任何训练之前证明相同有效步调度可实现；若做不到，不得把不同实际更新或缺类造成的收益归于几何。

参数、LoRA初始化、优化器、学习率、数值精度及裁剪顺序完全相同。以原304的1000尝试/955更新/45跳过为可行性约束；若几何支持不能在相同步骤维持匹配且保留真实Shadow学习的可检验性，方案不准入，不自行补样、重复步或增加预算。当前没有实施loss或掩膜，不以“通用loss+VFM”声称贡献。

选择仅用影像/Source/元数据；Source预测Cloud既可能误判真Shadow，也可能漏检云，故无配对只记unknown，保留原监督，不直接翻为Surface/Cloud。邻近/连通/形态不等于投影。遮挡、外部云、水体及地形失效须事前列明，缺元数据产品留在132清单及全32组汇总，不按结果删组。

所有选择/参数/端点冻结后才独立真值评价：Shadow伪标签precision/recall、三类全27格错误流、Cloud→Shadow及Surface→Shadow损伤、真实Shadow召回、全32组组均/池化IoU和逐组损伤。等数支持对照及方向安慰剂是可否证条件，报告漏配/不确定覆盖，不以减少Shadow监督数量或最大亮地表组修复作为几何证据。多次development暴露的局限保留；不据该池调高度/方向/阈值。

方法边界：Phase74是旧监督浅层特征中的无方向Cloud排除；Fmask/Sen2Cor为固定算法/后处理的几何与辐射组合；拟议分枝若成立，才是无标签训练监督选择的方向增量。三者评价支持不同。现有结果未证明新增贡献或方法收益，不作“首次”声明。不自动启动下一phase。
'''
    (ROOT/'MINIMAL_EXPERIMENT.md').write_text(protocol,encoding='utf-8')
    result=f'''# Phase313 云—云影关联约束输入核查

## 数据结果

固定原100fit＋32development，仅读清单/数组头/XML/旧报告：132景Source scores/valid均在；Q90-PL完整学生预测/评分32景，fit100景仅有Source教师伪标签及必要分数，未补前向。Source SHA `{SOURCE}`，9个512窗起点{{0,256,510}}²，FP32分数求和/覆盖数平均；缓存SHA继承固定历史锁，本轮核对文件存在与头部，未重读像元验SHA。

逐产品最终状态：ready0、approximate-only3、missing129。旧75三景T37LDL/T20PMQ/T46RBU全部是fit，原始product/granule/time/baseline与XML对应；tile太阳角为平均量，原始源格网已验。旧选择用公开Shadow比例筛选再hash取3，不能代表全132。其他129精确XML未发现，不用同tile同日替代。

补充历史索引发现：Phase76在3景B11独立验证了65像元内裁、行向下列向右；Phase77的B04十二候选0通过。保留B11坐标证据，RGB/全波段发布链仍未闭合，地理北转格网方向、可接受云高、地形及外部云上下文也未闭合。Phase309的1152→1024→1022解释文档足迹差异；65/侧从文档单独推导仍是条件假设，不能推广旧3景B11证据。

一手文献仅3个对象：ATBD与Fmask作者PDF取得并保存SHA/版本/页文本；Copernicus处理器网页TLS握手超时，内容不可核实，不编造其版本/算法。总响应体{network['response_body_bytes']:,}字节，低于20MB。ATBD§3.4使用辐射与几何互补；Fmask2015的Sentinel2部分是早期原型，并非当前132产品高度真值或性能证据。具体物理范围和引用见PHYSICAL_ASSUMPTIONS，版本与获取UTC见SOURCE_VERSIONS/NETWORK。

## 结论分析

**几何实验尚不可执行。** 水平面/无视角地形时，位移依赖h·tan(太阳天顶角)且背离太阳；未核验cube方位/高度/上下文前不能转row,col或生成配对。平均角不是逐像元角，固定云高不是真云高；视野外云、漏检及水体/地形暗区使未配对Shadow只能记unknown，不能翻成负类。

Phase70 A/B/C均未获继续资格；Phase74 E重拟合FPR0.889%，高于B/R0.734%，失败保持。新增是VFM缓存、文档与明确方向安慰剂设计，不是旧失败撤销。仅换骨干或加通用loss尚无充分贡献依据，无新方法收益/首次声明。

NEXT_ACTIONS交一次具体补齐方案：优先复用3景旧B11和六JP2的既有证据，取得可独立核验的RGB发布流程，明确北向/高度/上下文；剩129精确XML仅列未来对象与估算成本。现在不下载、不配准、不再次盘点或无限搜索。MINIMAL_EXPERIMENT规定原PL、等数置信度、无方向邻域、正确方向及固定反方向五臂，并要求类别支持与实际更新对齐，冻结后才看真值。高度/掩膜未冻结，训练不准入。

本轮训练/拟合/神经前向/原始栅格下载/投影/DEM下载均0。资产核查在gzs cloud-lite-pt210完成，本机只编辑、文件SHA和完整备份。初始路径错误的失败与网络握手异常保留；原历史输入/结果不改。逐产品、逐项状态与精确路径见INPUT_READINESS；完整证据/访问/代码锁一并备份。无定时任务，不启动下一phase。
'''
    assert len(result)<4300
    (ROOT/'RESULT.md').write_text(result,encoding='utf-8')
    (ROOT/'phase313.md').write_text('''# Phase313 实验记录

- 目标：一次性核查原132产品云—云影几何可检验性，复用70/74/75/76/77/309证据。
- 改动：新增有界元数据/数组头盘点、文献读取、旧B11证据补充及有条件五臂方案；无科学计算重跑。
- 网络：本地push、gzs pull；3文献对象，2PDF成功、处理器网页TLS握手超时。SSH/Git有连接关闭，首次只读报告补充因旧路径错误停止，修正为reports/summary.json后独立恢复，原入口/失败保留。详见OPERATIONS与NETWORK。
- 止损：CPU8GiB/1h、宿主空闲RAM/磁盘≥4GB、新持久含双机/备份≤300MB、新文献响应体≤20MB；不改环境/历史文件。
- 结果：132Source缓存可用；学生32；3fit有原太阳角/格网和B11局部坐标证据，RGB链未闭合；ready0/approximate-only3/missing129。几何实验尚不可执行；零训练/拟合/前向/投影/栅格和DEM下载。
- 边界：旧74失败及77有限候选失败保持，现有VFM缓存不等于新增方法贡献。缺项补齐及五臂仅提案，无下一phase或定时检查。
''',encoding='utf-8')
    write(ROOT/'OPERATIONS.json',dict(events=['initial read-only schema SSH session91256 closed at kex; retry48349 succeeded','read-only source text SSH99468 closed at kex','diagnostic SSH1971 closed before output; historical local source established reports/ path','git push of d69cfcc first connection closed, identical authorized push succeeded','review12975 failed before inventory writes: Phase76 path; preserved original code and REVIEW_FAILURE01, independent recovery path only','one ad-hoc read attempt extracted no newHTML as successful literature responses werePDF; no new network bodies'],scientific_restarts=0,original_audit_repeated=False,environment_installs=0))
    mem={k:int(v.split()[0])*1024 for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines())}
    assert mem['MemAvailable']>=4000000000 and shutil.disk_usage(ROOT).free>=4000000000
    state.update(status='completed',decision='geometry experiment not executable',seal_seconds=time.monotonic()-start,metadata_only=True,all_input_SHA_reverified=True,final_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    state['total_seconds']=state['metadata_and_review_seconds']+state['seal_seconds'];assert state['total_seconds']<3600
    state['minimum_available_memory']=min(state['minimum_available_memory'],mem['MemAvailable']);state['peak_cpu_bytes']=max(state['peak_cpu_bytes'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(ROOT/'STATUS.json',state)
    code={str(p.relative_to(R)):sha(p) for p in (R/'src/tools').glob('*phase313*') if p.is_file()}
    write(ROOT/'DECISION_LOCK.json',dict(code=code,decision=state['decision'],products=132,states=dict(ready=0,approximate_only=3,missing=129),all_inputs_reverified=True,first_failure_retained=True,utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(state))
if __name__=='__main__':main()
