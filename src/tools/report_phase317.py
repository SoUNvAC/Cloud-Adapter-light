"""Actual three-pool exploratory decision; CB is a baseline, not a claimed new method."""
from phase317_common import *
def main():
 resource_check();assert load(ROOT/'INDEPENDENT_ACCEPTANCE.json')['status']=='passed' and not (ROOT/'RESULT.md').exists()
 rows=read(ROOT/'METRICS.csv');support=read(ROOT/'DATA_AND_SUPPORT.csv');events=read(ROOT/'DAMAGE_BENEFIT.csv');flow=read(ROOT/'ERROR_FLOW.csv');per=read(ROOT/'PER_SCENE_METRICS.csv');seal=load(ROOT/'ALL_SEALED.json');history=load(ROOT/'HISTORY_IDENTITY.json');assert history['status']=='passed';decisions=[]
 def num(s):return None if s in ['NA','None',''] else float(s)
 def fmt(v):return 'NA' if v is None else f'{v:.3f}'
 def row(pool,a):return next(r for r in rows if r['pool']==pool and r['strategy']==a)
 def delta(pool,key):return num(row(pool,'CB')[key])-num(row(pool,'PL')[key]) if num(row(pool,'CB')[key]) is not None and num(row(pool,'PL')[key]) is not None else None
 gains=[]
 for pool in ['CMC','SPARCS','L8']:
  s=next(r for r in support if r['pool']==pool);r=next(r for r in events if r['pool']==pool and r['scene']=='POOLED');g=delta(pool,'Shadow_IoU_pct');gain=g is not None and g>0;gains.append(gain)
  classes=json.loads(s['accepted_classes']);insufficient=int(s['PL_updates'])==0 or classes[2]==0
  decisions.append(dict(pool=pool,question='Does simple per-present-class CE balancing help Shadow vssame-maskPL?',Shadow_IoU_delta_pp=g,macro_Shadow_recall_delta_pp=delta(pool,'macro_Shadow_recall_pct'),macro_Shadow_FPR_delta_pp=delta(pool,'macro_Shadow_FPR_pct'),group_mIoU_delta_pp=delta(pool,'group_mean_mIoU_pct'),D=int(r['D']),D_repaired=int(r['D_CB_Surface']),B=int(r['B']),B_retained=int(r['B_CB_Cloud']),outside_D_CB_Shadow=int(r['Doutside_CB_Shadow']),simple_CB_has_observed_Shadow_IoU_gain=gain,insufficient_Shadow_accepted_support=insufficient,uncertainty='one seed; historicallyexposed pool; display512SPARCS/L8 vs TOACMC; no crosspool pixelpooling',decision='supportinsufficient; no efficacy verdict/no threshold rescue' if insufficient else 'simpleCBalreadyhas observed effect; evaluate fullcost, cannot claim innovation' if gain else 'CBnot an across-the-board repair; retain exactfailure before newhypothesis',next_gap='teacheraccepted classsupport/noise vsclassweighting; anycandidate must beat thissame-maskCB and preservebenefit at matchedcost; no nextphase launched'))
 table(ROOT/'DECISION.csv',decisions)
 main='\n'.join('|'+r['pool']+'/'+r['strategy']+'|'+'|'.join(fmt(num(r[k])) for k in ['group_mean_mIoU_pct','pooled_mIoU_pct','Shadow_IoU_pct','macro_Shadow_recall_pct','macro_Shadow_FPR_pct'])+'|' for r in rows)
 dlines='\n'.join(f"|{r['pool']}|{fmt(r['Shadow_IoU_delta_pp'])}|{fmt(r['group_mIoU_delta_pp'])}|{fmt(r['macro_Shadow_FPR_delta_pp'])}|{r['D_repaired']}/{r['D']}|{r['B_retained']}/{r['B']}|" for r in decisions)
 text=f'''# Phase317 三池普通/类别均衡自训练

三份数据已按固定协议完成：新增5个端点，CMC旧PL只复用。CB相对同接纳集合PL在{sum(gains)}/3池的Shadow IoU提高；是否值得继续须同时看下面的地表误报和Cloud纠正撤销，不能把类均衡本身当新方法。单seed、旧开发池只支持探索判断；不据此宣称新盲测或稳定跨seed优势。本轮不自动加方法/seed。

|池/策略|组均mIoU|池化mIoU|Shadow IoU|宏Shadow召回|宏Shadow FPR|
|---|---:|---:|---:|---:|---:|
{main}

|池|CB−PL ShadowIoU pp|组均mIoU pp|宏FPR pp|D修复/PL损伤|B保留/PL纠正|
|---|---:|---:|---:|---:|---:|
{dlines}

DATA_AND_SUPPORT保存每池三类接纳、类别出现步、非空产品与实际更新/跳过；没有补步。CMC-CB应为955更新45跳过，SPARCS/L8实际值见表，不把1000尝试叫1000更新。全部臂从原Source新建seed30201的相同LoRA（rank4/alpha4/dropout0，294912参数），初始张量与原304一致；仅CE平均方式改变。CB先对本步每个非空接纳类求meanCE，再等权平均；缺类不补0、不重采样、不改阈值。所有初末冻结参数/buffer一致，无旧目标监督head/优化器进入初始化。

CMC沿100fit/32dev原TOA1022与1000步512裁剪翻转/9窗；SPARCS60fit/10val，L8按scene轮转SHA取100fit（18scene）/963val（8scene），均只用shadow=yes。另两池为历史RGB显示图的固定512Resize，黑色不是训练无效区，既有mmseg输入与Source得分≤1e-6门检通过；不把显示图写成原反射率，也不与原随机native监督分数公平排名。SPARCS参考nearest512；L8原转换把fill并入clear，本轮在全部端点及缓存封存后才用既有原掩膜核对963 PNG裁剪全局坐标、精确复现原编码，再仅评价排除rawfill，训练仍不读取mask。SPARCS保留雪/水/地表/洪泛子类，L8薄云及雪冰/水体biome分层另表；biome不冒充逐像元雪/水标签。

主表按scene/group平均及各池自身池化，绝不跨池合并像元压过失败。PER_SCENE_METRICS与PER_SCENE_NATIVE保留全部组、缺类union/NA、分母；METRICS另含三类IoU、宏/池化AP及可算组数。三组比较Source→PL、Source→CB、PL→CB的完整27格/六向错误边缘核对通过；DAMAGE_BENEFIT列CB修复D/残留Cloud/转Shadow、保留B/退回Shadow/转Surface与D外错误。CMC D=506798/B=784478精确复现，两新池只用自己的D/B，分母为0不编比例。旧manifest实际选中的{history['shared_image_matches']}张图像与shared副本SHA一致；旧交集元数据和跨池原sceneID交集另存，不以没有同ID证明空间独立。

Source/PL/CB Shadow评分均为softmax(s)[2]的float32，池化AP精确分组ties计算、不降评分精度；CPU独立复算与原始输入/代码/输出SHA核对通过。所有5末步、代码/日程/伪标签/日志及评价先封存，独立CPU随后才读参考。实际新增前向{seal['forwards']}，GPU管线{seal['gpu_seconds']:.2f}秒，含全部门检/构建；上限11000/12h。16GiB CPU、宿主空闲RAM4GiB/磁盘8GiB、双机新产物16GB，流式全量备份不保留大归档；实际备份回执另附。

DECISION逐池列效应与代价。若某池Shadow从未被接纳或更新为0，只记支持不足；不能据此判算法无效，也不调阈值救回。若简单CB已改善，多池效果仍需同时报告代价；任何未来候选必须超过此对照。若反向，先根据具体支持/错误位置形成可检验原因，不能无目标加模块。14天局部决策窗口按DECISION_WINDOW实际首次执行时间落锁，失败/等待不重置；不是项目总截止。CMC/SPARCS/L8历史标签开发暴露保留，三数据集是两个传感器系列，不构成独立目标域确认。本轮不改论文或旧phase、不自动启动下一phase。
'''
 (ROOT/'RESULT.md').write_text(text,encoding='utf-8')
 record=f'''# Phase317 实验记录

- 目标：三池Source/Q90-PL/同掩膜CB探索基线，判断简单均衡对Shadow的效应与代价；不是新方法/新盲测。
- 改动：CMC只新CB，SPARCS/L8各PL+CB；原Source全新30201 LoRA、同初始化/日程/接纳集合，冻结主干head/buffer。无下载/升级/调参/EMA/目标监督初始化。
- 输入：CMC原TOA/9窗；SPARCS原RGB预览、L8原RGB PNG固定512。L8仅shadow=yes，100fit来自18scene、963val来自8scene；训练不读mask，评价封存后从原mask恢复被PNG合并的fill并精确对齐。
- 网络：本地代码push→gzs pull；最终流式全量回传，故障如有保留独立记录；无新数据请求。
- 止损：新增前向≤11000、GPU≤12h、CPU≤16GiB、新持久含双机≤16GB、宿主空闲RAM≥4GiB/磁盘≥8GiB；不重训/补步/降精度/删val，不加seed/loss。
- 结果：5新臂均封存；前向{seal['forwards']}、GPU{seal['gpu_seconds']:.2f}秒；CB在{sum(gains)}/3池ShadowIoU相对PL增加，逐池收益/地表误报/Cloud纠正撤销及支持不足见DECISION，不以均值掩盖失败。独立CPU原mask/27格/六向流/D/B/SHA核对通过，CMC旧D/B精确。
- 局限与决定：单seed且历史暴露，显示512与TOA协议不同；本轮建立普通均衡基线，不包装创新。无自动下一phase/定时检查；14天窗口不随等待/失败重置。
''';(ROOT/'phase317.md').write_text(record,encoding='utf-8');print('Phase317 actual report complete')
if __name__=='__main__':main()
