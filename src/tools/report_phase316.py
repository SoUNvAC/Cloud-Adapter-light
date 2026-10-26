"""Publish the actual metadata audit, with no invented array verification or future access cost."""
from phase316_common import *
def main():
    resource_check();assert not (ROOT/'RESULT.md').exists()
    a=load(ROOT/'AUDIT_COMPLETE.json');assert load(ROOT/'INDEPENDENT_ACCEPTANCE.json')['status']=='passed'
    o=read(ROOT/'OVERLAP_AND_HISTORY.csv');source_only=sum(bool(json.loads(r['current_Source_roles'])) for r in o)
    old_other=sum(r['original_status']=='excluded_overlap' and not json.loads(r['current_Source_roles']) for r in o)
    splits=read(ROOT/'RELEASE_SPLITS.csv');months=a['month_keys'];tests=a['test_keys'];shared=a['month_test_shared_keys']
    report=f'''# Phase316 KappaSet准入审计

现在没有满足条件、可以直接进入少量样本读取核对的候选。主要障碍是原产品与足迹身份、实际存储缩放及Shadow正支持仍未核实；类别定义本身已有可靠文档。能否单独取得小文件尚无可验收的方案：若只能取整包，就要下载56,186,308,652字节（约56.19GB），另加解压空间。取得新证据的成本仍有unknown，本轮到此结束，不自动下载或再开同类核查。

## 查清了什么

复用Zenodo7100327 Version1、缓存API revision6及原README。README大小231988字节、MD5 `98bf16a2d90cba386b197133a24f402f`核对通过；归档MD5只是发布声明，56GB包未取得。0次新资料请求、0字节新资料响应；旧文档缓存获取与代码同步分账。

|对象|本轮复算/证据|解释边界|
|---|---|---|
|发布子图|9251个.nc成员，{a['unique_paths']}个唯一路径|目录计数非像元验收；不是9251独立样本|
|采集键|1079个tile/time键；同tile/day仍1079|比官方1038多41；键不是完整SAFE产品|
|发布目录|月份目录{months}键，test目录{tests}键，共享{shared}键|逐目录/重复与成员映射见表；不能据目录擅自改官方数字|
|精确原产品|0个完整原产品身份核实；1079键仍unknown|卫星/基线/轨道/生成时间未绑定；已有重处理代理不替代原件|
|旧排除/待核实|180键保留排除，899键保留待核实|794旧不完整连接分量非794独立组|
|当前Source同tile|{source_only}键；其中同tile/time {a['current_Source_same_tile_time_keys']}键|保守空间风险，不是确切像元重叠证明|
|旧排除且无当前Source同tile|{old_other}键|其他旧历史理由单列，不改写为当前Source训练交集|
|合格候选/真实Shadow支持|0 / unknown|没读新标签；分类名不能代替正像元或正组数量|

官方声明0 UNDEFINED、1 CLEAR、2 CLOUD SHADOW、3 SEMI TRANSPARENT CLOUD、4 CLOUD、5 MISSING；预备父类为1→Surface、2→Shadow、3/4→Cloud、0/5→ignore，保留原码。半透明云并入Cloud并不消除标注边界差异。旧5095024的0/5相反，禁止混用。CVAT/Segments.ai半自动、偏向挑战子图的标注不是独立物理金标准。

README声明512×512、10m、13波段加Label，较粗波段经Sinc IIR/Blackman重采样；test附FMC等算法图层不能当Label。README明确说子图不带地理参考，文件索引×512不提供CRS、原点与轴向。存储dtype、单位、fill、scale_factor/add_offset及真实同索引对齐均未开数组核验。作者代码有自定义`scaling_factor`和float/ubyte写出线索，但快照未绑定实际发布导出版本，不能据此假定DN÷10000。当前模型RGB=B04/B03/B02、TOA×255、先BGR再预处理回RGB及固定mean/std；这还不足以冻结Kappa读取器。原生10m的512窗为5.12km，而原Catalogue20m窗为10.24km；未选择缩放/重采样。

## 交集、历史与现实成本

对目标checkpoint SHA `{SOURCE}`只读字节校验。实际保存配置、CloudSEN12固定revision清单和既有逐PNG RGB映射证明8490 train/535 val，共9025条Source记录；本轮未重开RGB。全部1079键逐项保留旧判断/理由/匹配，当前Source、其他历史池、旧metadata-only使用分开。仅同tile足以保守排除风险，但未知子图足迹不能当精确交集；无同tile也不证明跨tile空间独立。没有所需原网格/权威完整tile范围证据时保持unknown，未增加云高/太阳角等物理配对要求。DINO预训练及未记录人工使用仍unknown；历史搜索范围、文件SHA和命中行在HISTORY_SEARCH，未命中不等于从未使用。

最小新资料是每个候选成员→原SAFE/网格/索引映射、可证与训练及已用历史足迹分离的范围、真实NetCDF头和缩放/同维度规则，以及正Shadow支持。现有ZIP目录可给单成员压缩/展开字节，却没有验收过的小成员访问承诺；不新增范围扫描、影像/标签请求或作者联系。全包取得后的.nc展开合计见ADMISSION_AND_COST，磁盘还须保留ZIP，解压CPU成本unknown。CC-BY-4.0要求保留作者、题名/版本/DOI/许可及改动说明；作者工具代码Apache2.0与数据许可分开。

IDENTITY_RECONCILIATION、INPUT_LABEL_MAPPING、OVERLAP_AND_HISTORY、ADMISSION_AND_COST交付全部关键项；PATCH_MEMBER_MAP保留9251成员与旧键映射。独立CPU核对两套名称解析、9251/1079计数、180/899原判断、9025条Source绑定及六原码父类映射。未训练/拟合/前向/读新标签/下载整包/更改论文或三个PL端点。本轮只减少数据可用性不确定性，尚未完成独立确认。
'''
    (ROOT/'RESULT.md').write_text(report,encoding='utf-8')
    record=f'''# Phase316 实验记录

- 目标：只判定固定KappaSet7100327 Version1能否进入下一次小样本输入核对，不做模型实验。
- 改动：增加本轮元数据脚本；复用官方README/API/旧目录与历史清单，保存9251成员映射、1079键原判断及当前Source9025条绑定；不改旧文件/端点/论文。
- 网络：新官方资料请求0/响应0字节，既有README MD5通过；首次gzs git pull遇TLS握手中断，未运行发现脚本，重试同步成功。代码先本地push再远端pull；无新ZIP范围请求、影像/标签/整包下载。
- 止损：CPU8GiB/1小时、宿主空闲RAM/磁盘≥4GiB、新持久含双机/备份≤300MB；训练/拟合/前向均0，请求≤3/3MB；无需定时检查。
- 结果：9251成员/1079键（月份{months}/test{tests}/共享{shared}），1038官方产品与键计数差仍未获完整原产品证据；180保守排除不变、899待核实；{source_only}键与当前Source同tile，其他旧理由不冒充Source交集；0候选、实际Shadow支持unknown。独立计数/身份/语义核对通过。
- 代价与限制：若只能全包，下载56186308652字节另加展开空间；小成员可靠访问/新资料费用unknown。不因未命中历史记录而宣称未见，不自动进入下一phase或再次同类核查。
'''
    (ROOT/'phase316.md').write_text(record,encoding='utf-8')
    print('Phase316 report:0eligible candidates; preserve180 exclusions/899unknown; no pixels/models/HTTP')
if __name__=='__main__':main()
