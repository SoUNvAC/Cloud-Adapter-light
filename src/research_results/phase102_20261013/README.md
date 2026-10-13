# Phase102：新外部池元数据准入审计

## 1．数据结果

仅审计固定KappaSet7100327及PixBox5036991。零训练、拟合、预测、性能评价、影像／掩膜像元读取；未分配校准／确认角色，未选择模型。

|来源|索引身份|排除重叠|待核实|元数据通过|剩余相关组|
|---|---:|---:|---:|---:|---|
|KappaSet|9251子块／1079个tile＋采集时间键，唯一原产品数未核实|180键|899键|0|794个已知连接分量，只是未解析连接的上界，非794独立组|
|PixBox|29个唯一原L1C产品／17351条位置记录|13产品|0|16产品|10组，限已登记项目池和本轮父产品缓冲筛查|

PixBox排除包括8个同tile产品和5个新增空间相交产品。CPU独立全历史距离遍历复核21个非同tile产品、这5项空间排除与16项通过；BFS复核相关组。全部29产品原PRODUCT_URI、六波段对象索引、XML格网与点位索引匹配，坐标相对10m像元角点最大残差7.6311m（检查界限15m）。这是位置元数据一致性，不是影像／标签像元配准或语义验收。

复用Phase86双方各1000m缓冲、相接算重叠、同tile任意日期／同UTC日同轨／缓冲交叠的传递分组。历史9717条位置记录覆盖9025个Source监督train/val产品映射、全部513个Catalogue发布产品、142个历史L8/SPARCS输入及全部37个ALCD产品；历史来源SHA与Phase86清单匹配。完整父产品格网用于保守排除，可能过度排除。

公开资料body累计记录6,229,407 bytes；对已取消批量请求额外按全部111个计划cap计费，加两页文字浏览1MB预留，保守核算16,109,518 bytes<20MB。浏览器底层传输未独立计量，详见inputs/source_access_sha256.json，不能把body核算称抓包测量。未下载56GB Kappa影像包或22GB PixBox影像包。

## 2．结论与限制

优先进入下一步输入／标签核验的候选为PixBox，但只可能提供选择性采样点上的有限参考证据；未知采样概率／权重，不能直接估计全场景FPR或密集分割mIoU。有效云影组数未读、未推断。10组不是已认证统计独立性或已满足确认支持量。

KappaREADME确实说明512倍子块索引；原产品处理基线、原生格网／轴向、NetCDF缩放和1079键与官方1038产品的对应仍未解决。仅1个CDSE同采集重处理父产品可作代理足迹，不能当原子块位置证明。批量OR查询超时后取消，没有转为整包下载。899键保持pending_metadata，不以不同日期／文件名推断独立。

PixBox小ZIP含标签CSV，已取得并保留本地；只词法投影ID、PRODUCT_ID、PIXEL_X/Y、LATITUDE/LONGITUDE，未转换、展示、统计或按逐点类别值筛选。类别定义文件仅用于理解字段语义。原ZIP与含标签CSV不入git、不传远端；传送的是坐标白名单。不能声称“从未取得标签文件”。

657个限定代码／协议／项目日记文件未发现候选使用记录；范围不含work_dirs、像元和性能文件，未记录的人工使用仍未知。Source基础预训练语料未知。旧H1与历史失败分枝保持不变，无旧sealed像元访问。

## 文件

- product_location_index.csv：每个产品／采集键的位置证据、输入可获得性、历史匹配与准入状态。
- inputs/kappa_patch_positions.csv、inputs/pixbox_positions_only.csv：子块索引与坐标白名单；无逐点类别字段。
- overlap_exclusions.csv、related_groups.csv、correlation_edges.json：排除证据与组成员；不分配实验角色，代表仅用于元数据命名。
- coordinate_grid_checks.json、metadata_audit.json、CPU_VERIFIED.json：CPU元数据核验及边界。
- inputs/source_records.json、inputs/source_access_sha256.json、inputs/access_ledger.json：版本／许可证／官方长度与checksum、实际访问SHA及失败；官方大包checksum只是目录声明，未验证大包内容。
- SOURCE_FACTS.md：类别、标注、输入、缩放与版本谱系。
- READINESS.md：只讨论下一步准入缺项，不启动下载／人工复核／实验。

代码先本地检查push，再远端ff-only pull，用既有cloud-lite-pt210进行CPU复核；只同步小元数据，不同步work_dirs。预算截止2026-10-20北京时间不变。

剩余PixBox覆盖2017-01-13至2018-03-03，10个MGRS tiles为19FDV、19GBQ、22WES、32SPJ、33VVE、40UEE、44SKJ、48NUG、48XWG、54TWN；区域依据只到格网／坐标，不推断国家。Kappa剩余键覆盖2020-01-02至2021-12-29，完整tile清单及年分布见coverage_metadata.json，不能把位置未核实的键当精确footprint通过。
