# Phase65：独立场景诊断 → 匹配信息对照 → 最小机制

预注册草案，2026-10-06；尚未读取新确认集、尚未训练。首次启动前必须完成 scene lineage 审计并冻结输入 SHA256。研究预算截至 2026-10-20（北京时间），这是投入上限，不保证完成；到期停止新作业，记录已完成与未完成项。

## 不可变协议

- 仅访问本地 D:/Cloud-Adapter-light，远端 /home/scv/Cloud-Adapter-light 与 /home/scv/shared。检查 realpath，拒绝链接越界。代码只在本地修改、验证、提交、push，然后远端 pull --ff-only。不得覆盖用户已有改动。
- 远端必须 conda activate cloud-lite-pt210。一次一个训练任务，work_dir 不复用、不覆盖。保存 git SHA、输入哈希、配置、环境、PID、退出码和原始日志。
- L8 仅 USGS Shadows?=yes（shadow?=1）；no、缺失、未知均排除。NoData/无效标签继续 ignore，不能当 Surface。三父类顺序 Surface/Cloud/Shadow，Thin 与 Thick 合并。雪冰、水体和失败场景不删除。
- 已有八场景只提供探索性假说，连同旧实验参与假说形成或调参的场景全数登记排除。现有 sealed test 保持封存；只允许读取 scene ID 清单用于排除，不读影像、标签、预测或指标。
- 新确认集从合格且具有可核验未使用 provenance 的目标训练池中按 scene 留出，不得看结果选场景。至少 8 个确认场景、8 个开发场景，否则报告不可识别并停止。按 SHA256('phase65:'+scene) 排序，前 8 个确认，其余开发。这是最低可行性约束，不代表统计功效已充分。
- 确认集只最终评估一次，训练/早停/模型选择不得使用。开发池内部按 scene 划分拟合/验证，冻结后再启封新确认集。增加 seed 不增加独立场景数。

### 2026-10-06：新数据入口（在新模型评估前登记）

用户指定全新 Sentinel-2 Cloud Mask Catalogue，Zenodo record 4172871 / DOI 10.5281/zenodo.4172871，双机此前未下载。作为候选新目标池替代已用L8确认入口；不改变H1/H2终点或止损门，也不解封旧test。

发布方定义：513个1022×1022子场景、20m、13波段float32 TOA；mask为三通道bool one-hot。仅使用classification_tags.csv的shadows_marked=1作为完整云影监督候选，其余89项排除并报告范围限制。MAIN/CALIBRATION/VALIDATION是作者标注过程分组，不能直接当本研究训练/测试划分。类别顺序与无效/非one-hot mask须依README及数组审计核实，不猜编码；RGB band indices [3,2,1]。

下载全部七个发布文件，在本地data/sentinel2_cloud_mask_catalogue_4172871和远端/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871保存原始包、record.json、官方MD5+本地SHA256及ZIP CRC报告。安全断点续传，未校验包保留.partial。下载不等于训练就绪；冻结前必须审计源CloudSEN12及历史实验的产品ID、MGRS tile、采集时间与空间足迹交叉，按独立场景/相关产品组划分，无法核验的重叠候选不进入确认集。地表/云型等人工标签和真值比例不得用作部署风险特征。几何元数据缺失则明确标缺失，不捏造太阳角或云高。

这是同传感器跨数据集诊断，不能据此宣称Landsat跨传感器收益。统计代码与新数据loader完成并冻结前，训练仍不启动。

### 2026-10-06：元数据分组规则（尚未选取或评估确认集）

对513个产品，按同MGRS tile或同采集日期+relative orbit作传递闭包分组，避免相关产品落在不同split。`shadows_marked=0`仅可作为分组连接节点，绝不进入训练/评估样本。group只要任一成员与source的MGRS tile（任意日期）相同、或合格候选与source/旧target-train-val空间footprint重叠，整个group排除。历史目标审计仅打开训练/验证TIFF头信息；sealed test只读取manifest的scene IDs，不打开对应栅格。旧历史L8 shadow-invalid场景只参与旧使用谱系排除，不作为新实验数据。

source及历史geometry均保守扩张100m，历史用整个已使用scene栅格的地理外包矩形而非挑选patch；这会过度排除，须报告覆盖边界。保留极地投影场景并按头信息CRS+线性单位正确重投影，不猜坐标系。

此规则尚只产生`groups_prepared_not_split_locked`元数据清单，不授权训练。最终group-based训练/开发验证/确认划分、每独立group的统计代表及H1/H2有效分母支持规则须在任何新模型预测或指标之前写清楚并冻结；不把产品数量当独立场景样本量。前述L8逐scene冻结脚本不可直接用于Sentinel相关产品池。

### 2026-10-06：Sentinel确认设计定稿（新预测/指标仍未生成）

初稿的8-scene留出针对当时未知的新数据可用性，是最低可行门。元数据审计现有377个合格相关组，可增加独立场景覆盖；因此本次在任何新source/adapter预测或相关性计算前改为：按SHA256('phase65:split:'+group_id)排序，前64组为65a_confirmation，接着64组为65b_confirmation，接着64组为65c_final，接着32组为development_val，其余为fit。各split互斥，65b/65c保留组不参与65a模型选择、诊断或评价。作者标注阶段及difficulty不参与选取。固定seed65与原4,000-step配方不变；目标标签预算精确登记fit中全部shadow-valid产品与最终有效像元数。

每组固定一名统计代表：按SHA256('phase65:representative:'+product_id)排序的首个shadow-valid产品。训练使用fit组的全部合格产品；模型选择只用development_val代表。机制诊断、风险分类、置信区间均使用一组一个固定代表，增加同组产品不增加统计独立样本量。65a固定64代表独立确认，不能用65b/65c弥补失败或追加到显著。

H1是以真Shadow为条件的离线诊断，空Shadow类不能定义条件概率。新Catalogue的shadows_marked=1含“可以标注，但实际未出现Shadow”的场景；旧初稿“任一scene无Shadow则停止”不适合无条件混合这两种支持。本次在看任何预测/指标前明确支持域：public metadata shadow_percent>0的固定代表组成H1支持集；其他代表不计算H1条件率，但保持在全64组总体mIoU与Surface→Shadow安全报告中。支持名单在split锁中冻结，禁止事后按误差、gain或difficulty删scene。若fit H1代表<16、65a_confirmation H1代表<16，停止主检验；若实际有效mask的Shadow存在性与冻结public支持名单不一致，停止并报告协议失配，不重新抽样或静默删scene。H1仍为同一d对delta Shadow IoU的Pearson正关联、95%场景bootstrap下界>0及预测RMSE优于冻结常数预测，两门不变。该支持域不能泛化成所有无Shadow场景的关联主张。

H2使用fit全体代表。风险真值定义为适配后Shadow→Surface错误像元数增加（与有Shadow时条件率增加符号相同；无Shadow时增加数为0）。教师真值只用于训练风险标签，推理特征禁止真值/人工标签/标签比例。固定12个可观测输入：source的Surface/Cloud/Shadow概率均值、归一化熵均值、Shadow argmax比例、top1-top2 margin均值；原始TOA RGB平均亮度均值/标准差及亮度<0.08比例；原始TOA B8/B11/B12均值。source概率均在图像有效像元上汇总，不用GT-valid mask作部署特征。原始光谱不强行clip到[0,1]。暂不使用缺失的太阳角/云高或作者云型标签，不捏造几何信息。

风险模型固定训练中位数补缺+缺失标记、StandardScaler及LogisticRegression(C=1, class_weight=None, solver=lbfgs, max_iter=1000, random_state=65)。fit正/负风险代表均>=8方可拟合；确认正/负代表均>=8方可估计场景AUROC CI，否则报告不可识别并停止，不选择新阈值/新特征。主要AUROC的64组bootstrap使用10000次、seed65；CI下界>0.5不变。H2可在额外光谱观测上确认风险可识别，但65a不训练六波段分割网络；65b须在另外保留的64组上，以匹配RGB/六波段及删除证据对照确定信息贡献。缺失列若在fit全缺失则停止输入门，不用确认数据补齐。

此为新数据入口引起的预分析设计修订，不由新性能/相关性显著性触发；所有旧初稿条款保留用于审计上述变更。首次新模型评价前锁定本文件SHA、group/支持名单、source SHA、模型选择、数据管线与脚本。split冻结本身仍不授权训练，须完整下载、loader与统计代码检查通过。

## phase65a：两周内判定是否值得继续

冻结旧八场景探索产物及哈希，固定 hypothesis，不重新定义相关性追显著。

H1 离线机制诊断：适配前 d=P(预测 Cloud|真 Shadow)-P(预测 Surface|真 Shadow) 与适配后的净收益 delta Shadow IoU 正相关。唯一主统计为场景级 Pearson r；拟合开发场景线性关系，确认集检验 r 及预测 RMSE 对比开发均值预测。场景有零 Shadow 分母则不能计算，记录并停止主检验，不事后删除。95% CI 用 10000 次独立场景 bootstrap，seed=65；确认 r 的下界>0 且 RMSE 小于冻结常数预测才支持 H1。未通过或不可估计即停止该假说，不改终点、不替换相关性。

H2 部署可观测证据：仅用影像、原始元数据、source 模型输出，禁止真值混淆作为部署输入。开发集固定浅层 logistic regression，预测 delta Shadow→Surface>0 的场景风险。标准化仅在开发拟合；L2 C=1；缺失值训练中位数加缺失标记。候选为 source 概率均值/熵、暗像元比例，以及仅在可用时加入预先定义的光谱/几何/上下文观测。冻结特征定义与可用性后，确认 AUROC 场景 bootstrap 下界>0.5 才支持 H2；正负场景不足则不可识别。不能先用确认集选择特征。八个确认场景不够精确时报告不确定性，不能用更多 seed 替代场景。

固定 Phase64 三父类源 checkpoint（先核实实际文件 SHA，不以旧四类模型替代），MsRE 固定 4000 iterations、seed65、lr=1e-4；目标密集标签预算由冻结开发 manifest 精确计数。开发集选 checkpoint，再对确认集成对评估 source/adapted。训练配方复用须审计标签与源 checkpoint，不能直接沿用旧 target_val。

同时报告 delta mIoU、delta Shadow IoU、Shadow→Cloud、Shadow→Surface、Surface→Shadow（按各自真类分母），各 scene 原始 confusion 和有效像元数。上述两道门均支持才允许进入 65b。通过是立项证据，不是新算法成立。

## phase65b：确定缺失的信息

先匹配 RGB/六波段的 scene、footprint、空间网格、分辨率、辐射处理、有效标签与归一化拟合池。禁止用旧 RGB baseline 对比旧六波段下降归因。

检查原始栅格 CRS/transform/像元尺寸/配准、NoData 与有效 mask；normalization 仅训练拟合；同一小样本过拟合应达到训练 parent mIoU>=95%，否则先停止输入诊断，不能造复杂网络。源匹配六波段相对匹配 RGB 损失不得超过1 pp。阶段脚本在元数据审计之外仍需像素抽查、标签审核与真实拟合报告；元数据通过不等于输入正确。

最多三组证据：NIR/SWIR；云位置+太阳方位/高度（未知云高作为软不确定区间，处理裁块外云/地形/云检测错误）；大范围上下文。每组单独与匹配 RGB 比较，固定浅层探针，不联调复杂网络。独立场景增量 AUROC 95% CI 下界>0（相对 RGB）才支持区分能力，三候选作 Holm 多重检验校正。保持 label-valid mask，对关键难例独立复核，人工分歧不等于真值错误率。失败证据停止；若全失败，停止65c。

## phase65c：只实现证据支持的最小机制

结构在65b诊断冻结后才定稿；目前禁止训练新机制。仅一种带可靠性估计的约束，验证是否控制 Shadow→Surface，而非重复 RGB shadow head。

统一 source SHA、目标有效标签数量、训练步数/优化器、模型选择、RGB/多光谱输入信息与源损失约束。比较 LoRA、MsRE、共享头；完整机制、删除证据、打乱证据、参数量匹配对照必须齐全。与65a相同错误流报告及场景CI；目标 Shadow IoU增量CI下界>0、源mIoU损失<=1 pp、Surface→Shadow恶化CI上界<=1 pp，并验证完整证据优于删除/打乱证据，才可主张机制成立。65a/65b确认集已参与推进，不可充当最终独立方法测试；另行冻结未使用确认场景或报告仅开发验证，不解封旧 test。

## 日志和备份

### 2026-10-07 用户授权新增ALCD数据入口与手动同步

用户新增Zenodo1460961的双机下载及适用性审计。此入口先单独登记，不覆盖2026-10-06的失败split_lock，不重新抽取377组，不将原65b/65c保留组借入65a。下载本身不授权训练。新场景必须先核验原始多光谱影像可用性、产品ID/MGRS/时间/足迹、源域和历史及4172871重叠、影像与标签网格、NoData和有效标签规则；未知项不进入确认。若确有可用独立增量，在任何新模型预测前另行记录并冻结扩展研究协议、场景支持与训练配置，再决定能否启动。

ALCD官方描述给出38场景目录且两个子集有重复；标签由主动学习随机森林产生，0/1无效、2/3云、4云影、5地表、6水、7雪。不能把目录数当独立场景数，不能将置信图解释为独立人工真值正确率，也不能以参考标签/作者统计作部署风险特征。原始影像缺失时只能称参考标签已取得，不能称训练就绪。

用户已取消所有work_dirs自动打包/传回/备份和续传，改为用户手动操作；此指令覆盖下面旧备份要求。既有临时包和远端结果保留，不再自动检查或恢复接收脚本；代码git push/pull和双机数据下载校验继续执行。

统一 src/phase65.md，按65a/65b/65c追加每轮目标、改动、网络情况（模型及SSH/传输）、止损线、真实结果和artifact路径。未运行就写未运行。

所有远端 work_dirs 下的实验产物（包括既有 phase 的结果、checkpoint、日志和有效掩膜）需传回本地保留原相对路径；传输前核算磁盘容量，源目录只读。backup_phase65.py 打包不越界文件、拒绝外链，生成全文件SHA256清单；scp传回后本地校验，未校验前不得标记备份完成。大型历史目录可分批完成，明确未备份目录，不删远端产物。
