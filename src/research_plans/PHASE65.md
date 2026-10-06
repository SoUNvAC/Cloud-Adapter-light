# Phase65：独立场景诊断 → 匹配信息对照 → 最小机制

预注册草案，2026-10-06；尚未读取新确认集、尚未训练。首次启动前必须完成 scene lineage 审计并冻结输入 SHA256。研究预算截至 2026-10-20（北京时间），这是投入上限，不保证完成；到期停止新作业，记录已完成与未完成项。

## 不可变协议

- 仅访问本地 D:/Cloud-Adapter-light，远端 /home/scv/Cloud-Adapter-light 与 /home/scv/shared。检查 realpath，拒绝链接越界。代码只在本地修改、验证、提交、push，然后远端 pull --ff-only。不得覆盖用户已有改动。
- 远端必须 conda activate cloud-lite-pt210。一次一个训练任务，work_dir 不复用、不覆盖。保存 git SHA、输入哈希、配置、环境、PID、退出码和原始日志。
- L8 仅 USGS Shadows?=yes（shadow?=1）；no、缺失、未知均排除。NoData/无效标签继续 ignore，不能当 Surface。三父类顺序 Surface/Cloud/Shadow，Thin 与 Thick 合并。雪冰、水体和失败场景不删除。
- 已有八场景只提供探索性假说，连同旧实验参与假说形成或调参的场景全数登记排除。现有 sealed test 保持封存；只允许读取 scene ID 清单用于排除，不读影像、标签、预测或指标。
- 新确认集从合格且具有可核验未使用 provenance 的目标训练池中按 scene 留出，不得看结果选场景。至少 8 个确认场景、8 个开发场景，否则报告不可识别并停止。按 SHA256('phase65:'+scene) 排序，前 8 个确认，其余开发。这是最低可行性约束，不代表统计功效已充分。
- 确认集只最终评估一次，训练/早停/模型选择不得使用。开发池内部按 scene 划分拟合/验证，冻结后再启封新确认集。增加 seed 不增加独立场景数。

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

统一 src/phase65.md，按65a/65b/65c追加每轮目标、改动、网络情况（模型及SSH/传输）、止损线、真实结果和artifact路径。未运行就写未运行。

所有远端 work_dirs 下的实验产物（包括既有 phase 的结果、checkpoint、日志和有效掩膜）需传回本地保留原相对路径；传输前核算磁盘容量，源目录只读。backup_phase65.py 打包不越界文件、拒绝外链，生成全文件SHA256清单；scp传回后本地校验，未校验前不得标记备份完成。大型历史目录可分批完成，明确未备份目录，不删远端产物。
