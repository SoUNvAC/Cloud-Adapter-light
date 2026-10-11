# Phase85：原LoRA方案的训练随机性复测

## 做什么

关闭Phase84固定特征读出，也暂不继续读出压缩分枝。回到Phase80的正向结果：检查LoRA＋光谱相对固定MsRE＋光谱的优势，是否能在两个预先指定的新训练seed上复现。

预算：**seed66、seed67各一次4000步LoRA训练，共两个训练、两个浅层拟合**。原seed65仅作历史参考，不挑选最好seed、不集成预测。

## 1．固定训练

1. 原Catalogue 153个fit组155产品用于训练；153个fit代表用于浅层拟合；32个development代表用于checkpoint选择及评价。产品、标签、有效mask和预处理均复用Phase80。
2. 两作业分别从原Source严格初始化，不能从已训练LoRA或Phase82学生续训。完全复用Phase80的rank=4、alpha=4、dropout=0、注入层、冻结范围、batch1、FP32、AdamW、学习率及调度；唯一计划变化是训练seed。
3. 分别在模型初始化、数据采样、增强和数据worker启动前设置seed66／67。记录实际配置、可训练参数及deterministic warn-only状态。不要把两作业误设成相同的数据随机流。
4. 每500步仅按原development池化mIoU选checkpoint，平手取最早；保存全部8个。禁止用云影AP改选或比较其他checkpoint的浅层结果。
5. 开跑前复用Phase80零残差Source一致性和可训练参数检查。一次一个GPU任务，记录时间、峰值显存、日志及权重SHA。数值失败记录失败，不追加seed替代。

## 2．每个seed一次光谱拟合

选定checkpoint后，在153个fit代表、32个development代表各推理一次，保存原云影分数、非云影胜者和三类预测。

- 仅拟合本seed的L＋S：网络云影logit＋B08/B11/B12。抽样索引、组权、BCE、正则0.001、优化器及系数约束全部复用Phase80。
- 标准化仅用fit；每个seed分别按原exact-ties规则校准fit全组平均FPR≤1%的阈值。
- **浅层抽样及优化设置不随训练seed改变**；两seed使用同一原fit抽样索引。网络分数不同导致系数与阈值不同是预期结果。
- 各seed的参数／阈值先落锁，再评价其development浅层结果。不能沿用seed65系数或阈值。

## 3．固定对照

原MM＝Phase80 MsRE＋光谱，保持原2000步网络和浅层模型；不重训MsRE。

每个新seed只构造两行：

| 组合 | 云影评分器 | Surface／Cloud读出 |
|---|---|---|
| LL_seed | 本seed冻结L＋S | 本seed LoRA |
| LM_seed | 同一个本seed L＋S | 原MsRE |

主要比较：每个新seed的LL_seed宏AP减原MM宏AP；LM_seed与LL_seed的全部云影指标必须一致。再比较LM_seed与MM的整体分割取舍。

先复现原MM、LL65、LM65逐组混淆矩阵；新增四行及三个历史参考完整报告，不只交付最好结果。

## 4．报告和继续条件

- 列宏／池化AP、冻结阈值宏召回、全组及有／无云影组FPR、单侧95%上界、宏／池化mIoU、三类IoU、零召回组数和最差损伤。附全部32组混淆矩阵、7组AP及各seed选中步骤。
- 每个新seed相对MM的AP差，按7个有云影相关组配对bootstrap10000次、seed85011，报告描述性区间。
- 同时报告两个新seed逐组AP差的均值，再在7个相关组上bootstrap；同一次抽样对两个seed使用相同组索引。不能把seed×场景当成14或21个独立场景，也不把两个seed当充分的随机性统计样本。
- 原seed65不进入本轮晋级平均值；两个新seed逐项列出，不选赢家、不用平均预测制造新方法。

固定探索继续条件，全部满足才保留当前LoRA／LM分枝：

1. 两个新seed相对MM的宏AP增量均>0，且两增量的算术平均≥1个百分点。
2. 每个新seed的宏召回相对MM下降均不超过1个百分点，全32组平均FPR点估计均≤1%。
3. 每个LM_seed的池化mIoU均不低于MM，宏mIoU相对MM下降均不超过1个百分点。
4. 两个seed的交叉云影不变量全部通过。

这些是开发集探索筛查，不能称正式非劣、稳定满足误报预算或独立确认。若失败，记录不稳健的具体指标，不追加seed、不改超参数或换checkpoint。本轮只检验**相对这一固定MsRE的条件稳健性**，不能概括成两种适配方法的总体优劣。

## 交付与边界

写入`outputs/phase85/phase85_20261011/`：短README、总表、逐组CSV、混淆矩阵、两训练配置／曲线／checkpoint选择记录、浅层锁、输入及模型SHA、成本和一致性检查。

不读取Phase66、65b、65c、旧sealed或ALCD，不取新数据，不增加蒸馏、特征或读出。现有development已反复使用；通过后仍需另立新数据验证协议，不能把旧确认池重新称作未见确认。LM部署需要两套网络，成本单列。

代码检查后按原流程push／pull执行，不覆盖历史、不自动同步work_dirs；沿用2026-10-20北京时间预算截止。本文件仅立项，未执行。
