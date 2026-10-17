# 供Phase125使用的陈述边界

## 可支持

- 固定Catalogue池中，两次C100在各自冻结校准阈值下均出现较低组均误报及较高宏召回点估计；实际FPR不同，召回不确定性保留。
- 主表同一模型换阈值AP不变；校准改变决策工作点，不修复排序本身。
- Catalogue重复AP改善52UDG/35RPM，下降32TML/40XDR；PixBox重复改善33VVE/44SKJ，下降19FDV/54TWN（严格±1pp描述边界）。全部组及NA均交付。
- PixBox召回方向随seed变化，且两seed误报代价都增加。宏AP与池化AP可以给出不同趋势，均应报告。
- 119固定三景各两次的单R GPU模型耗时为E约36.4%，读入至取点分段估计为约93.2%；这两个口径必须同时保留。
- 单RegNet18,039,132参数；C100训练14,135,988参数（约78.4%），部署一个模型，需要基础权重及decoder overlay重建。
- 274条历史交付SHA、6个本地末步模型SHA和72项汇总算术已核验；本轮零模型调用、像元读取和区间重算。

## 不可支持

- “C100全面提升排序”“全面优于R”或“在评价侧相同FPR预算下已证明召回更好”。
- “两个seed已证明总体稳定”“seed数使独立组数翻倍”或“两seed指标平均是可部署集成模型”。
- “1%校准目标保证每景安全”或“PixBox点参考FPR代表全景FPR/三类mIoU”。
- “整景快一倍”“C100部署延迟与R已实测相同”或把控制器墙钟/调用总数当纯推理速度。
- “只训练微型补丁”“56.56MB decoder就是完整可独立部署模型”。
- “C100与C153差异证明训练组数量的因果作用”。
- “重复损伤已确定来自暗地表/某机制”，本轮没有新增影像判读或机制检验。
- “本轮是未见数据独立确认”或改写原固定主检验失败结论。训练地点交集、历史暴露及参考语义限制保持。

所有区间直接复用Phase123、bootstrap seed123015，未在124重算或另做子组检验。下一步仅供用户授权的Phase125分支总结使用；本轮未启动。


## 未测/未知项保持原状态

| item | status | detail | impact |
|---|---|---|---|
| C100_matched_latency_and_peak | unmeasured | no C100 x3scene x2repeat matched119 benchmark | cannot copy R latency/worker peak or claim C100 timing measured |
| C100_merged_standalone_weight_size | unmeasured | existing56.56MB file is decoder overlay; base required | base+overlay reconstruction128.87MB is not merged deployment file size |
| pretraining_geographic_overlap | unknown | unknown overlap with pretrained model data | no independent confirmation claim |
| original_Google_drive_tensor_equivalence | unknown | historical model reconstruction metadata says not verified | HF identity/accepted receipts do not prove original Google Drive equivalence |
| PixBox_full_scene_FPR_and_three_class_mIoU | unmeasured | only chosen negative reference point FPR | cannot extrapolate full-scene FPR or three-class mIoU |
| seed_population_variation | unmeasured | only two fixed seeds; conditional related-group intervals reused | cannot claim cross-seed population stability |
| training_group_count_causal_effect | unmeasured | C100 vs C153 changes data membership and seed | cannot identify data-size effect alone |
| original_E_R_N_binary_reverification | unknown_this_audit | original author binaries not delivered in119-123 local model overlays;119 asset/resource identity receipts reused | do not call metadata identity a new original-weight byte verification |

本分支收口不等于所有研究问题已解决。未进行新的独立确认；后续新方向从Phase126起，须按对应授权协议推进。
