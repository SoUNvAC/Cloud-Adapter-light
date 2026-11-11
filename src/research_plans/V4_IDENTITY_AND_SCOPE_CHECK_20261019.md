# V4身份和验证范围核验

2026-10-19。只阅读本地Phase200记录、官方版本表／模型索引／加载代码。未运行模型、训练、读取影像像元或新增下载。

## 已核到

- 官方当前版本表列V4双成员为RegNet与EdgeNeXt，SMP＋timm后端；V1是RegNet与ConvNeXt，fastai后端。不能把两版decoder状态名称、参数量或BN数量当相同。
- 官方SMP分支使用`smp.Unet(encoder_name=..., encoder_weights=None, in_channels=3, classes=4)`后加载指定权重；公开接口具备重建同一模型的条件，实际训练白名单仍须执行端核验。
- Phase200已记录并执行过V4两权重的加载／同输入推理。锁定HF revision为`c9a4fb88188709127aa25cfe51ae7fd41b0132f8`，两文件合计57,805,992 bytes。
- RegNet SHA：`7f6e4202e17ee73efa4aba7abb5c34f4f90a9f7eb42480820714994dff2db660`；EdgeNeXt SHA：`d5fe67ad00f6fdb73eb8382ad925849e97fb82cedb46225344afff1a734a9c1d`。
- 官方列V4训练包含CloudSEN12 High／PC／SR／Scribble／2k、KappaSet和OCM hard negatives／scribble；官方说明部分OCM数据尚未公开。因此原V1预训练交集审计不能直接作为V4的完整无交集证明。

## 尚未独立核到

- Phase200受控shared实际权重本轮是否仍可读、当前文件SHA和runtime状态；本地身份记录不等于这次已重新核验二进制。
- SMP的实际decoder／head参数量、BN列表及可训练范围；没有从V1复制数值。
- V4自有未公开数据的完整产品／footprint列表；不能宣称Catalogue／PixBox与V4预训练完全独立。
- V4在原固定32组Catalogue及10组PixBox的完整原始／适配后结果。Phase200单例24候选点类别一致不回答此问题。

## 资源决定

按[Phase205](PHASE205_V4_ADAPTATION_DAMAGE_CHECK_20261019.md)先过实现门，再执行两次固定训练。主结论观察完整V4双模型，成员结果辅助；只适配RegNet、EdgeNeXt冻结的范围明确披露。没有重复损伤就结束本轮推广，不以V1结果代替。

来源：[本地Phase200身份](../../outputs/phase200/phase200_20261017/input_model_lock.json)；[官方版本表](https://github.com/DPIRD-DMA/OmniCloudMask/blob/main/MODEL_CHANGELOG.md)；[官方权重索引](https://github.com/DPIRD-DMA/OmniCloudMask/blob/main/omnicloudmask/model_download_links.csv)；[官方加载实现](https://github.com/DPIRD-DMA/OmniCloudMask/blob/main/omnicloudmask/model_utils.py)；[训练数据说明](https://omnicloudmask.readthedocs.io/en/latest/how-it-works.html)。网页反映本轮查询内容；实际执行锁定Phase200源码SHA，不能直接随main升级。
