# phase65 实验记录

## 65a / 2026-10-06 / 启动前审计
- 目标：在未参与假说形成/调参的独立场景确认混淆方向与净收益关系，以及部署可取得证据的风险识别能力。
- 改动：新增预注册草案、65a场景冻结审计、65b匹配对照检查、65c阶段门与work_dirs备份脚本。尚未建立模型/数据冻结锁。
- 网络情况：本地 Windows OpenSSH 执行 ssh gzs 无法解析主机名；未连接远端，GPU、conda、远端代码及训练状态均未核实。git推送情况待实测。模型计划沿用三父类源模型及固定MsRE，尚未训练。
- 止损线：截止2026-10-20；无可核验未用scene、不足8开发+8确认scene、封存集泄漏或标签规则错误均不启动；独立关系不稳定停止假说，不修改统计定义追显著。
- 结果：未产生实验指标。65b/65c未运行。远端work_dirs尚未备份，原因是SSH别名不可用。已有用户日志整理与outputs改动保持原状。

## 65a / 2026-10-06 / 远端真实就绪审计
- 目标：核验训练环境、scene使用历史和全部work_dirs备份条件。
- 改动：提交032b36c与a40dc23已本地push、远端pull --ff-only；增加phase65a_readiness.py，只读划分元数据和旧探索结果，不读取test像素；本地/远端5项协议检查均通过。
- 网络情况：初次ssh报错来自沙箱不能使用既有SSH配置；获准使用后gzs连接正常。指定cloud-lite-pt210环境为torch 2.1.0+cu121，CUDA=True，1张可见GPU；系统nvidia-smi不可用，GPU型号未核实。远端data链接解析为授权/home/scv/shared/data。
- 止损线：旧训练场景不可冒充独立确认集；封存test不解封；不足合格独立scene不启动。
- 结果：USGS metadata共32个Shadows?=yes、64个no。现有manifest的yes场景：train18/1560patch、val8/963patch、sealed-test6/747patch（只读清单统计，不评估test）。8个val就是旧探索场景；18个train已用于Phase64训练。现有可用独立scene=0，状态blocked_no_independent_confirmation，训练未启动，65b/65c未运行。此为数据就绪阻塞，不是H1/H2被统计否定。
- 产物：远端src/work_dirs/phase65a/readiness.json已传回本地outputs/phase65/readiness.json。manifest SHA256=885c2d7f61cae23409a7b1ccbe65739c302fc1e59b089aab0cc05badf674872b；shadow metadata=c735dc14ba5d40ffc0443ba9e5441d2716acd897c80f341a0093319f18862390；探索csv=d33275e1fdd7880e918d67d58476ab63a188ffec8c035dce4ca221654fc3fe51。
- 备份：远端src/work_dirs约34G；远端可用706G、本地D盘可用约188GB。全部work_dirs备份进程已启动，输出src/result_backups/phase65_20261006/all_work_dirs.tar及SHA清单；尚未传回/校验完整包，不能称备份完成。打包期间须保持work_dirs静止；若变化脚本拒绝认证。已创建每30分钟心跳phase65，负责完成传回、校验和记录，状态不变保持安静。

## 备份 / 2026-10-06 / 全量打包完成，传回中
- 目标：将所有既有work_dirs实验结果完整传回本地并核验。
- 改动：新增并经PowerShell语法检查的receive_phase65_backup.ps1，已本地push、远端pull；本地隐藏后台接收器PID保存在outputs/phase65/backup/RECEIVER_PID，避免重复传输。
- 网络情况：远端打包退出码0，包含5230个文件，约34GB；SHA manifest已传回。本地all_work_dirs.tar.partial正在下载。
- 止损线：下载失败保留partial，SHA或文件清单不匹配禁止标记完成；不删除远端源文件。
- 结果：仅打包和manifest传回完成，全量包仍传输中、本地逐文件校验尚未完成。心跳将检查接收器PID、TRANSFER_EXIT_CODE、BACKUP_VERIFIED并追加真实结论。

## 65a / 2026-10-06 / 用户指定新Sentinel数据，双机下载校验
- 目标：引入此前双机未下载的Sentinel-2 Cloud Mask Catalogue（Zenodo4172871），解除旧L8未用scene不足的入口阻塞；首次下载并不自动证明与源训练scene无重叠。
- 改动：新增download_phase65_sentinel.py、下载完整性/Range续传测试，以及metadata-only发布清单审计；本地提交da6aeba已push，远端pull后在cloud-lite-pt210执行。新数据入口登记于PHASE65.md，H1/H2终点和旧test封存不变。
- 网络情况：本地data/sentinel2_cloud_mask_catalogue_4172871与远端/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871分别从官方Zenodo下载；两端均实际连通。全部7文件共15354849917 bytes。下载进程与每10秒状态JSON已落盘；支持安全断点续传，缓存文件重新验证。
- 止损线：每文件官方MD5、长度及ZIP CRC必须通过；记录SHA256用于两机一致性；不把.partial当完成；未知/未标注shadow不是Surface负例；新源/目标产品、tile、时间、足迹交叉与场景冻结审计未通过，不训练。
- 结果：本地classification_tags.csv、README.pdf、masks.zip、alt_masks.zip、shapefiles.zip共5文件已验证，三个ZIP CRC通过。下载主影像包和缩略图尚未全部完成，不能声称双机完整校验通过。标签CSV核验513个唯一product，shadows_marked=1有424个、=0有89个；标注过程组MAIN453、VALIDATION50、CALIBRATION10不等于本研究训练测试split。仅424个有效cloud-shadow候选进入后续审核，排除89个会收窄困难场景覆盖，应报告这一边界。
- 数据协议：README确认mask为CLEAR/CLOUD/CLOUD_SHADOW顺序bool one-hot；影像1022×1022×13 float32 TOA、20m、波段数值序（B8A在B8与B9间），RGB索引[3,2,1]。标注含主观不确定性，不能将人工分歧解释为真值错误率。尚未审计数组像元或进行任何新模型评估。
- 产物：两端dataset根目录record.json/download_status.json；本地outputs/phase65/sentinel_download/PID、console.log、error.log；远端src/work_dirs/phase65a/sentinel_download/PID、console.log、EXIT_CODE；本地README提取文本outputs/phase65/sentinel_readme.txt。classification_tags SHA256=46b88e07480f36a4d2fa27f01a1ee9987b1def7ad08f1eb721a88ef0b5288d25。

### 2026-10-06 16:51 北京时间进度
- 双机均已验证6/7文件（增加thumbnails.zip，官方MD5/SHA256/ZIP CRC通过），均正在下载15197196473-byte主影像subscenes.zip；各自约48/50MB，完整数据尚未下载/校验完。
- 新下载与清单审计6项离线测试通过，原5项协议测试通过；所有变更已本地push、远端pull。每30分钟心跳已更新，负责下载失败恢复、两机全文件SHA一致性、发布清单审核及后续独立scene审计，禁止提前称完成。旧全work_dirs备份接收器继续运行。

## 65a / 2026-10-06 17:10 北京时间 / 定时巡检与source provenance准备
- 目标：确认双机下载/历史备份继续运行，准备新目标与源训练场景重叠审计。
- 改动：未重启或重复下载。新增prepare_phase65_source_metadata.py，仅下载官方CloudSEN12-high train/val metadata，冻结HF commit并校验Git blob SHA1或LFS SHA256；不下载test metadata、任何源像元/标签/指标。
- 网络情况：本地下载python PID3236仍存活、error.log无报错；remote launcher PID81799仍存活，无EXIT_CODE/错误日志。读取状态时本地主影像2043674624/15197196473 bytes，远端1760559104 bytes（随后stat为1777336320）；两端6/7已校验。原work_dirs备份partial已增长至511639552 bytes，仍未完整回本地校验。
- 止损线：校验未完成不训练；官方元数据未证实本地PNG映射则不能假定场景谱系完整，不能以新下载替代独立scene审计。
- 结果：本地source metadata成功获取train4759992 bytes、val300257 bytes；固定官方commit=d97728dd61858a21c7d6bc61385687d328433cba，Git blob校验通过。train SHA256=d4a4bb41fba0e39d49328db29868273d093f982e84716d7f2b836e852f5349f4；val=35e7afd8f5feafa6ce6b04415a3e4426ad147b5beed3f84db68839ac12d29097。只读远端源train文件名确认8490个，0..8489连续编号；官方CSV index为原始索引而非同一数字序列，精确转换映射仍需核验，不能直接用PNG文件名join官方index。
- 产物：本地outputs/phase65/source_metadata/acquired/中train_metadata.csv、val_metadata.csv、source_metadata_manifest.json。未读取新确认影像/标签，未训练，暂无实验性能指标。
