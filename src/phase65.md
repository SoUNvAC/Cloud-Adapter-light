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
- 补充：远端同样获取官方train/val metadata，commit、bytes、SHA256与本地一致。新增audit_phase65_source_ids.py并运行：9025条source train/val metadata、8946唯一products，对424个shadow-valid目标候选，exact product重叠0、tile+acquisition重叠0、同tile任意日期重叠26。仅为ID层初审，不等于无空间重叠；本地PNG映射、空间footprint、历史谱系和独立group划分仍未完成。不删除或按模型表现筛选候选。可复现产物outputs/phase65/source_metadata/provisional_overlap.json。

## 65a / 2026-10-06 17:40–17:48 北京时间 / 下载巡检与source空间审计
- 目标：保持双机下载和历史备份正常运行，补齐源PNG谱系与真实空间重叠证据。
- 改动：新增audit_phase65_source_png_mapping.py和audit_phase65_spatial_overlap.py；均本地检查/提交/push、远端pull后在cloud-lite-pt210执行。缺少的pyshp2.3.1仅放置src/work_dirs/phase65a/geospatial_python，临时目录亦在授权项目内；既有conda/系统安装未修改，安装报告已备份。
- 网络情况：17:40读取时本地主影像4929355776/15197196473 bytes，远端5014290432 bytes，两端6/7文件已校验、原进程存活、无错误；未重复启动。旧全work_dirs备份partial=999424000 bytes，接收器存活，完整回传/校验未完成。
- 止损线：源PNG行序不匹配、官方RGB SHA不符则不承认scene映射；source footprint重叠候选不得进入独立confirmation；空间、历史目标谱系及group划分未冻结，不训练。
- 结果：源train8490/8490、val535/535 PNG均与对应官方原始RGB逐像元完全一致；6个原始RGB dat的官方LFS SHA256均通过。CSV按发布行序与raw/PNG ordinal对应，不能按原始index字段join。此证据解决原PNG行序映射缺口，不涉及标签、test或目标像元。
- 空间结果：以原生UTM几何加100m保守缓冲后跨CRS检查，source train/val共有1805个唯一footprint；424个shadow-valid新候选中6个与source footprint重叠。无同产品/同采集ID不代表无空间重叠。状态source_spatial_audited_split_not_locked；尚未审计历史目标footprint/相关产品group、完整下载和数组loader，未冻结split、未训练，无新性能指标。
- 备份：源映射和source_provenance新增work_dir结果已按目录层级传回outputs/phase65/work_dirs/phase65a/，与远端SHA核对一致；mapping_audit SHA256=c3c521a375c50f81ff0e58682621c7fcb72fc47472a7b37931c587036f12d0bb，spatial_overlap=4e9bef32a0f17d8119cb3f06c79d8dcf6db4113c0c17b246721c9268fbba715d。source映射EXIT_CODE=0。历史34GB包仍传回中，不称全量备份完成。

## 65a / 2026-10-06 18:10–18:20 北京时间 / 历史目标footprint和相关产品分组
- 目标：检查正常传输，补齐旧目标空间排除，避免相关Sentinel产品跨split。
- 改动：新增audit_phase65_historical_footprints.py、prepare_phase65_scene_groups.py及3项有意义的分组独立性测试；本地提交push、远端pull。分组规则在新预测/指标之前写入PHASE65.md，暂不锁定split。
- 网络情况：18:10读取时本地主影像7826571264/15197196473 bytes、远端7574913024 bytes，两端仍6/7文件已校验、下载进程存活无错误，未重启。历史work_dirs备份partial=1473413120 bytes，接收器存活无错误，未完成全量回传。
- 止损线：未知CRS或未解析历史footprint不进入确认集；任何source tile/footprint或历史used-scene footprint重叠，整个相关group排除；仅metadata准备完成不授权训练，禁止将产品数量作为独立scene样本量。
- 结果：首轮历史审计解析134/142，8个场景因首版只接受UTM而blocked（产物保留）。检查头信息确认8个均为米制EPSG:3031；修复为按已知projected CRS及单位换算、101点加密重投影后，142/142解析完成，无未解析项，11个新shadow-valid候选与历史train/val的保守bounds重叠。仅打开train/val TIFF头，未读取任何栅格像元、标签或sealed test文件。
- 分组：513产品按同MGRS tile或同采集date+relative orbit做传递闭包，得到493组；源tile任意日期/源footprint/历史footprint及shadow-invalid-only组排除后，377候选组含384个shadow-valid产品，116组排除。89个shadows_marked=0不作为样本，只可作为相关性连接节点。没有按difficulty、真值比例、预测或性能筛选。H1/H2支持分母和group-based模型选择/统计代表规则仍须预先定稿，完整下载、数组/loader审核及冻结仍待完成，训练未启动。
- 验证与备份：3项分组测试本地/远端通过，原5项协议测试本地通过；两机完整group成员及排除标记相同，其canonical SHA256=a5a47c521f54adc1b5cdbc5eea1ab763a9bbb8e20429c37cccd1f8cd678c9acd。新work_dir结果已按原层级回传outputs/phase65/work_dirs/phase65a/source_provenance/并核对SHA：historical_overlap首版35ca7899679bc5333883bba33988617f0d55e74e38fdf2698ad3940e48365dd6；v2=5643bc9c3be55b8a9d467c6bed50d26f9cc4c702a745c9683e40e784c20a1ea5；远端scene_groups备份=41251fcc6177acd42782e2974cd76c82f333a14d0af65ee87aeb10114e79cb57。尚无新模型性能指标。

## 65a / 2026-10-06 18:40–18:53 北京时间 / 预分析支持门停止
- 目标：在新预测/性能分析前冻结独立组划分、统计代表、H1支持域和source checkpoint，检查是否允许进入训练准备。
- 改动：PHASE65.md保留旧稿并明确预分析修订理由；按固定SHA顺序划分64组65a、64组65b、64组65c、32组开发验证、153组fit，固定每组代表。新增freeze_phase65_sentinel_split.py及3项划分/支持门测试，本地和远端均通过；代码已本地push、远端pull。失败方案写入split_lock.json，不重抽或放宽支持门。
- 网络情况：18:52 SSH正常，远端HEAD=2a5b8b9、cloud-lite-pt210检查已完成；本地PID3236及远端launcher PID81799继续下载，两端6/7文件已验证，subscenes.zip.partial分别11923357696和11805917184/15197196473 bytes，无下载错误。旧34GB备份接收器PID24076仍活跃，partial=2098757632 bytes，未完成全包/逐文件校验。
- 止损线：预先设定fit和65a_confirmation各至少16个固定代表的public shadow_percent>0，任一不足则停止主检验；该真值支持字段仅用于离线分析支持域，不得作部署风险输入。不借用65b/65c、不改相关性或抽样追显著；下载与安全备份继续。
- 结果：fit支持41、65a_confirmation支持15（64个固定代表中），低于所需16。状态stopped_insufficient_h1_support，training_authorized=false；未训练、未读取新模型预测或目标像元，未估计相关性、AUROC或分割收益。此为支持样本门不足，不是H1/H2的实证否定；65b/65c不得推进。
- 冻结和备份：source checkpoint SHA256=64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9；失败方案含377组完整固定分配及预注册SHA，远端src/work_dirs/phase65a/split_lock.json已传回outputs/phase65/work_dirs/phase65a/split_lock.json，两端SHA256=a431745e7a17bbd4a7b9e12ef3bf76767d4eb827911bd5cb5e632683c2177f7b一致。完整发布清单审核仍须等待双机7文件全部验证，不能称数据或历史全量备份完成。

## 65a / 2026-10-06 19:10 北京时间 / 传输巡检
- 目标：研究支持门停止后，继续完成已授权双机数据校验与全部work_dirs备份。
- 改动：仅检查进程、状态及日志，不重复下载/传输，不修改失败划分或启动训练；保留用户现有改动。
- 网络情况：SSH正常，远端代码1ce6fce；双机下载进程仍活跃、日志无错误、partial持续增长，两端均6/7已验证。本地主影像状态13676576768/15197196473 bytes，远端13484687360/15197196473 bytes；本地备份接收器活跃，partial=2383511552 bytes，transfer.log无错误。
- 止损线：7文件未全部校验不做双机完整清单审核；完整历史包未逐文件SHA校验不标记备份完成；固定15/16支持门继续禁止训练及65b/65c。
- 结果：下载和备份尚未完成，未发现需要恢复的失败；BACKUP_VERIFIED及TRANSFER_EXIT_CODE均尚不存在，无新模型预测或实验指标。下一轮继续检查全文件校验条件。

## 65a / 2026-10-06 19:40–19:43 北京时间 / 双机发布文件完整校验通过
- 目标：完成授权Sentinel下载、双机一致性核验及metadata-only清单审计，备份本轮远端work_dir产物。
- 改动：使用已push/pull的audit_phase65_sentinel_catalogue.py分别在本地及远端cloud-lite-pt210运行；互传download_status并检查7文件bytes、固定官方MD5、SHA256、ZIP CRC一致。不改失败划分，不训练。
- 网络情况：SSH/scp正常；远端下载完成于UTC11:30:50，EXIT_CODE=0；本地完成于UTC11:39:04，下载进程已退出、error.log无错误。两端all_files_verified。历史34GB包接收器PID24076仍活跃，19:40读取partial=2813460480 bytes、transfer.log无错误，完整包尚未传回校验。
- 止损线：完整发布文件校验不能替代独立scene或训练管线审核；15/16支持门保持停止，65b/65c不推进；历史包BACKUP_VERIFIED尚不存在，不称全量备份完成。
- 结果：7/7文件长度/官方MD5通过，全部ZIP CRC通过且保存SHA256，两机逐文件一致。主影像15197196473 bytes、MD5=0ad1de0ebeaff529782f456cad2e966f、SHA256=04949b0d04250ace486386ac166caa4825289cd9d4f1bff67fa323ba23eaca88。两端清单审计通过：513唯一products与影像/主mask ZIP成员一致；424 shadows_marked=1（421 tiles），89无可靠shadow标注排除；MAIN453/VALIDATION50/CALIBRATION10仅标注阶段。未读取目标像元/标签或产生模型预测/性能指标。
- 产物与备份：本地outputs/phase65/sentinel_download/remote_download_status.json及catalogue_audit.json；远端work_dirs/phase65a/sentinel_download/catalogue_audit.json、AUDIT_EXIT_CODE=0。该目录7个文件已传回outputs/phase65/work_dirs/phase65a/sentinel_download并逐文件SHA核对全部一致，清单保存outputs/phase65/sentinel_download/remote_workdir_sha256.txt；远端审计报告SHA256=6cdb58b0b069e706e6bb5b9f585eb72d3d74d3b294758b9cfd32eea1d2c23645。审计状态release_verified_not_training_ready；其通用pending列表不是既有source审计失败，也不覆盖已记录的支持门停止状态。

## 65a / 2026-10-06 20:10 北京时间 / 校验后备份巡检
- 目标：保持历史全work_dirs备份推进，确认已完成下载及冻结失败方案未变化。
- 改动：仅只读巡检和记录；初次未激活conda的状态查询提示python不存在，随即按既有hook激活cloud-lite-pt210重查成功，不修改环境。不重复下载、传输或清单审计。
- 网络情况：SSH正常，远端HEAD=250dac6；两机all_files_verified、7文件已验证，远端下载与清单审计退出码均0；本地error.log无错误。历史包接收器PID24076活跃，partial增至3266379776 bytes，transfer.log无错误。
- 止损线：全包及逐文件校验前不称历史备份完成；支持门停止不变，不启动训练或65b/65c。
- 结果：远端失败划分及审计报告SHA与既有本地备份一致；历史备份尚未完成，无新模型预测或指标，无需恢复进程。

## 65a / 2026-10-06 20:40 巡检 / 历史备份断线恢复
- 目标：恢复中断的全work_dirs备份，保留已接收内容并验证安全续传。
- 改动：receive_phase65_backup.ps1新增远端前缀SHA校验与SSH keepalive；首轮SFTP reget在打开本地大partial时失败，保留全部失败记录。新增receive_phase65_archive.py，以Python二进制文件IO接收SSH远端tail流，独立再次校验已有前缀与最终长度，完成后仍执行原完整归档SHA及5230文件逐项核验。本地PowerShell语法及Python编译检查通过，5f42673/f4a3627均已push、远端pull后启动。
- 网络情况：原receiver PID24076已退出，transfer.log明确Connection reset/Broken pipe，partial=3295805440 bytes。SSH重新连通；保留前缀本地与远端SHA256=ca93a3e0fe7ad146ad638d7db4ffcb071b14b7f2685c4fdd86080ff5d05cec29一致。SFTP尝试PID31176确认退出后，隐藏接收器PID20696启动；实际读取partial已增长至3311534080 bytes，未出现新的TRANSFER_EXIT_CODE。
- 止损线：进程退出才恢复；前缀不匹配禁止追加，不覆盖partial、不删除远端结果。完整归档及逐文件SHA未通过不写BACKUP_VERIFIED；固定研究支持门不变，继续禁止训练及65b/65c。
- 结果：断线已诊断，SSH二进制续传已实际恢复，历史备份尚未完成。双机7文件仍all_files_verified、清单审计退出码0，失败划分和审计报告SHA与本地备份一致，无新实验指标。原失败PID/退出码保存为outputs/phase65/backup/*failed_20261006_2040，日志保留。

## 65a / 2026-10-06 21:10 巡检 / 长连接超时后分段续传
- 目标：恢复再次超时的历史全量备份，缩短单次SSH连接时长。
- 改动：receive_phase65_archive.py改为每连接最多32MiB的GNU dd字节范围读取，严格核对各段长度，断线保留已收前缀；原完整SHA/逐文件校验仍必需。新增3项测试覆盖精确分段续传、前缀不符禁止追加、断线保留已收字节；本地通过。首次远端测试因测试目录尚不存在而失败，修复夹具显式创建授权目录后两端3/3通过。493acc9/74611a1均本地push、远端pull后使用。
- 网络情况：SSH长连接报Timeout, server frp-gap.com not responding，旧PID20696已退出、TRANSFER_EXIT_CODE=1，partial=3974168576 bytes。重新连接成功，远端HEAD=74611a1；前缀两端SHA256=f04df397c5fe0da103788e3be919317dad683f84cdcced2f4c5f36cb48f6b371一致。新隐藏接收器PID32212活跃，partial已增长至3976265728 bytes，无新TRANSFER_EXIT_CODE；尚不能声称分段方案已证明长期稳定。
- 止损线：旧接收器退出才恢复；前缀错或段长度错停止，不覆盖已收数据、不删除远端结果。完整包及逐文件校验前禁止标记备份完成；研究支持门保持停止。
- 结果：分段续传已实际开始，历史全量备份未完成。双机7文件仍all_files_verified、下载及审计退出码0，失败方案与审计报告SHA未变；无新模型预测/训练或性能指标。原超时PID及退出码保存outputs/phase65/backup/*timeout_20261006_2110，原日志保留。

## 65a / 2026-10-06 21:40 北京时间 / 分段备份巡检
- 目标：确认分段续传推进，保持已校验数据和研究停止状态。
- 改动：仅检查进程、文件长度、日志与远端SHA，不重启接收器、不重复下载/清单审计，保留用户改动。
- 网络情况：SSH正常，远端HEAD=73bede8；接收器PID32212活跃，partial增至4469096448 bytes，相比上轮3976265728继续增长；日志无新增错误，当前TRANSFER_EXIT_CODE尚不存在。分段续传本轮仍运行，不据单轮观察保证长期稳定。
- 止损线：全包及逐文件SHA通过前不写BACKUP_VERIFIED；支持门停止不变，不训练或推进65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载/审计退出码均0，失败方案及审计报告SHA与既有备份一致；历史全量备份未完成，无新实验指标，无需恢复进程。

## 65a / 2026-10-06 22:10 北京时间 / 备份续传巡检
- 目标：继续检查历史全量备份，确认已验证数据及冻结记录不变。
- 改动：仅巡检和追加记录，不重复下载、审计或启动接收器，保留用户改动。
- 网络情况：SSH正常，远端HEAD=8860b24；接收器PID32212存活，partial=4782620672 bytes，比上轮4469096448增加313524224 bytes；transfer.log无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均尚不存在。
- 止损线：完整包及逐文件SHA校验前不标记备份完成；15/16支持门停止保持不变，不训练或推进65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载与清单审计退出码均0；失败划分及审计报告SHA与既有本地备份一致。全量备份仍传输中，无新模型预测或性能指标，无需恢复进程。

## 65a / 2026-10-06 22:40 北京时间 / 备份续传巡检
- 目标：持续检查历史备份传输及既有校验状态。
- 改动：仅巡检和记录，不重复启动下载/接收器或重跑已完成清单审计，用户改动保持原状。
- 网络情况：SSH正常，远端HEAD=7b6b144；接收器PID32212存活，partial=4945149952 bytes，比上轮4782620672增加162529280 bytes；transfer.log无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA未通过不标记备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：两机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；全量备份继续传输，暂无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-06 23:10–23:14 北京时间 / 备份远端断线恢复检查
- 目标：恢复再次被远端关闭的历史备份连接，严格保留已收内容。
- 改动：使用现有分段续传脚本，不修改代码或研究方案；确认PID32212已退出、TRANSFER_EXIT_CODE=1后，将原PID/退出码保留为outputs/phase65/backup/*disconnect_20261006_2310，启动隐藏接收器PID24452。
- 网络情况：日志明确Connection to frp-gap.com closed by remote host和segment incomplete；partial保留5311561728 bytes。SSH巡检正常，远端HEAD=7bb6b5b。恢复脚本第一层前缀校验通过，SHA256=74ad8e19dc970fc92568f2a53bb7e8a81f5279d21bdfeeacb44472105bdbcdb2；Python接收辅助进程已启动，独立第二层前缀校验尚无完成输出，partial暂未增长，新接收器活跃、无新TRANSFER_EXIT_CODE。
- 止损线：第二层校验通过才追加；已有前缀不匹配或段长度错误停止，完整包及逐文件SHA未通过不称备份完成；研究支持门停止不变。
- 结果：恢复流程已启动，但本轮尚不能确认实际续传恢复，下一轮检查第二层校验结果及partial增长。双机7文件仍all_files_verified，远端下载/审计退出码均0，失败方案与审计报告SHA未变；无新训练或实验指标。进程详情CIM只读查询被系统拒绝，改用Get-Process检查存活，未修改权限或系统。

## 65a / 2026-10-06 23:40 北京时间 / 恢复后的备份巡检
- 目标：确认上轮第二层前缀校验及实际续传状态，继续安全备份。
- 改动：仅巡检与记录，不重复启动接收器、下载或清单审计，用户改动保持原状。
- 网络情况：SSH正常，远端HEAD=e1040f7；接收器PID24452活跃。日志确认第二层SSH前缀SHA校验通过，5311561728-byte前缀SHA与上轮一致；partial已增长至5823266816 bytes，增加511705088 bytes。日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整包及逐文件SHA通过前不称全量备份完成；15/16支持门停止不变，禁止训练及65b/65c。
- 结果：实际续传恢复已确认，全量包仍传输中。两机7文件仍all_files_verified，远端下载/审计退出码0，失败方案及审计报告SHA未变；无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 00:10 北京时间 / 备份续传巡检
- 目标：持续确认历史全量备份传输及既有校验状态。
- 改动：仅巡检与记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH正常，远端HEAD=eeccf26；接收器PID24452活跃，partial=6740770816 bytes，比上轮5823266816增加917504000 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门停止保持不变，禁止训练及65b/65c。
- 结果：两机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；全量备份仍传输中，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 00:40 北京时间 / 备份续传巡检
- 目标：持续检查历史备份传输和冻结产物状态。
- 改动：仅巡检与记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH正常，远端HEAD=3cfdf2b；接收器PID24452活跃，partial=7061635072 bytes，比上轮6740770816增加320864256 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整包及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：两机7文件仍all_files_verified，远端下载/审计退出码0，失败划分与审计报告SHA未变；历史全量包继续传输，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 01:10 北京时间 / 备份续传巡检
- 目标：持续确认历史全量备份及已校验产物状态。
- 改动：仅巡检与记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH正常，远端HEAD=7b2c937；接收器PID24452活跃，partial=7596408832 bytes，比上轮7061635072增加534773760 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门停止不变，不训练或推进65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；历史包继续传输，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 01:40 北京时间 / 备份续传巡检
- 目标：确认历史备份持续推进及已校验产物状态。
- 改动：仅巡检与记录，不重复启动接收器、下载或清单审计，用户改动保持原状。
- 网络情况：SSH正常，远端HEAD=774b569；接收器PID24452活跃，partial=7838629888 bytes，比上轮7596408832增加242221056 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整包及逐文件SHA通过前不称备份完成；15/16支持门停止不变，禁止训练及65b/65c。
- 结果：两机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；全量包继续传输，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 02:10 北京时间 / 备份续传巡检
- 目标：持续检查历史全量备份与既有数据校验状态。
- 改动：仅巡检与记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH正常，远端HEAD=6a5533d；接收器PID24452活跃，partial=8089239552 bytes，比上轮7838629888增加250609664 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：两机7文件仍all_files_verified，远端下载/审计退出码0，失败划分与审计报告SHA未变；历史包继续传输，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 02:40 北京时间 / 备份超时恢复流程
- 目标：恢复超时的历史全量备份，保留已有partial和失败证据。
- 改动：沿用现有分段接收脚本，不修改代码或研究划分；确认旧PID24452已退出、TRANSFER_EXIT_CODE=1后，将原PID/退出码保留为outputs/phase65/backup/*disconnect_20261007_0240，启动隐藏接收器PID19920执行前缀校验再续传。
- 网络情况：日志明确Timeout, server frp-gap.com not responding及segment incomplete；partial=8091533312 bytes，比上轮8089239552增加2293760 bytes后中断。SSH巡检重新连通，远端HEAD=beece8f；恢复进程已启动，前缀校验及恢复增长尚待确认，不能声称已恢复实际传输。
- 止损线：前缀SHA不匹配禁止追加，旧进程退出才恢复；完整归档及逐文件SHA未通过不称备份完成；15/16支持门仍禁止训练及65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载/审计退出码均0，失败划分及审计报告SHA未变；全量备份未完成，下一轮检查新接收器和partial增长，无新模型预测或实验指标。

## 65a / 2026-10-07 03:10 北京时间 / 备份远端断线恢复流程
- 目标：继续恢复不稳定SSH链路上的历史备份，保留新增已接收字节。
- 改动：沿用已验证分段脚本，不改代码或研究划分；确认PID19920退出且TRANSFER_EXIT_CODE=1后，保留原PID/退出码为outputs/phase65/backup/*disconnect_20261007_0310，启动隐藏接收器PID12404，重新校验前缀后续传。
- 网络情况：上轮8091533312-byte前缀两层SHA校验均通过，SHA256=263e25424b6009616a58fb0c1b6ccda7cedfb17fea1592ebaeb9f6ac45c4fd2b，实际续传到8198881280 bytes（增加107347968）后日志报Connection to frp-gap.com closed by remote host及segment incomplete。SSH巡检已重新连通，远端HEAD=26239dc；本轮新恢复流程尚未确认前缀校验完成或增长。
- 止损线：旧进程退出才恢复，前缀不符禁止追加；完整归档及逐文件SHA未通过不称备份完成；15/16支持门保持停止，禁止训练及65b/65c。
- 结果：全量备份未完成，恢复流程已启动，继续检查增长及退出码。两机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；无新训练、预测或实验指标。

## 65a / 2026-10-07 03:40 北京时间 / 恢复后的备份巡检
- 目标：确认上轮恢复流程通过前缀校验并实际继续传输。
- 改动：仅巡检与记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH正常，远端HEAD=4bd9cdc；接收器PID12404活跃。8198881280-byte前缀两层SHA校验通过，SHA256=3d3eeefafc3715dd0e3fc047ab0a5e9f81d62743fc9dfebf30df7e497c33be6b；partial已增长至8389722112 bytes，增加190840832 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：实际续传恢复已确认，但全量包仍传输中。双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分与审计报告SHA未变；无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 04:10 北京时间 / 备份续传巡检
- 目标：持续确认历史全量备份传输与既有校验状态。
- 改动：仅巡检与记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH正常，远端HEAD=e278c6e；接收器PID12404活跃，partial=8854241280 bytes，比上轮8389722112增加464519168 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分与审计报告SHA未变；全量包仍传输中，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 04:40 北京时间 / 备份续传巡检
- 目标：持续确认历史备份传输和冻结产物状态。
- 改动：仅巡检与记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH正常，远端HEAD=c382d8e；接收器PID12404活跃，partial=9091219456 bytes，比上轮8854241280增加236978176 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分和审计报告SHA未变；历史包继续传输，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 05:10 北京时间 / 备份远端断线恢复流程
- 目标：继续恢复历史全量备份，保留已收数据与断线记录。
- 改动：沿用现有分段脚本；确认旧PID12404退出、TRANSFER_EXIT_CODE=1后，将原PID/退出码保留为outputs/phase65/backup/*disconnect_20261007_0510，启动隐藏接收器PID34600进行前缀校验再续传。不改研究方案或代码，保留用户改动。
- 网络情况：日志明确Connection to frp-gap.com closed by remote host及segment incomplete；partial=9331638272 bytes，比上轮9091219456增加240418816 bytes后中断。SSH巡检已重新连通，远端HEAD=43dee97；本轮新恢复流程尚未确认前缀校验完成或文件增长。
- 止损线：旧进程退出才恢复，前缀SHA不符禁止追加；完整包及逐文件SHA未通过不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：历史全量备份仍未完成，恢复流程已启动，下一轮检查新接收器及增长。两机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；无新训练、模型预测或实验指标。

## 65a / 2026-10-07 05:40 北京时间 / 恢复后的备份巡检
- 目标：确认上轮前缀校验及实际续传恢复，继续安全备份。
- 改动：仅巡检与记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH正常，远端HEAD=d17ac56；接收器PID34600活跃。9331638272-byte前缀两层SHA通过，SHA256=34ff12dc455b3d986ff5dc0746273e89eaeddf5a9d62482498dd1fe0f2840428；partial已增长至9572810752 bytes，增加241172480 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整包及逐文件SHA通过前不称备份完成；15/16支持门保持停止，禁止训练及65b/65c。
- 结果：实际续传恢复已确认，全量包仍传输中。双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 06:10 北京时间 / 备份续传巡检
- 目标：持续确认历史备份传输及既有校验状态。
- 改动：仅巡检与记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH正常，远端HEAD=6aac5e1；接收器PID34600活跃，partial=9815031808 bytes，比上轮9572810752增加242221056 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分和审计报告SHA未变；历史全量包继续传输，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 06:40 北京时间 / 备份续传巡检
- 目标：持续检查历史全量备份与既有校验记录。
- 改动：仅巡检与记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH正常，远端HEAD=3d7c0b0；接收器PID34600活跃，partial=10122264576 bytes，比上轮9815031808增加307232768 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；全量包继续传输，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 07:11 北京时间 / 备份续传巡检
- 目标：持续确认历史全量备份及已校验产物状态。
- 改动：仅巡检与记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH正常，远端HEAD=5f675e9；接收器PID34600活跃，partial=10502897664 bytes，比上轮10122264576增加380633088 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整包及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分和审计报告SHA未变；历史全量包仍传输中，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 07:41 北京时间 / 备份续传巡检
- 目标：持续确认历史全量备份及冻结产物状态。
- 改动：仅巡检与记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH正常，远端HEAD=e2caba3；接收器PID34600活跃，partial=10833199104 bytes，比上轮10502897664增加330301440 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整包及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；全量包仍传输中，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 08:11 北京时间 / 备份续传巡检
- 目标：持续检查历史全量备份及既有校验记录。
- 改动：仅巡检与记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH正常，远端HEAD=81e12e2；接收器PID34600活跃，partial=11079614464 bytes，比上轮10833199104增加246415360 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分和审计报告SHA未变；历史全量包仍传输中，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 08:41 北京时间 / 备份续传巡检
- 目标：持续确认历史全量备份和既有校验记录。
- 改动：仅巡检与追加记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH正常，远端HEAD=5e995d5；接收器PID34600活跃，partial=11354341376 bytes，比上轮11079614464增加274726912 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整包及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：两机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；历史全量包继续传输，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 09:11 北京时间 / 备份续传巡检
- 目标：持续确认历史全量备份和冻结产物状态。
- 改动：仅巡检与追加记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH正常，远端HEAD=ecbe991；接收器PID34600活跃，partial=11872337920 bytes，比上轮11354341376增加517996544 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；历史全量包继续传输，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 09:41 北京时间 / 备份续传巡检
- 目标：持续确认历史全量备份和已校验产物状态。
- 改动：仅巡检与追加记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH正常，远端HEAD=8999a8f；接收器PID34600活跃，partial=12171182080 bytes，比上轮11872337920增加298844160 bytes；未出现新增传输失败，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。SSH日志新增其自动记录转发IP的host-key提示，未手工访问或修改SSH配置、凭据或其他目录。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：两机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；历史全量包继续传输，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 10:11 北京时间 / 备份连接重置恢复流程
- 目标：恢复历史备份连接，保持只使用既有SSH主机信任的访问边界。
- 改动：接收脚本与Python helper加入CheckHostIP=no、UpdateHostKeys=no、StrictHostKeyChecking=yes及20秒连接超时，防止转发IP变动导致自动新增host-key记录，仍校验既有主机密钥；不读取或修改SSH配置/凭据文件。PowerShell语法检查及3项续传测试本地通过，39177f6已本地push、远端pull后使用。前缀/段长度及最终完整SHA规则不变。
- 网络情况：日志明确client_loop: send disconnect: Connection reset及segment incomplete；旧PID34600退出、TRANSFER_EXIT_CODE=1，partial=12419039232 bytes，比上轮12171182080增加247857152 bytes。SSH严格主机校验连接及远端pull成功。原PID/退出码保存outputs/phase65/backup/*disconnect_20261007_1011，新隐藏接收器PID14724启动，前缀校验和恢复增长尚待确认。
- 止损线：旧进程退出才恢复，前缀SHA不符禁止追加；未知主机密钥拒绝连接，不自动写入；完整包及逐文件SHA未通过不称备份完成；研究支持门停止保持不变。
- 结果：历史全量备份未完成，恢复流程已启动。双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；无新训练、模型预测或实验指标。

## 65a / 2026-10-07 10:41 北京时间 / 恢复后的备份巡检
- 目标：确认上轮前缀校验通过并实际恢复续传，继续安全备份。
- 改动：仅巡检与追加记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH严格既有主机校验连接正常，远端HEAD=d1a550f；接收器PID14724活跃。12419039232-byte前缀两层SHA均通过，SHA256=7053511e48e3adc13ab3e867f169b4fe9dbf865ab78ab70ff180173171b5a6dc；partial已增长至12519702528 bytes，增加100663296 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：实际续传恢复已确认，全量包仍传输中。两机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 11:11 北京时间 / 备份握手断线恢复流程
- 目标：恢复历史备份连接，保留已有partial及失败证据。
- 改动：沿用现有严格主机校验分段脚本；确认旧PID14724退出、TRANSFER_EXIT_CODE=1后，将原PID/退出码保存outputs/phase65/backup/*disconnect_20261007_1111，启动隐藏接收器PID40868进行前缀校验再续传。不修改代码或研究划分，保留用户改动。
- 网络情况：日志报kex_exchange_identification: Connection closed by remote host及segment incomplete；partial=12653920256 bytes，比上轮12519702528增加134217728 bytes后失败。常规SSH巡检成功，远端HEAD=820cb3d；本轮新恢复流程尚未确认前缀校验完成或增长。
- 止损线：旧进程退出才恢复，前缀SHA不符禁止追加；完整包及逐文件SHA未通过不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：历史全量备份未完成，恢复流程已启动，下一轮检查新接收器及增长。双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；无新训练、模型预测或实验指标。

## 65a / 2026-10-07 11:41 北京时间 / 备份查询握手失败恢复
- 目标：恢复未产生新字节的握手失败，减少瞬时SSH查询错误造成的提前退出。
- 改动：receive_phase65_archive.py对远端长度/前缀SHA查询的SSH退出255加入最多3次尝试、10/20秒退避；校验不符、非255失败仍停止，校验完成前不追加。新增瞬时失败恢复及重试耗尽不追加测试，共5项两端通过。7d1f7d6本地push后首次远端同步握手失败，再次连接pull --ff-only成功后才启动新版接收器。
- 网络情况：上轮第一层12653920256-byte前缀SHA通过，SHA256=736b1edfda2d775defe5765c0897b9c44cee083c58bf1a2b2bc9879864c2760b；随后helper查询远端长度时kex_exchange_identification: Connection closed by remote host、退出255。旧PID40868已退出、TRANSFER_EXIT_CODE=1，partial仍12653920256 bytes，无新增字节。原PID/退出码保存outputs/phase65/backup/*handshake_20261007_1141，新隐藏接收器PID33192启动，实际校验/增长尚待确认。
- 止损线：有限重试不绕过主机或前缀校验；完整归档及逐文件SHA未通过不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：恢复流程已启动，历史全量备份未完成。常规SSH审计此前成功，双机7文件仍all_files_verified、远端下载/审计退出码0，失败划分与审计报告SHA未变；无新训练、模型预测或实验指标。

## 65a / 2026-10-07 12:11 北京时间 / 恢复后的备份巡检
- 目标：确认恢复校验及实际续传增长，继续安全备份。
- 改动：仅巡检与追加记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH严格既有主机校验连接正常，远端HEAD=c83039f；接收器PID33192活跃。12653920256-byte前缀两层SHA校验通过，SHA256=736b1edfda2d775defe5765c0897b9c44cee083c58bf1a2b2bc9879864c2760b；partial已增长至13026164736 bytes，增加372244480 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：实际续传恢复已确认，全量包仍传输中。双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 12:41 北京时间 / 备份续传巡检
- 目标：持续确认历史全量备份和已校验产物状态。
- 改动：仅巡检与追加记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH严格既有主机校验连接正常，远端HEAD=dc3792d；接收器PID33192活跃，partial=13526335488 bytes，比上轮13026164736增加500170752 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；历史全量包继续传输，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 13:11 北京时间 / 备份连接重置恢复流程
- 目标：恢复历史备份连接，保留新增接收字节与失败记录。
- 改动：沿用现有严格主机校验分段脚本，不改代码或研究划分；确认旧PID33192退出、TRANSFER_EXIT_CODE=1后，将原PID/退出码保存outputs/phase65/backup/*disconnect_20261007_1311，启动隐藏接收器PID20828进行前缀校验再续传，保留用户改动。
- 网络情况：日志明确client_loop: send disconnect: Connection reset及segment incomplete；partial=14135853056 bytes，比上轮13526335488增加609517568 bytes后中断。常规SSH巡检成功，远端HEAD=cc9b16d；本轮新恢复流程尚未确认前缀校验完成或增长。
- 止损线：旧进程退出才恢复，前缀SHA不符禁止追加；完整归档及逐文件SHA未通过不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：全量备份未完成，恢复流程已启动，下一轮检查新接收器和增长。双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；无新训练、模型预测或实验指标。

## 65a / 2026-10-07 13:41 北京时间 / 恢复后的备份巡检
- 目标：确认上轮前缀校验及实际续传恢复，继续安全备份。
- 改动：仅巡检与追加记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH严格既有主机校验连接正常，远端HEAD=dbb2bec；接收器PID20828活跃。14135853056-byte前缀两层SHA均通过，SHA256=d7598c41ebd9bad31b8ef971cf3759b8907b6e5ec0f3600f36429454c7023bb4；partial已增长至14447280128 bytes，增加311427072 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：实际续传恢复已确认，全量包仍传输中。双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 14:11 北京时间 / 备份续传巡检
- 目标：持续确认历史全量备份及已校验产物状态。
- 改动：仅巡检与追加记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH严格既有主机校验连接正常，远端HEAD=49972df；接收器PID20828活跃，partial=14773387264 bytes，比上轮14447280128增加326107136 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；历史全量包继续传输，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 14:41 北京时间 / 备份续传巡检
- 目标：持续确认历史全量备份和冻结产物状态。
- 改动：仅巡检与追加记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH严格既有主机校验连接正常，远端HEAD=4fddc0d；接收器PID20828活跃，partial=15023996928 bytes，比上轮14773387264增加250609664 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整包及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；历史全量包继续传输，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 15:11 北京时间 / 备份续传巡检
- 目标：持续确认历史全量备份和已校验产物状态。
- 改动：仅巡检与追加记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH严格既有主机校验连接正常，远端HEAD=34402ac；接收器PID20828活跃，partial=15410921472 bytes，比上轮15023996928增加386924544 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；历史全量包继续传输，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 15:41 北京时间 / 备份续传巡检
- 目标：持续确认历史全量备份和冻结产物状态。
- 改动：仅巡检与追加记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH严格既有主机校验连接正常，远端HEAD=4ef48de；接收器PID20828活跃，partial=16115564544 bytes，比上轮15410921472增加704643072 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；历史全量包继续传输，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 16:11 北京时间 / 备份续传巡检
- 目标：持续确认历史全量备份和已校验产物状态。
- 改动：仅巡检与追加记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH严格既有主机校验连接正常，远端HEAD=94bbe55；接收器PID20828活跃，partial=16641949696 bytes，比上轮16115564544增加526385152 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；历史全量包继续传输，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 16:41 北京时间 / 备份续传巡检
- 目标：持续确认历史全量备份和冻结产物状态。
- 改动：仅巡检与追加记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH严格既有主机校验连接正常，远端HEAD=c647fae；接收器PID20828活跃，partial=17327718400 bytes，比上轮16641949696增加685768704 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；历史全量包继续传输，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 17:11 北京时间 / 备份续传巡检
- 目标：持续确认历史全量备份和已校验产物状态。
- 改动：仅巡检与追加记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH严格既有主机校验连接正常，远端HEAD=7e73cd7；接收器PID20828活跃，partial=18095276032 bytes，比上轮17327718400增加767557632 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；历史全量包继续传输，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 17:41 北京时间 / 备份续传巡检
- 目标：持续确认历史全量备份和冻结产物状态。
- 改动：仅巡检与追加记录，不重复启动接收器、下载或清单审计，保留用户改动。
- 网络情况：SSH严格既有主机校验连接正常，远端HEAD=d3b72aa；接收器PID20828活跃，partial=19050528768 bytes，比上轮18095276032增加955252736 bytes；日志无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档及逐文件SHA通过前不称备份完成；15/16支持门继续禁止训练及65b/65c。
- 结果：双机7文件仍all_files_verified，远端下载/审计退出码0，失败划分及审计报告SHA未变；历史全量包继续传输，无新模型预测或实验指标，无需恢复进程。

## 65a / 2026-10-07 18:11 北京时间 / 备份续传巡检
- 目标：确认历史全量备份持续推进及冻结数据状态。
- 改动：仅追加巡检记录；首次巡检误读不存在的files字段，修正读取status后成功核验，不改下载或训练代码。
- 网络情况：SSH严格既有主机校验连接正常，远端HEAD=b1efecf；接收器PID20828活跃，18:12观测partial=19674431488 bytes，比上轮19050528768增加623902720 bytes；transfer.log仍11425 bytes，无新增错误，当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED均不存在。
- 止损线：完整归档SHA及5230文件逐项验证前不称备份完成；固定支持门保持停止训练和65b/65c。
- 结果：双机仍all_files_verified，远端下载/审计退出码0，失败划分和审计SHA未变；未重复启动传输，未读取封存测试像素，无新模型预测或实验指标。

## 65a / 2026-10-07 18:41 北京时间 / SSH中断后安全续传
- 目标：诊断历史备份传输中断并保留全部已接收数据。
- 改动：确认PID20828已退出，保存本轮RECEIVER_PID和退出码历史记录；启动既有接收脚本PID23340，保留用户改动及20217298944 bytes临时包，未修改代码。
- 网络情况：18:37传输报Connection closed by remote host、segment incomplete，退出码1；两次远端巡检均在SSH握手阶段被关闭。新接收器活跃，正在既有远端打包状态重试流程，尚未完成本轮前缀SHA核验，也尚无恢复增长证据。
- 止损线：不绕过严格主机校验及前缀SHA；接收器活跃不重复启动；完整归档及逐文件校验前不称备份完成；固定支持门继续禁止训练和65b/65c。
- 结果：本地7官方文件仍all_files_verified、下载错误日志为空；远端本轮状态因SSH不可达未核验，不据此宣称变化或成功。临时包比上轮19674431488增加542867456 bytes后中断，无新模型预测或实验指标。
- 同步结果：本轮记录已git push成功；远端git pull尝试仍因SSH握手被关闭而失败，远端同步待后续连接恢复。接收器PID23340继续既有重试，未宣称续传恢复。

## 65a / 2026-10-07 19:11 北京时间 / SSH不可达持续巡检
- 目标：检查SSH恢复及既有备份重试，补齐远端待同步记录。
- 改动：仅巡检与追加记录；PID23340仍活跃，未重复启动接收器或下载，保留用户改动。
- 网络情况：远端ff-only pull及状态核验再次在SSH握手阶段被关闭；接收脚本持续既有重试，日志14425 bytes。partial仍20217298944 bytes，与上轮相同，尚未完成本次前缀SHA核验或恢复传输；当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED不存在。
- 止损线：不绕过SSH严格既有主机校验、前缀SHA或完整逐文件验证；固定支持门保持禁止训练和65b/65c。
- 结果：本地7官方文件仍all_files_verified，下载error.log为0 bytes；本轮远端数据和冻结SHA无法核验。没有新增实验指标；本轮记录本地验证后push，远端同步仍受同一SSH故障阻碍。

## 65a / 2026-10-07 19:41 北京时间 / 既有SSH重试巡检
- 目标：确认备份接收器状态并尝试补齐远端同步与核验。
- 改动：仅巡检和追加记录，PID23340活跃，不重复启动或改动传输代码，保留用户改动。
- 网络情况：SSH握手仍被远端关闭，远端pull和状态读取未执行成功；transfer.log增长至16693 bytes，内容为同一连接错误。partial仍20217298944 bytes，无恢复增长证据；当前TRANSFER_EXIT_CODE及BACKUP_VERIFIED不存在，既有接收器持续有限次数重试。
- 止损线：保持严格主机校验和续传前缀SHA；完整5230文件校验前不称备份完成；固定支持门继续禁止训练及65b/65c。
- 结果：本地7文件仍all_files_verified，下载console.log未变、error.log为空；本轮远端冻结SHA未能核验。没有新增模型预测或实验指标。记录本地验证后push，远端同步待SSH恢复。

## 65a / 2026-10-07 19:49 北京时间 / 用户取消自动work_dirs同步
- 目标：按用户新指令，将work_dirs同步改为用户手动操作。
- 改动：核验PID23340命令属于receive_phase65_backup.ps1后停止接收器及对应传输子进程；更新phase65定时任务，保持30分钟ACTIVE巡检，取消所有work_dirs自动打包、传回、备份、续传及备份巡检，包括审计报告，不再启动接收脚本。
- 网络情况：保留既有SSH配置及严格主机校验；此前握手故障仍未确认恢复，代码git push/pull流程继续保留。
- 止损线：保留20217298944 bytes临时包、已有本地备份及远端全部结果；未完成全量验证，不称备份完成。研究固定支持门继续禁止训练及65b/65c。
- 结果：接收器PID23340已不存在；定时任务更新成功且频率不变。后续work_dirs同步由用户手动操作，无新模型预测或实验指标。

## 65a / 2026-10-07 20:11 北京时间 / 下载与冻结状态巡检
- 目标：核验双机Sentinel完成状态及研究止损记录。
- 改动：仅巡检与追加记录，未重复下载或审计，保留用户改动；按用户新指令不巡检、不启动、不传回任何work_dirs备份。
- 网络情况：SSH严格既有主机校验连接正常，远端HEAD=41de1e7；本地下载console.log未变、error.log为空，远端下载退出码0。
- 止损线：固定支持门继续禁止65a训练及65b/65c，work_dirs同步继续由用户手动操作；不读取封存测试像素、预测或指标。
- 结果：两端7文件仍all_files_verified；失败划分两端SHA为a431745e7a17bbd4a7b9e12ef3bf76767d4eb827911bd5cb5e632683c2177f7b，远端清单审计SHA未变。无新研究授权、模型预测或实验指标。

## 65a / 2026-10-07 20:16 北京时间 / 用户授权ALCD新增数据入口
- 目标：双机下载核验Zenodo1460961，并判断是否提供额外独立云影场景。
- 改动：新增ALCD下载、gzip/TAR完整性校验、metadata-only清单审计和双机启动脚本；复用既有独占锁与Range续传。官方固定MD5=ee035e0d22a441086cfaabcface3cf24；本地6项既有下载测试和2项新增归档校验测试通过。
- 网络情况：官方页面可访问，记录说明38场景目录且两子集有重复，云影为class4；原始多光谱影像是否包含尚未核验。下载未完成，暂不报场景增量。
- 止损线：保留原失败split及支持门，不将标称场景数视为独立支持，不读旧sealed test像素/指标。ALCD参考标签由主动学习随机森林生成，置信图不是人工真值可靠率；无原始影像或独立性未知的产品不进入确认。work_dirs仍由用户手动同步。
- 结果：用户本轮授权新增数据下载与评估；尚未授权任何未经冻结的新split或训练。先push/pull代码，再启动双机下载和无像素清单审计。

## 65a / 2026-10-07 20:23 北京时间 / ALCD双机完成及适用性初审
- 目标：完成新数据双机校验，判断是否能直接补足确认场景。
- 改动：新增产品/tile与归档内容审计脚本并本地执行；计划记录新增数据入口与用户手动同步覆盖旧备份条款；30分钟定时任务已加入ALCD监控。
- 网络情况：双机下载完成，官方长度234569318 bytes、MD5、gzip CRC、TAR完整性均通过；两端SHA256相同5912fbcfe9edbc1c2cdffbb7ef0119ffdb3099bc9eec37efe301929008bf966d。本地下载/审计退出码0，远端下载/审计退出码0；一次额外远端摘要读取未激活conda导致python不可用，非下载失败。
- 止损线：保留旧失败split，不直接拼入确认或宣称支持门已过；无原始影像、足迹独立性及有效shadow支持核验前不训练，不查看旧sealed像素或指标。
- 结果：本地metadata-only审计38目录、37唯一cloudy产品、15tiles；1256文件中76 TIFF全为classification_map/confidence_enhanced，无原始多光谱输入。11目录覆盖4个source同tile，1目录与4172871同tile；精确产品字符串重合0不能替代标准化采集ID/足迹审计。没有新增可用独立H1场景数结论、模型预测或性能指标。work_dirs未同步。

## 65a / 2026-10-07 20:34 北京时间 / 用户请求work_dirs垃圾审计
- 目标：只读检查结果目录容量和清理候选，不执行用户尚未要求的打包移动清空。
- 改动：新增文件元信息/同大小checkpoint SHA重复审计脚本，本地验证push后远端pull执行；报告在outputs/phase65/work_dir_audit及远端shared/work_dir_audits，不同步work_dirs。
- 网络情况：SSH连接及git push/pull成功；本地393471936 bytes/1399文件，远端36240857331 bytes/5259文件。远端src/result_backups另占约34GiB，src总约68GiB，未找到90多GB单work_dirs。
- 止损线：不删除、移动、打包；不读取封存测试像素预测指标。当前Phase65源权重和冻结审计不可随清空失去引用，手动归档须先核验后清空。
- 结果：明显临时候选仅68740 bytes pyc；同SHA重复checkpoint0、外链0。270权重23.84GB，cache路径11.21GB，95零文件多为状态标记，均不据名称认定垃圾。迭代权重17.05GB可后续按最佳/末次/引用需求精简，但此次未授权删除。详细结论见outputs/phase65/work_dir_audit/review_20261007.md，无新增训练或实验指标。

## 65a / 2026-10-07 20:41 北京时间 / 双数据入口冻结巡检
- 目标：确认Catalogue与ALCD两端校验完成状态及冻结研究停止记录。
- 改动：仅核验和追加记录，保留用户改动；不重下载、不覆盖审计、不打包传输或清理work_dirs及result_backups。
- 网络情况：SSH连接正常，远端HEAD=680a193；Catalogue与ALCD远端下载退出码0，本地两下载error.log和远端ALCD error.log均为空。
- 止损线：保留旧支持门失败split，不启动训练或65b/65c；ALCD仍缺原始多光谱影像，未确认独立H1增量，不以37唯一产品替代有效场景支持。
- 结果：双机Catalogue 7文件、ALCD 1文件均all_files_verified；ALCD SHA256仍5912fbcfe9edbc1c2cdffbb7ef0119ffdb3099bc9eec37efe301929008bf966d，gzip/TAR通过；split SHA两端未变。无新模型预测、性能指标或需要重复通知的状态变化。

## 65a / 2026-10-07 21:13 北京时间 / 双数据状态与研究门巡检
- 目标：核验双机数据完成记录与固定停止门，保持用户手动同步安排。
- 改动：仅只读巡检和追加日志，保留用户改动，不重下载、不覆盖审计、不巡检或恢复备份，不清理目录。
- 网络情况：SSH正常，远端HEAD=f82639b；本地Catalogue/ALCD日志未变且错误日志为空，ALCD退出码0；远端两下载退出码0、ALCD错误日志为空。
- 止损线：旧split维持stopped_insufficient_h1_support及training_authorized=false；不重抽、不降低门槛、不借65b/65c，不读取旧封存测试像素预测指标。
- 结果：双机Catalogue 7文件和ALCD 1文件均all_files_verified，ALCD固定SHA256及gzip/TAR通过状态未变；split两端SHA256仍a431745e7a17bbd4a7b9e12ef3bf76767d4eb827911bd5cb5e632683c2177f7b。ALCD原始影像缺项无新进展，没有新增独立H1支持数、模型预测或实验指标，不重复通知。

## 65a / 2026-10-07 21:20 北京时间 / 用户确认第三方备份后的远端清理
- 目标：依据用户明确授权尽量腾出远端空间，保留当前研究最重要的文件。
- 改动：本地编写清单式删除脚本与3项保护规则测试，两端测试通过、push/pull后计划并执行；保留整个phase65a、Phase65源权重、JSON/CSV/日志/配置和小体积证据，删除已归档历史大权重、NPZ/NPY缓存、导出模型和ZIP。本地文件不动。
- 网络情况：SSH和git同步正常；执行前未见训练进程。src/result_backups执行前已不存在，不是本轮删除。用户确认第三方备份，代理未访问或独立核验第三方。
- 止损线：删除前严格限定授权目录、拒绝外链、核验文件列表/大小/mtime和关键source SHA；保留冻结支持门及training_authorized=false。自动备份仍禁用，定时任务已登记预期历史大文件缺失，不自动恢复或再次删除。
- 结果：成功删除2365文件、35745742890 bytes（35.75GB），结果目录剩495114441 bytes（du483MiB）；文件系统可用748038963200 bytes。source SHA仍64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9，split SHA仍a431745e7a17bbd4a7b9e12ef3bf76767d4eb827911bd5cb5e632683c2177f7b。完整清单及逐文件删除日志在/home/scv/shared/phase65_cleanup/cleanup_20261007_2115.json、.deleted.jsonl、.completed.json。此次清理完成，无训练或新增实验指标。

## 65a / 2026-10-07 22:25 北京时间 / 清理后保留项与数据状态巡检
- 目标：核验关键source、冻结划分和双机数据状态，不恢复已清理历史文件。
- 改动：仅只读巡检与追加日志，保留用户改动；不再清理，不打包续传，不重下载或覆盖审计。
- 网络情况：SSH正常，远端HEAD=58c94e3；双机Catalogue 7文件及ALCD 1文件均all_files_verified，本地两下载error.log及远端ALCD error.log为空，远端两下载退出码0。
- 止损线：保留固定支持门stopped_insufficient_h1_support、training_authorized=false；ALCD影像缺项未解决，不重抽、不降低门槛、不借65b/65c，不读取旧封存测试像素预测指标。
- 结果：清理后重要source SHA仍64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9，split两端SHA仍a431745e7a17bbd4a7b9e12ef3bf76767d4eb827911bd5cb5e632683c2177f7b。无新独立H1场景数、模型预测或性能指标，无状态变化需通知。

## 65a / 2026-10-07 23:25 北京时间 / 双数据与保留关键项巡检
- 目标：确认双机数据完成状态、重要source和固定支持门。
- 改动：仅只读巡检并追加记录，保留用户改动，不下载或覆盖审计，不清理或恢复历史权重缓存，不打包或同步work_dirs。
- 网络情况：SSH正常，远端HEAD=8ba202a；本地两下载console.log未变、error.log为空，ALCD退出码0；远端两下载退出码0、ALCD error.log为空。
- 止损线：固定门仍stopped_insufficient_h1_support、training_authorized=false；ALCD仍缺原始影像，独立H1支持未成立，不重抽、不降门槛、不借65b/65c，不读取旧封存测试像素预测指标。
- 结果：双机Catalogue 7文件、ALCD 1文件仍all_files_verified；保留source SHA仍64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9，split两端SHA仍a431745e7a17bbd4a7b9e12ef3bf76767d4eb827911bd5cb5e632683c2177f7b。无新增模型预测、性能指标或需通知变化。

## 65a / 2026-10-08 00:25 北京时间 / 双数据与保留关键项巡检
- 目标：核验双机数据完成状态、关键source与固定停止门。
- 改动：仅只读巡检与追加日志，保留用户改动，不重下载、不覆盖审计，不清理或恢复历史文件，不打包或同步work_dirs。
- 网络情况：SSH正常，远端HEAD=f7c4347；本地两下载日志大小未变、error.log为空，ALCD退出码0；远端两下载退出码0，ALCD error.log为空。
- 止损线：固定支持门仍stopped_insufficient_h1_support、training_authorized=false；ALCD原始影像仍未取得，不重抽、不降门槛、不借65b/65c，不读取旧封存测试像素预测指标。
- 结果：双机Catalogue 7文件、ALCD 1文件仍all_files_verified；保留source SHA仍64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9，split两端SHA仍a431745e7a17bbd4a7b9e12ef3bf76767d4eb827911bd5cb5e632683c2177f7b。无新增独立H1支持、模型预测或性能指标，无需重复通知。

## 65a / 2026-10-08 01:26 北京时间 / 双数据与冻结门巡检
- 目标：核验双机数据完成记录、保留source与冻结划分。
- 改动：仅只读检查和追加日志，保留用户改动；不重下载、不覆盖审计、不恢复或清理历史文件，不打包同步work_dirs。
- 网络情况：SSH正常，远端HEAD=fb81870；本地两下载日志大小未变且error.log为空，ALCD退出码0；远端两下载退出码0、ALCD error.log为空。
- 止损线：支持门仍stopped_insufficient_h1_support、training_authorized=false；ALCD影像缺项未解决，不重抽、不降低门槛、不借65b/65c，不读取旧封存测试像素预测指标。
- 结果：双机Catalogue 7文件、ALCD 1文件仍all_files_verified；source SHA仍64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9，split两端SHA仍a431745e7a17bbd4a7b9e12ef3bf76767d4eb827911bd5cb5e632683c2177f7b。无新增独立H1场景、模型预测或性能指标，无需重复通知。

## 65a / 2026-10-08 11:19 北京时间 / 本地完成状态与SSH失败巡检
- 目标：核验双机完成状态及关键保留项，保持研究停止和手动同步安排。
- 改动：仅只读检查和追加记录，保留用户改动，不重下载、不恢复历史文件、不清理或打包同步work_dirs。
- 网络情况：两次远端巡检在SSH握手阶段报kex_exchange_identification: Connection closed by remote host，本轮远端命令未成功执行，未核验远端HEAD、source或split，不推断远端文件变化。本地两下载日志未变，error.log为空，ALCD退出码0。
- 止损线：保持既有严格主机校验，不修改系统/凭据；原固定研究门与训练禁令继续生效，不重抽、不降低门槛、不借65b/65c，不读取封存测试像素预测指标。
- 结果：本地Catalogue 7文件、ALCD 1文件仍all_files_verified，split SHA仍a431745e7a17bbd4a7b9e12ef3bf76767d4eb827911bd5cb5e632683c2177f7b。ALCD仍缺原始影像，无新独立H1支持或实验指标。记录push后尝试远端pull，若连接未恢复则留待后续同步，不反复通知相同SSH故障。
- 同步结果：本轮git push及远端git pull --ff-only最终成功（6de9e2b），连接在同步尝试时恢复；此前两次数据巡检失败仍如实保留，未将代码同步成功替代数据与source校验。


## 65a / 2026-10-08 / ALCD原始影像检索与文献审查
- 目标：按新增授权读取used_parameters，检索CDSE并获取六波段，审查标签质量证据。
- 改动：新增原始影像检索/校验下载、格网与有效像元审计脚本及双机启动器；独立文献记录PHASE65_ALCD_LABEL_REVIEW.md；未变更原split。
- 网络情况：CDSE匿名目录正常，产品下载端点实测401；用户明确选择无需登录公开镜像。Google公开L1C镜像37/37原始产品严格匹配，222个请求波段合计17232408688 bytes；小型产品XML长度/镜像MD5/SHA256验证成功。
- 止损线：仅新增ALCD入库，不读旧封存测试；歧义产品/校验失败停止、不绕过锁、不换L2A；原停止门及训练禁令有效，work_dirs不自动同步。
- 结果：发现37景可获取原始影像，不代表全量下载完成或独立H1数量够；编译与旧产品错误采集防护检查通过，指定远端conda已有rasterio/numpy。文献有23/29 CESBIO景被修订云/影漏误标的直接证据，此数字是作者子集的修订景数而非像素错误率。

## 65a / 2026-10-08 / ALCD双机影像下载与标签版本比对
- 目标：落实公开镜像六波段下载，核查参考标签中的已知缺陷和版本差异。
- 改动：双机下载器启动，远端下载后自动做几何审计；新增compare_phase65_alcd_reference.py，文献审查补入真实标签清点与修订版比对。有效掩膜排除XML中NODATA与SATURATED特殊值；读取首景XML确认量化10000、基线02.01、特殊值0/65535。定时巡检已加入新影像任务并保留现有运行频率。
- 网络情况：git push→远端pull --ff-only成功；本地启动器PID32348，远端启动器PID23110；Google镜像双机下载实际进行，本地已校验172对象8480293689 bytes（快照），远端首波段已verified且partial增长，error日志未见报错，不重复启动或覆盖现有文件。作者9MB修订标签固定公开MD5、SHA和gzip CRC通过。
- 止损线：不自动同步/备份work_dirs，不动旧split与source；无新训练/预测，旧sealed test持续封存。新数据质量与独立性未冻结前不能称训练就绪；版本差异不冒称真实错误或模型错误流向。
- 结果：原版37景全部1830×1830/60m，重复目录标签一致，无异常类号，17655048格nodata、2景无类4。作者公开修订包29景全部格网匹配，27景在有效稳定60m内部格有标签差异；具体计数在独立审查文档及shared报告。影像全量下载和六波段联合有效mask审计尚未完成；远端label_intake与reference_disagreement报告来自shared，未传work_dirs。
- 后续实测：geometry_progress_20261008.json对已齐备的前3景六波段完成JPEG2000解码、CRS/边界/原生格网核对，3景all_six_grids_match、34景pending_verified_bands、0景grid_mismatch；联合有效掩膜同时排除XML NODATA=0及SATURATED=65535。本地后续快照240对象11528805265 bytes已校验，远端较早快照27对象1471015235 bytes已校验且partial仍增长。两机继续下载，未宣布全量完成或标签语义认证。


## 65a / 2026-10-08 12:50 北京时间 / 原始影像巡检
- 目标：核验双机L1C六波段下载和原停止门，继续已授权数据审查。
- 改动：仅只读巡检与追加本轮记录，保留用户改动；不重启下载，不同步work_dirs，不训练。
- 网络情况：SSH正常；本地397对象17255829220 bytes均verified、退出码0、无partial/锁、error.log为空；Catalogue和ALCD官方包本地状态仍all_files_verified。远端89对象4406603375 bytes已verified，下载继续，error.log为空、geometry_audit.json尚未生成。
- 止损线：活进程和partial增长时不重复下载、不绕过锁；原失败split不重抽、不降低门槛、不借65b/65c，封存测试不读像素/预测/指标。
- 结果：本地原始影像已校验完成，远端仍进行中，双机完整校验与最终37景几何审计未完成；不把下载完成称实验完成，不新增模型指标。仅待远端下载与后续自动审计，暂无用户操作要求。


## 65a / 2026-10-08 13:50 北京时间 / 原始影像巡检
- 目标：核验双机原始影像下载进展，保留既有研究停止门。
- 改动：仅追加本轮记录，保留用户日志整理及 outputs 修改；不修改代码，不同步或清理 work_dirs。
- 模型/网络：未训练、未预测。SSH 状态读取成功；本地 Catalogue 与 ALCD 官方包均 all_files_verified，原始影像397对象17255829220 bytes全部 verified，EXIT_CODE=0、error.log为空。远端官方包均 all_files_verified，原始影像144对象6766429815 bytes verified，较上轮89对象继续增长，B04 partial为6291456 bytes，error.log为空，geometry_audit.json尚未生成。
- 止损线：不重复启动正常下载，不绕过 IMAGERY_LOCK 或校验；保持原失败 split，不重抽、不降低门槛、不推进65b/65c，不读取旧 sealed test 像素、预测或指标。
- 真实结果：远端下载仍在推进，双机完整影像与最终几何审计尚未完成；没有新的标签语义可靠性或独立支持数量结论，无需用户操作。


## 65a / 2026-10-08 14:50 北京时间 / 原始影像巡检
- 目标：核验双机下载与最终几何审计进展，继续已授权数据审查。
- 改动：仅追加巡检记录；保留用户日志整理和 outputs 修改，不同步或清理 work_dirs。
- 模型/SSH/网络：没有训练或预测；SSH 状态读取成功。双机 Catalogue、ALCD 官方包均 all_files_verified；本地原始影像397对象17255829220 bytes全部 verified，EXIT_CODE=0、error.log为空。远端172对象8480293689 bytes verified，较上轮144对象继续增长；B08 partial为32505856 bytes，error.log为空，最终 geometry_audit.json尚未生成。
- 止损线：保持原失败 split 与训练禁令，不重抽、不降门槛、不推进65b/65c；下载推进时不重复启动，不绕过锁或校验错误；旧 sealed test 只允许 manifest IDs。
- 真实结果：远端原始影像下载仍在推进，等待完整校验和自动几何审计；未形成新的标签语义可靠性或独立场景支持结论，无需用户操作。
