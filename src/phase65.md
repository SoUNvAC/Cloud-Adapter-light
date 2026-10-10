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


## 65a / 2026-10-08 15:50 北京时间 / 原始影像巡检
- 目标：核验原始影像双机下载进展与最终几何审计状态。
- 改动：仅追加本轮日志，保留用户现有修改，不同步、备份或再清理 work_dirs。
- 模型/SSH/下载网络：未训练或预测；SSH 状态读取成功。双机 Catalogue、ALCD 官方包均 all_files_verified；本地原始影像397对象17255829220 bytes全部 verified，EXIT_CODE=0、error.log为空。远端215对象10192246730 bytes verified，较上轮172对象继续推进；B04 partial为57671680 bytes，error.log为空，geometry_audit.json尚未生成。
- 止损线：原失败 split 保持冻结，不重抽、不降低支持门、不推进65b/65c；活进程下载推进时不重复启动，不绕过锁或校验错误；旧 sealed test 像素、预测和指标禁读。
- 真实结果：仍等待远端下载完整校验与自动几何审计；没有新的标签语义质量或独立场景支持结论，无需用户操作。


## 65a / 2026-10-08 16:50 北京时间 / 原始影像巡检
- 目标：核验双机原始影像下载、最终几何审计及冻结 split。
- 改动：仅追加实际巡检记录，保留用户修改，不同步或清理 work_dirs。
- 模型/SSH/下载网络：没有训练或预测；首次SSH握手被远端关闭，一次只读重试成功。本地Catalogue与ALCD官方包均all_files_verified；原始影像397对象17255829220 bytes全部verified，EXIT_CODE=0，error.log为空。远端原始影像279对象12927185966 bytes verified，较上轮215对象继续推进，主进程23117存活，B02 partial为58720256 bytes，error.log为空，geometry_audit.json尚未生成。
- 止损线：原失败split SHA256仍为a431745e7a17bbd4a7b9e12ef3bf76767d4eb827911bd5cb5e632683c2177f7b；不重抽、不降门槛、不训练或推进65b/65c，不读旧sealed test像素/预测/指标；正常下载不重启，不绕过锁与校验。
- 真实结果：瞬时SSH握手问题已恢复，下载继续推进，等待远端完整校验与几何审计。没有新标签语义可靠性或独立支持结论，无需用户操作。


## 65a / 2026-10-08 17:50 北京时间 / 双机影像校验及最终几何审计完成
- 目标：核验远端下载完成与完整几何审计，冻结独立于模型的标签复核方案。
- 改动：从shared数据目录取得完整下载状态及几何报告，仅用于数据审查；写入PHASE65_ALCD_VISUAL_REVIEW.md v1和本轮日志，不同步work_dirs，不清理文件。
- 模型/SSH/网络：没有训练或预测；SSH正常，双机Catalogue与ALCD官方包均all_files_verified。本地原始影像退出0；远端397对象17255829220 bytes全部verified，无partial，error.log为空。逐对象size/md5_base64/SHA256/generation/status双机一致。
- 真实结果：37/37景全部六波段原生格网匹配60m标签，联合有效106252111格，几何报告SHA256=a8729d197458b1a29c4024bda4cf8b071ee1094cd19ba7307b076ce7b861dcca。完整影像校验与几何审计完成；语义质量和新增独立H1支持仍未成立。复核方案冻结版本差异诊断与不依赖标签/模型的逐景空间抽样，视觉判读待执行。
- 止损线：原split SHA仍a431745e7a17bbd4a7b9e12ef3bf76767d4eb827911bd5cb5e632683c2177f7b；不重抽、降门槛、训练或推进65b/65c，旧sealed test禁读像素/预测/指标。下载与几何审计完成不称实验完成。


## 65a / 2026-10-08 / 用户授权抽样材料本地交付
- 目标：按冻结方案生成ALCD复核图、原生窗口和可填写页面，明确人工交付文件。
- 改动：prepare_phase65_alcd_visual_review.py已本地语法验证、push后远端pull运行；只读新增ALCD数据，在shared数据目录生成复核材料并交付本地outputs/phase65/alcd_visual_review_v1，不涉及work_dirs。界面锁定揭示前判断，备注允许补充；未修改原标签或旧split。
- 模型/SSH/网络：没有训练、模型预测或sealed test像素访问；首次材料scp握手被关闭，重试成功，完整107929600-byte数据复核包已接收，本地逐文件SHA核验通过。
- 真实结果：37景常规851处，版本差异728处，共1579唯一位置；74无有效观测区块记录跳过，差异点无无效中心。保留六波段原生窗口、有效mask、坐标、输入哈希、显示拉伸和两个参考标签面板。浏览器页面可分批填写导出review_results.csv，用户只需交回此填写文件。
- 验证：程序逐景复算联合有效mask与最终几何审计计数一致；抽样不使用模型。图像预览检查红框与两版面板；页面JS语法、盲判断锁定、中文CSV导出及全部编号通过功能检查。
- 止损线：这只是复核材料交付，尚未完成人工语义判读，不能称标签可靠或独立支持足够；原失败split、训练与65b/65c禁令保留。


## 65a / 2026-10-08 / 新授权探索试跑准备
- 用户暂停人工复核并明确授权先做Phase65a；新增独立探索协议，只用原153 fit组155产品及32 development代表，不改变原失败split，不读取确认集或旧sealed test，不推进65b/65c。
- 新增官方数组准备/原生有效mask、Catalogue加载器、seed65 MsRE配置、source与适配成对评估和描述性汇总；本地数据转换两项测试及Python语法检查通过。代码先push再远端pull后执行。
- 远端CUDA/指定conda可用，source与split固定SHA匹配；新授权覆盖此前对本次探索训练的禁令，原主检验停止门继续保留。work_dirs自动同步仍取消；尚未产生训练结果。

- 启动核验：远端703d98f、PID510296；187产品准备及185代表source评估完成，MsRE seed65已到400/4000步，loss=5.1531为有限值，lr=9.3071e-05；原split SHA不变。只读远端日志，无work_dirs传回。
- 运行限制：沿用deterministic_warn_only，CUDA cumsum/grid_sample及CuBLAS报告非严格确定性；不声称逐位可复现。定时任务已更新为跟进本次明确授权的探索作业，保留原停止门、复核暂停和同步取消。


## 65a / 2026-10-08 / 探索训练完成，汇总依赖修复
- 真实状态：seed65完成4000/4000步，最佳development checkpoint为2000步；source和适配均完成153 fit代表与32 development代表评估。主PID510296已退出，管线EXIT_CODE=1，唯一最终错误为汇总缺少sklearn；不重训、不重跑已完成评估。
- 修复：改用远端已有SciPy L-BFGS-B实现同一固定C=1、fit标准化、无惩罚截距逻辑回归目标，保留特征、支持门和划分；本地运行时缺少SciPy，未执行sklearn动态对照；本地语法、固定目标解析梯度有限差分、常量标准化及AUROC平分规则通过，求解器需远端验证。只重跑尚未生成的汇总，保留原失败退出记录，不安装系统依赖。

- 最终核验：汇总修复EXIT_CODE=0，原管线EXIT_CODE=1保留为缺依赖历史；4000步和185代表成对评估均完成，原split SHA不变。结果保留远端phase65a_explore_20261008；未自动传回任何work_dirs文件。
- 真实描述结果：development像元汇总mIoU source67.13334%→MsRE77.70362%，Shadow IoU21.23065%→48.80934%；development场景均值mIoU变化+25.18420pp，分母32，与像元汇总定义不同。best checkpoint=2000。development Shadow→Surface像元166595→224074，准确率增益不表示所有错误流改善。
- H1仅fit40、development7个有支持且有有效shadow代表，描述相关fit0.021025/development-0.133429；H2 fit19正134负、development4正28负，固定逻辑回归收敛，但development类别支持不足，不声称AUROC或正式假设通过。探索结果不替代原失败支持门，不授权65b/65c。


## 65b / 2026-10-08 / 新授权固定光谱风险证据对照
- 用户明确要求推进65b查看结果；新增独立探索协议，使用原65b_confirmation的64固定代表作本轮信息诊断；原65a失败split不改、不重抽、不读取65a_confirmation/65c/旧sealed test。ALCD不参与。
- 冻结source及65a最佳2000步MsRE SHA、153 fit代表风险拟合输入SHA，比较前9维RGB可观测特征与12维额外B08/B11/B12均值，并增加删除和打乱光谱对照。只拟合浅层风险探针，不重训分割模型。
- 本地配对分母、组唯一性及零增量bootstrap三项测试和语法检查通过；远端无现存训练任务，65a汇总修复退出0。代码先push后远端pull执行，未产生65b结果。
- 形式限制：官方数组内部格网匹配不能替代外部CRS/footprint、独立标签复核、小样本RGB/六波段分割95%及六波段source保留门；本次只探索证据，不称正式65b主门通过、不授权65c、不同步work_dirs。

- 完成核验：远端PID511996已完成，EXIT_CODE=0，64/64代表准备及source/MsRE成对推理、9维/12维固定风险探针拟合和打乱证据分数均完成；原split SHA不变。产物仅保留远端src/work_dirs/phase65b_explore_20261008/evidence_report.json，未自动传回work_dirs。
- 真实结果：64代表66837646有效像元，source→MsRE mIoU58.13830%→63.33544%、Shadow IoU13.95644%→24.48705%；真实Shadow分母1003961，正确Shadow591149→280731（召回58.8816%→27.9624%），Shadow→Surface375998→582688（37.4515%→58.0389%），Shadow→Cloud36814→140542。整体IoU增加不表示漏影受控。
- 风险支持：fit19正134负；65b5正59负，恶化场景小于固定8门。按协议不报告三探针AUROC/CI或光谱增量显著性，不补抽、不改风险阈值；本次为完成的65b探索性证据首轮，不是正式65b全门通过。太阳几何/独立大上下文未执行，原输入审核/小样本95%/匹配六波段source保留门仍待完成；不启动65c。


## 65D / 2026-10-08 / 用户授权分数排序及冻结阈值诊断
- 目标：全像元提取Source/MsRE云影分数，PR/相同FPR召回，固定T18FYG暗地表排序诊断，development-only选择全局校准决策基线后冻结。ALCD不参与，不训练新机制，不改原split，不读取原65a确认/65c/旧sealed test。
- 方法：先32 development双模型提分数、绑定固定checkpoint SHA及先前argmax confusion一致性；仅用development Source误报预算选择softmax/margin/原MsRE候选，独占写calibration_lock后才提取64已查看65b代表分数。精确ties AP/排序和固定FPR0.1/0.5/1/2/5/10%，按场景而非像元bootstrap；明示heldout/单景排序阈值为oracle诊断，不调确认阈值。
- 核验：Mask2Former实现输出查询类别softmax与mask sigmoid的聚合，不是校准概率；保存聚合Shadow、此前softmaxShadow和决策margin三类分数。五项本地tie/AP/单调变换/保守FP预算/无效标签测试及Python语法通过。网络SSH正常，shared可用655GB，未开始新推理。
- 交付：新分数、锁和诊断在shared数据目录，既有work_dirs只读，不自动同步。按本次画图请求交付新shared诊断/图到本地；科研绘图技能用于PR、工作点曲线和CDF，先数据剖析再图像自检。本次是事后探索诊断，不能称新独立确认。

## 65D / 2026-10-08 21:32 / 完成固定诊断，校准迁移失败
- 真实执行：初次提取第二景Source confusion不一致，EXIT_CODE=1，已确认主PID退出；对齐前轮model.train→eval冻结生命周期及推理环境，在独立shared/data/sentinel2_cloud_mask_catalogue_4172871/phase65d_20261008_recovery01重跑。旧失败现场、部分分数及原记录不覆盖；不声称已独立定位两项设置中哪一项造成差异。恢复run EXIT_CODE=0，32 development+64已查看65b，两模型192份原argmax confusion全部与前轮一致，100261134有效参考像元全量分数/精确ties统计。Source/MsRE固定SHA匹配，原split SHA仍a431745e7a17bbd4a7b9e12ef3bf76767d4eb827911bd5cb5e632683c2177f7b。
- PR/排序：development池化AP Source0.205947→MsRE0.688675、1%FPR召回1.4886%→73.8106%；65b池化AP0.132402→0.388243、1%FPR召回6.0570%→41.6930%。65b5%FPR反转，Source68.0415%>MsRE55.6313%。只有7/12景含有效Shadow，逐景宏AP分别0.693150/0.673652和0.528044/0.545511；1%FPR逐景召回差bootstrap区间均跨0，不能由池化优势声称所有独立场景改善。
- development冻结：全局softmaxShadow阈值0.22136211395263672，先锁再读取本轮65b分数，锁SHA0530a622970f7d519bb4d192d3e6e94b52c74a687a683e08a7e2a9f7866292ba未变。dev场景宏召回51.6626%→80.4467%，池化召回61.1721%→86.8972%，FPR0.5070%→4.8553%，mIoU77.7036%→70.0040%。该阈值原样65b召回27.9623%→57.6728%，FPR0.2164%→5.9268%，mIoU63.3354%→56.4888%，未保持Source误报4.9089%或mIoU58.1383%；仅保留基线，不宣布修复成功、不按65b再校准。
- T18FYG：Shadow主分数中位数0.455236→0.212577，Surface0.133580→0.212249，原Shadow聚合中位数1.197557→0.000505。主分数全非Shadow AP0.835331→0.620536、AUC0.947345→0.830944、5%FPR召回45.2679%→16.1445%；冻结阈值仅0→7.3152%召回。支持分数塌缩且排序下降，不只是全局平移，单调校准不能恢复AP。绝对RGB<0.08的Surface只有3像元，不报告可靠暗地物结论；补充协议预写后事后检查全部Surface及其最低亮度10%（边界0.1236、37511像元含ties），主分数AUC分别0.959186→0.772554、0.854934→0.699289。相对较暗不等于人工验证暗地物；原始聚合Shadow对Surface AUC仍0.917532但对全非Shadow0.552209，Cloud也影响排序，不能全归因于暗地表。
- 核验/交付：5项本地指标测试及新增Python语法通过；576行逐景分数CSV数据剖析、PR/FPR/CDF预览/灰度和可见文字边界检查通过。另导出300dpi PNG、PDF及SVG，放大T18主分数重叠范围仅改变显示，不改指标/锁。新shared报告/图交付outputs/phase65/phase65d_20261008及README.md；原work_dirs未自动传回，全像元分数留远端。Windows scp多源文件校验报错后改逐文件/单目录接收，未绕过SSH校验。网络SSH/git正常，无额外模型/数据下载。
- 止损：ALCD不入本轮、人工复核仍暂停；原15/16支持门和training_authorized=false保留，未读原65a确认/65c/旧sealed像素，不训练新机制，不推进65c。固定校准基线不足以稳定迁移；后续新机制需另有训练授权和协议，预算截止仍2026-10-20北京时间。
- 本地最终交付核验：5份新诊断/锁/曲线/逐景CSV/补充报告及9份最终PNG/PDF/SVG，共14份SHA与远端shared完全一致，delivery_verification.json保存证据。三组PNG为300dpi，strict图件检查无FAIL；工具对PDF Type0父字体提示可能未嵌入，进一步逐页核查DescendantFonts FontDescriptor FontFile2均确有TrueType嵌入，无Type3。README给出解释及全部像元/逐景支持区别。本地报告与图完整交付，未同步任何work_dirs产物。

## 65D后续 / 2026-10-08 / checkpoint轨迹、损伤结构与额外信息筛查
- 目标与边界：用户授权三项后续诊断，预写PHASE65D_FOLLOWUP_20261008.md；复用Source/2000步分数，仅读现存3500/4000步checkpoint，32 development相关组代表先统计冻结，再对T18FYG事后追踪。原split、Source、2000步checkpoint及65D校准锁不变，不按T18选模型，不训练新分割网络、不进原65a确认/65c/旧sealed、不使用ALCD，不自动同步work_dirs。
- 真实执行：初次serialized dataset索引失败、恢复运行分析阶段嵌套日志路径失败，均保留EXIT_CODE=1；确认退出后修复索引和唯一日志定位，复用已经完成的64次development推理，只恢复分析及2次T18推理。shared/phase65d_followup_20261008_recovery01/run_logs/analysis_resume01/EXIT_CODE=0；共66次新前向，无优化器更新。现存checkpoint只有2000/3500/4000，500/1000/1500/2500/3000仅有日志mIoU，不补造AP或声称定位损伤首次出现阶段。
- development数据：32组33423488有效参考像元，AP支持仅7组。Source/2000/3500/4000的池化mIoU为67.13334/77.70362/77.35788/77.30785%，池化AP为0.205947/0.688675/0.734689/0.729858，1%FPR召回1.4886/73.8112/76.7686/76.4971%，5%FPR召回73.1111/87.0123/95.5430/94.8540%；原argmax预测Shadow比例6.1873/1.6976/2.0907/2.0091%。7组宏AP为0.693150/0.673652/0.689758/0.686725。3500的池化排序指标更好但mIoU略低，不替换原选定2000模型；FPR扫描为参考标签条件下oracle排序诊断，不是部署阈值。
- 结构：固定1 AP百分点边界，2000步3改善/3下降/1稳定，晚期4改善/2下降/1稳定。T52UDG三点AP差-6.86/-9.57/-9.71pp，T40XDR-16.51/-5.02/-7.29pp持续下降，T35RPM-1.17/-0.38/-0.37pp晚期基本恢复。三个checkpoint同seed同路径，是组内重复测量，不证明跨seed重复性、不以像元数增加独立样本量。
- 可识别性：固定ridge惩罚10、每折标准化、7组留一，折内均值/RGB+Source九特征/增加NIR-SWIR三均值的MAE为6.49/6.04/5.51 APpp；T40XDR实际-16.51pp但两探针仅预测-0.23/-0.63pp。特征不含GT比例；样本仅7组且2000模型此前已由整个development选定，不能称端到端独立验证、可靠部署风险识别或增量显著。
- 光谱筛查：固定负RGB/B08/B11/B12/三光谱均值，不拟合像元分类器、不调权重。Surface最低RGB亮度10%子集中，T52UDG Source/MsRE/B11 AUC=0.913/0.791/0.961，T40XDR=0.943/0.908/0.842，T32TML=0.963/0.985/0.619。支持部分场景有额外光谱区分信号，但没有融合恢复证据，不能全局替换原分数；相对低亮度不是人工确认暗地物，负类不含Cloud，几何/更大上下文无核验输入故未测试。
- T18事后：Source/2000/3500/4000 AP=0.835331/0.620536/0.654324/0.648710，5%FPR召回45.2679/16.1445/19.6040/19.1450%，原argmax召回62.6263/0/1.1691/1.0784%。晚期部分恢复但仍明显落后Source；未用此结果调整模型或原阈值。
- 核验与交付：三项留一统计测试及代码语法通过；128行32组四模型逐景CSV完成数据剖析。所有新诊断/图仅从shared交付outputs/phase65/phase65d_followup_20261008，原分数留远端，新README按数据结果/结论探讨分开。SSH/git出现瞬时握手关闭时只重试，未改凭据；无新数据下载。保留原停止门、预算2026-10-20；先完善development决策/选择目标及Source能力保留基线，再考虑固定匹配的低容量光谱增量探针，本轮不设计或训练新机制。
- 最终本地交付：四份报告/逐景CSV及首选九份PNG/PDF/SVG的SHA256与远端shared全部一致，delivery_verification.json记录证据；图件预览、灰度、可见文字边界及strict检查无FAIL，PNG均300dpi，三份PDF逐页确认DescendantFonts嵌入FontFile2。光谱首版栅格格网及自动栅格色条经检查后另存vector_v2为全矢量路径，保留此前图件、不改变任何指标。未同步work_dirs或重跑模型。

## 65D恢复 / 2026-10-08 / 用户授权一次固定三层浅层恢复
- 目标/冻结：先写PHASE65D_RECOVERY_20261008.md，153 fit相关组代表拟合/校准，32 development评价；固定2000步MsRE、Source，不换checkpoint、不重训分割网络，不读T18/65b/原65a确认/65c/旧sealed，不用ALCD。一层逻辑回归分别1/2/5输入：MsRE Shadow logit、再加Source Shadow logit、再加原生B08/B11/B12；分数系数非负、光谱系数自由，组等权BCE+固定0.001 L2，只有fit标准化、无交互/门控/调参或标签比例风险输入。
- 执行/核验：四项梯度/组等权/保守ties预算/单调基线测试本地和远端均通过，代码本地验证push后远端pull。主PID517318完成EXIT_CODE=0；306次新fit模型前向与既有153代表两模型confusion全部一致。各组均匀抽8192有效参考像元，1253376样本中25473 Shadow来自40组；三层12/12/20次L-BFGS-B收敛。按全部fit非Shadow像元冻结逐组平均FPR<=1%阈值后才加载development缓存，32景均复用核验；未重跑development分割模型。
- development数据：含Shadow支持7组，33423488有效参考像元；原MsRE/L1校准/L2加Source/L3加光谱的宏AP=0.673652/0.673652/0.686039/0.736307，池化AP=0.688675/0.688675/0.692631/0.671820，7组平均召回51.6626/65.7240/66.0360/64.9087%，32组平均FPR0.5915/1.1415/1.1603/0.8645%，mIoU77.7036/77.8100/77.8147/78.0235%。L1保留排序（float32变换AP差约1e-7无实质意义），以更多误报换召回；不能把fit预算当development固定误报率。
- 配对结果：L2-L1平均AP+1.24pp，7组条件bootstrap描述区间[-0.08,3.46]pp；L3-L2+5.03pp，区间[1.92,8.30]pp，6组改善1组下降。区间只条件于已拟合模型，不包含fit及先前development选模型不确定性，不作正式确认/像元iid显著性。T40XDR AP原MsRE0.69647→L2 0.77280→L3 0.89270（Source0.86157）；T52UDG0.62112→0.63415→0.72719（Source0.68977）；T35RPM L3也高于Source。T32TML L3相对L2下降2.62pp，不能称全部场景恢复。
- 操作限制：L3比L2平均召回低1.13pp；池化1%/5%oracle FPR召回73.97/87.31%→72.20/78.10%，不能由逐景宏AP改善宣布统一排序/操作恢复。T40XDR冻结召回48.18→46.63%且FPR1.122→0.403%，仍低于Source召回73.75%；25个无Shadow组平均FPR L2 0.0532%→L3 0.1045%，与全32组平均FPR方向不同，完整记录异质性。参考标签、7个已多轮查看development组和单seed限制仍在，像元量不增加独立支持。
- 交付/止损：新报告、锁、fit抽样审计、160条逐组CSV和图在shared/phase65d_recovery_20261008及本地outputs/phase65/phase65d_recovery_20261008，README分数据结果/结论探讨；缓存不自动传回，不同步work_dirs。保留三个冻结基线，不按结果调阈值/权重、不扩大为新机制/65c。SSH/git正常，未增加数据或模型下载；原split/Source/原65D锁保持，原正式支持门和2026-10-20预算不变。
- 最终交付核验：5份新shared统计/锁/抽样/曲线文件及3份最终图SHA完全匹配，delivery_verification.json保存证据；160行五策略数据剖析、相关组原始点/AP增量/FPR/PR彩色及灰度预览与布局检查通过，strict无FAIL，PNG300dpi、SVG全矢量，PDF Type0父字体提示后逐页确认FontFile2嵌入。三层fit阈值预算与锁绑定再次核验通过，原split与原65D锁SHA不变。

## Phase66 / 2026-10-10 / 冻结光谱候选的独立验证（启封前）
- 用户授权结束development探索，一次评价原65a_confirmation64相关组原代表；明确选择新AP终点按L3-L2组等权平均差的配对bootstrap95%区间下界>0判支持，披露预计15个云影组的小样本限制，不套用旧H1的16组门槛。原H1停止结论、split、65c禁令不变，不补样、不调参。
- 已冻结L3候选/L2删除光谱对照/L1与Source、原MsRE基线：原recovery_lock逐字节保留；协议PHASE66_20261010.md、五策略评价程序、64组及全部模型实现/配置/fit输入哈希保存于phase66_frozen_20261010。按组平均FPR固定1%预算及单侧95%上界判断，固定10000次seed66010；AP/实际阈值操作指标/oracle诊断分开。
- 本地配对单位/支持边界/单侧预算区间/exact-tie AP四项测试通过，语法检查通过。SSH前两次握手关闭后第三次恢复，原组及两个checkpoint和浅层锁SHA一致；nvidia-smi命令不可用，不据此臆测GPU不可用，推理前由Torch核验。此条写入时未读取新确认像元、未运行独立评价。先push/远端pull再核对元数据来源与既有访问清单；通过后只保留两个必要冻结模型到shared候选包，不同步work_dirs。无新训练/下载，预算2026-10-20不变。

## Phase66 / 2026-10-10 / 一次独立评价完成
- 真实执行：本地测试push36f9a9b、远端pull后统计测试通过；Torch2.1.0+cu121确认CUDA及RTX4090D可用。shared/phase66_control_20261010主PID555500完成EXIT_CODE=0；原组41251f...、来源依赖及既有manifest/index访问审计通过，先记录unseal_receipt后准备原65a_confirmation64代表。Source/MsRE共128次新前向，五策略320组记录，无拟合/阈值调整/补样，无65b/65c/sealed像元读取；仅必要两个冻结网络另存shared candidate并核对原SHA，不同步work_dirs。
- 数据：64相关组、15有效Shadow组、49无Shadow组，66846971有效参考像元；公开/实际支持一致。L3-L2组等权AP增量+10.8610pp，10000次seed66010配对组bootstrap95%CI[5.1980,17.6094]pp，13改善/0稳定/2下降，按冻结规则主要AP终点获支持。L2-L1辅助+3.9538pp，CI[-0.4589,10.4289]pp。
- 运行预算：L3全部64组平均FPR0.4813%，双侧95%CI[0.1706,0.9033]%，单侧95%上界0.8200%，本批平均预算获更强支持；8/64组超过1%，最大T39XWB8.6731%，不是每景保证。49无Shadow组平均FPR L2 0.0583%→L3 0.0650%。
- 五策略Source/原MsRE/L1/L2/L3宏AP=41.6860/32.6079/32.6079/36.5617/47.4228%，池化AP=13.5603/31.2347/31.2347/34.8870/37.5881%；宏实际召回49.8626/15.1313/25.7638/27.2969/28.7291%，池化mIoU61.6414/65.2154/66.4562/66.4509/66.9060%。宏mIoU56.1024/78.2050/75.8934/74.3596/75.2642%，不得用池化提升掩盖与原MsRE的宏mIoU代价。精确率、三类IoU、漏影流向、无Shadow逐组值及所有分母完整保存。
- 损伤/诊断：L3-L2最差T45STS AP−7.2548pp、召回−11.1572pp；T35MQV AP−1.8422pp；4个Shadow组冻结召回仍0。相对Source，T39XWB最差AP−24.9247pp，T53TLN最大召回下降83.8280pp；不同原操作点的代价不能当同FPR比较。池化1%FPR oracle召回L2 42.3951→L3 54.6058%，5%则69.7501→61.0385%，不把扫描阈值用于部署；光谱增量不等于所有预算/场景排序或实际识别全面恢复。
- 解释/止损：按预先规则本批独立AP增量和平均FPR预算联合获支持，但只有15个Shadow相关组，CI宽12.4115pp、条件于冻结候选，不覆盖fit/选模型/标签质量不确定性。结束这轮development探索，不在确认集继续调参/阈值，不改原H1失败或自动推进65c；L3作为后续机制必须超过的基线，机制投入需新独立协议。
- 本地交付：outputs/phase66/phase66_20261010/RESULTS.md分数据/探讨；报告、320行CSV、启封记录、输入manifest四份shared文件双机SHA一致，CSV独立重算主要配对AP及五策略FPR区间完全一致；模型原SHA及splitSHA再次确认不变。SSH瞬时握手关闭后恢复，无数据下载或新训练，2026-10-20预算保持。

## Phase67 / 2026-10-10 / 三个必要输入对照（执行前）
- 用户仅授权L2+RGB（同为五输入）、Source+NIR/SWIR、MsRE+NIR/SWIR；153 fit拟合/校准，32 development比较选择，固定原Source/2000步MsRE及原样本/组权重/标准化/0.001正则/优化器/1%fit误报规则。只新增3次浅层拟合，无CNN训练或新网络前向、无下载；不读取Phase66/65b/65c/sealed像元、分数或指标，不改Phase66结论及原H1失败。
- 预写PHASE67_20261010.md：Source+光谱使用Source原Surface/Cloud胜者以完全删除MsRE依赖，另外两者沿用MsRE；Shadow排序/召回/FPR可直接比较，三类mIoU另含非Shadow分类规则影响。固定比较L3分别减三个对照，7组配对bootstrap仅作描述；旧L3在development必须逐像元分数、逐组confusion/AP复现。
- development选择仅旧L3及3个新对照：实际32组平均FPR点估计<=1%中选7组平均AP最高，精确平手选低维及名称字典序；不因结果放宽预算/重新校准。输入映射/Source删除MsRE不变性/预算先于AP选择两项新测试及原4项拟合数学测试本地通过，语法通过。SSH正常、既有fit/dev缓存可用、原recovery锁SHA一致；代码push/远端pull后执行，产物仅新shared/phase67_20261010，不同步work_dirs。预算2026-10-20保持。

## Phase67 / 2026-10-10 / 三个必要输入对照完成
- 本地6项数学/输入/选择测试通过，push820b9fd后远端pull、bash语法及同6项测试通过。shared/phase67_20261010主PID556071完成EXIT_CODE=0；153 fit抽样索引/原五列样本/标签SHA逐项与65D相同，1253376样本/25473Shadow样本；L2_RGB、Source_spectral、MsRE_spectral仅3次逻辑拟合，分别25/17/20步收敛，同0.001正则及1%fit组均FPR校准，无新网络前向。
- fit阈值分别-2.3724517822265625/-1.0312467813491821/-2.306410789489746，fit平均FPR均保守<=1%。先锁control_lock后评价32 development/7Shadow/25无Shadow，旧L3像元分数、逐组AP/confusion完全复现；Source+光谱使用Source的非Shadow分类输出，其他用MsRE，不把三类mIoU差异全归于Shadow融合。
- 数据按L3/L2_RGB/Source+光谱/MsRE+光谱顺序：宏AP73.6307/71.3232/75.0050/73.5422%，池化AP67.1820/66.7388/21.6030/66.9700%，宏实际召回64.9087/63.2642/31.2227/64.9963%，宏FPR0.8645/1.0427/1.0880/0.8386%，池化mIoU78.0235/77.4485/66.9226/78.1946%。无Shadow25组FPR0.1045/0.0664/1.2094/0.0569%。完整精确率、三类IoU、漏影流向、预测比例及oracle诊断保存。
- 三个L3减对照宏AP差为+2.3075/-1.3743/+0.0884pp，7组2000次seed65310条件bootstrap描述95%区间[0.6193,4.3077]/[-5.4050,2.1046]/[-0.8603,0.9354]pp，±1pp改善/稳定/下降5/1/1、3/2/2、1/5/1；均非独立确认。L3相对MsRE+光谱在T40XDR+1.9214pp、T32NKF-2.2350pp，不能说Source处处无贡献。
- 解释/选择：匹配五输入RGB后L3平均AP仍更高，此development不支持单纯维数/RGB完全解释；纯Source+光谱逐组AP高但池化AP/实际召回很差且预算迁移超标，不能宣布Source+光谱足够。去Source的MsRE+光谱几乎持平，Source平均额外贡献未显示清楚，不能将跨零当严格等效；预设点FPR<=1%后宏AP最高规则仍选L3，不以0.0884pp微小领先夸大复杂组合必要性。L3/MsRE+光谱FPR单侧95%上界1.5099/1.4966%，development稳定预算仍不确定；不改阈值或选择规则。
- 交付/边界：outputs/phase67/phase67_20261010/RESULTS.md分数据与探讨；8策略256行CSV、21行配对原始差、3模型参数锁/153组抽样审计/原L3选择参数保存。四份shared文件SHA双机一致，新4候选宏AP/FPR和3组bootstrap独立CSV重算完全一致。原split/recovery锁SHA未变；本轮程序无Phase66/65b/65c/sealed像元、分数或指标访问，不改原H1/Phase66结论，不自动同步work_dirs。SSH正常，无新下载或分割网络训练；仅完成3必要对照，停止追加特征/正则/阈值搜索，2026-10-20预算保持。

## Phase68 / 2026-10-10 / 按用户协议执行前冻结
- 用户明确要求执行PHASE68_20261010.md；保留原立项稿，另写执行附记将未运行状态变为本次授权。冻结全部七个继续条件，不加容忍界限、不追加候选；153原fit代表/32原development代表，A/B旧L3、C/D同一个条件组均衡新拟合，各自全部组/双分层1%fit校准。复用原五输入、抽样、L3标准化、优化器、0.001惩罚及非负分数系数；只拟合一次，无网络前向，不读Phase66/65b/65c/sealed像元或分数、不加入ALCD。
- 数学/抽样权重/组内重复不变性/固定尺度梯度/exact-ties分层保守性/组bootstrap/全部继续条件六项本地测试通过，语法通过。SSH正常，远端HEAD306b90c、原recovery锁bcb91c...与Phase67锁c2a839...一致，Phase68新目录不存在。先本地验证push再远端pull运行；input_lock先于拟合、policy_lock先于development，全部32组A精确复现后才继续BCD。产物只写shared并交付本地outputs，不同步work_dirs，不改原H1失败或65c状态；无模型/数据下载，预算截止2026-10-20保持。

## Phase68 / 2026-10-10 / 固定一次拟合与四策略评价完成
- 执行：本地测试push347204a后远端pull，shared/phase68_20261010主PID559541正常完成EXIT_CODE=0，远端六测试通过。原1253376样本/25473Shadow的五输入与标签SHA、抽样索引逐项相同；有效掩膜下完整fit与抽样均40正类支持组/153负类支持组，无抽样遗漏正类支持。唯一新模型22步收敛，原L3均值/尺度原样复用；正负总权重各0.5、固定惩罚及非负分数系数。该标签支持审计不改写原H1元数据门槛/失败结论。
- 校准/复现：A/B/C/D阈值-2.2208902835845947/0.2354600578546524/2.4860661029815674/7.303001403808594；B/D的fit两层FPR均<=1%，fit总体FPR分别0.2629/0.2619%。原A在fit总体约1%时Shadow层3.6067%，确有总体掩盖分层差异。输入锁先于拟合、四策略锁先于development；全部32组A逐像元分数/逐组confusion/AP精确复现后才评价BCD。未重跑分割网络，未访问Phase66/65b/65c/sealed像元/分数或加入ALCD。
- 数据（A/B/C/D）：7组宏AP73.6307/73.6307/72.5048/72.5048%，池化AP67.1820/67.1820/68.8389/68.8389%；宏实际召回64.9087/46.9360/65.7748/47.6490%；全部32组宏FPR0.8645/0.2978/0.9247/0.3569%，Shadow7组FPR3.5787/1.3208/4.0484/1.6034%，无Shadow25组FPR0.1045/0.0114/0.0501/0.0078%；池化mIoU78.0235/76.5349/78.2569/76.9609%，零召回组均0。精确率支持分母、各类IoU、漏影流向、预测比例和1%/5%oracle诊断完整保存，不能以池化收益替代主要条件。
- 主要比较D-B宏召回+0.7130pp，10000次seed68010配对相关组bootstrap95%CI[-0.9845,3.1756]pp；D-A为-17.2597pp，CI[-22.8959,-12.0949]pp。D-A宏AP-1.1259pp，CI[-2.5009,0.0732]pp；最差T40XDR AP-4.0286pp/召回-29.3046pp。D Shadow层FPR单侧95%上界2.7655%，点估计也超1%；无Shadow上界0.0192%。D总体FPR上界0.7034%不能代替分层预算。
- 结论/止损：仅3/7继续布尔条件满足，D未超过A召回、Shadow层预算失败、宏AP及池化mIoU低于A。固定条件均衡损失的微小D-B召回增量跨零，未补回收紧校准约18pp的召回代价；池化AP/oracle改善不代表逐组排序全面恢复。按预设停止本假说，不追加候选/正则/权重搜索，不启封65c；保留原L3和既有简化候选。7组开发条件区间不是独立确认，原H1失败及Phase66旧L3/L2结论不变。
- 交付/真实校验：本地outputs/phase68/phase68_20261010/RESULTS.md按数据结果/结论探讨分开，128行四策略原始表、35行配对差、输入/模型阈值锁/153组支持审计/A复现及shared新执行日志已交付。六份shared交付文件双机SHA一致，本地独立CSV重算宏/池化confusion指标、5组配对区间、各层FPR区间及全部继续条件一致；verification证据保存。SSH/git正常，无模型或数据下载、无work_dirs同步，未执行新独立评价，预算截止2026-10-20不变。

## Phase69 / 2026-10-10 / 云预测邻域筛查执行前
- 按用户指定PHASE69_20261010.md（实际位于src/research_plans）执行，固定153fit/32development原代表及Source/2000步MsRE/L3。四行A原L3/B加中心云指示/D再加31及127邻域/S共同置换邻域；只新增B/D/S三个原组等权BCE逻辑拟合，原0.001惩罚/优化器/全部fit负类校准1%平均FPR，禁止Phase68新损失或分层校准。精确复现全部32组A先于新拟合，缺缓存即停且不补跑网络。
- 可观测特征只依MsRE原argmax与image_valid：中心对齐实际边界窗口，排除中心，无效邻居不入分母；先在全部image-valid像元完成成对置换，再选择参考有效像元用于监督/评价，标签不参与特征构造或置换。固定seed69010+产品哈希及置换SHA；局部函数无真值输入。缺有效邻居停止，不拼产品。
- 七项窗口/掩膜/中心/边界/置换/输入映射/组bootstrap与继续规则测试及原四项拟合测试本地通过，语法通过。SSH正常、远端HEAD0ff00a6且新Phase69目录不存在，既有fit/dev预测索引与image_valid路径可用；代码先push远端pull后运行，输入/特征/抽样/阈值分别落锁。D-B及D-S宏AP区间下界均>0、D召回/mIoU/零召回不劣A、全组点FPR<=1%全部满足才继续；任一不满足不加窗口/特征/拟合。无模型下载、新网络前向或确认集访问，shared新产物授权交付本地outputs，不同步work_dirs，预算2026-10-20不变。

## Phase69 / 2026-10-10 / 三个固定云邻域探针完成
- 执行：本地11测试通过，push49b2737后远端pull及bash语法/同11测试通过；shared/phase69_20261010主PID561015正常完成EXIT_CODE=0。先核验必需MsRE预测图SHA、原split/两模型/recovery锁/缓存，再完成32组A逐像元分数/逐组confusion/AP精确复现，之后准备153fit+32development的image-valid可观测云邻域。185产品中心排除/边界裁剪/无效邻居不计数及成对全image-valid置换按冻结实现；无零有效邻居异常、无缓存缺项，无新网络推理或数据扩展。
- 拟合/校准：原1253376样本、25473Shadow、五输入/标签/抽样索引SHA保持，仅B/D/S三次原组等权BCE拟合，27/32/27步收敛。原0.001惩罚、fit标准化、前两系数非负，未用Phase68损失/分层校准。B/D/S全局阈值-2.2051703929901123/-2.193368434906006/-2.1946308612823486，完整fit全部组平均FPR分别0.99999945/0.99999986/0.99999993%（精确值见锁）；模型/阈值/特征与抽样锁保存后才评价新策略。
- 数据（A/B/D/S）：7组宏AP73.6307/73.6832/73.5689/73.6026%，池化AP67.1820/66.9232/66.2683/66.5648%，宏实际召回64.9087/64.6898/64.1679/64.2518%，全部32组宏FPR0.8645/0.8601/0.8510/0.8467%，池化mIoU78.0235/77.9643/77.8713/77.9474%，零召回组均0。Shadow7组FPR3.5787/3.5323/3.4813/3.4682%，无Shadow25组0.1045/0.1119/0.1146/0.1127%；完整精确率/IoU/逐组损伤/漏影流向与支持分母保存。
- 固定比较：D-B宏AP-0.1143pp，10000次seed69010配对相关组bootstrap95%CI[-0.4792,0.1824]pp；D-S为-0.0338pp，CI[-0.3362,0.2443]pp；D-A为-0.0618pp，CI[-0.4500,0.2659]pp。D-B召回-0.5218pp，CI[-1.0572,-0.1485]pp；D-S-0.0839pp，CI[-0.3549,0.1631]pp；D-A-0.7408pp，CI[-1.5887,-0.1322]pp。D-A最差AP为T32TML-1.0825pp，最差召回T40XDR-3.0806pp。
- 结论/止损：仅2/6冻结继续条件满足，两项邻域AP区间下界不>0，D召回及池化mIoU不及A。D全部组FPR点0.8510%满足1%，但单侧95%上界1.4783%，稳定预算仍不确定；Shadow层点3.4813%/上界5.4230%，无Shadow点0.1146%/上界0.2125%，不能称每景或每层保证。按协议停止，不追加窗口/方向/特征/拟合；本次固定线性探针未见超过中心云指示或打乱对照的稳定开发增益，不证明空间信息不存在或D/S等效，更不证明太阳几何/物理对应/新算法。原H1失败、Phase66原L3/L2及65c状态不变。
- 交付/核验：outputs/phase69/phase69_20261010/README.md分数据结果/结论探讨，四行summary.csv、128行逐组表、21行配对差、三模型参数/全局阈值/输入/特征/抽样锁及shared新日志齐备。七份交付文件双机SHA一致，本地独立重算宏/池化confusion算术、AP/召回与各层FPR bootstrap、六条件及185产品置换索引SHA全部一致；verification证据已保存。SSH/git正常，无模型或数据下载、无work_dirs同步，未进行独立确认，2026-10-20预算保持。

## Phase70 / 2026-10-10 / 固定三枝探索执行前
- 用户授权执行PHASE70_20261010并确认晋级单组改善严格ΔAP>1pp且至少4/7。仅原153fit/32development，先精确复现L3；A六个冻结标准化交互/B三列61窗口局部光谱对比/C三列原生归一化波段差，分别另拟合产品内全image-valid共同置换对照，共最多6个串行原组等权BCE浅层拟合，无特征组合/搜索。沿用原样本、0.001惩罚/优化器、全部fit负类全局1%平均FPR阈值；模型/阈值先落锁再评价。
- 九项特征/有效邻居/边界/置换/阈值/组bootstrap/严格晋级测试本地通过，语法通过。SSH及既有conda正常；现有shapefile有UTM CRS及FID字段，不能假定为cube格网或太阳角度，D仅查185产品元数据并记录缺项。代码先本地验证push远端pull后运行；新shared结果及日志授权交付本地新outputs/phase70目录，不同步work_dirs。不访问确认集、不新增分割网络前向/下载、不改旧失败结论；2026-10-20预算保持。

## Phase70 / 2026-10-10 / 三枝与匹配打乱对照完成
- 执行/输入：本地及远端13测试通过，冻结代码f6de77d本地push远端pull后启动shared/phase70_20261010，PID561721最终EXIT_CODE=0。原split/Source/MsRE2000/recovery锁/缓存与代表边界核验；32组原L3逐像元分数、AP、confusion精确复现先于新拟合。153fit+32development三枝可观测特征全部有限、B无零有效邻居、C三列分母保护计数均0/193229001 image-valid像元，未截断。原1253376样本、25473Shadow、原五列/标签/每组抽样索引SHA保持。
- 拟合/阈值：仅A/A_shuffle/B/B_shuffle/C/C_shuffle六次串行原组等权BCE浅层拟合，40/47/36/36/33/32步均收敛，无补跑/调参/新网络前向。6阈值分别-2.2133591175/-2.1163117886/-1.2939959764/-2.2204942703/-2.2743988037/-2.1933946609，各自在完整fit负类校准的全部组平均FPR严格<=1%，模型/标准化/阈值全部落锁后才评价新development分数。原L3非Shadow仍用固定MsRE胜者。
- 数据（L3/A/A_shuffle/B/B_shuffle/C/C_shuffle）：7组宏AP73.6307/72.1854/71.7253/79.0048/73.6202/74.5496/73.3138%；池化AP67.1820/65.2970/66.5273/72.2023/67.1633/66.8917/66.1739%；宏实际召回64.9087/62.2576/64.7569/64.5788/64.9097/65.6339/65.1347%；全部32组宏FPR0.8645/0.9110/0.9689/0.7335/0.8640/0.8400/0.8408%；池化mIoU78.0235/77.4796/77.7001/78.1293/78.0249/78.0267/77.8718%；零召回组均0。完整分层FPR/精确率/IoU/漏影流向及支持分母见交付。
- 固定配对AP（10000次seed70010相关组bootstrap描述95%CI，单位pp）：A-L3 -1.4453[-4.4923,0.3615]，A-A_shuffle +0.4601[-2.4847,2.5069]；B-L3 +5.3741[-0.6937,12.1196]，B-B_shuffle +5.3845[-0.6883,12.1296]；C-L3 +0.9189[-0.4173,2.5161]，C-C_shuffle +1.2358[-0.6039,3.2655]。A/B/C相对L3按严格>1pp的改善/稳定/下降分别0/5/2、4/0/3、2/4/1，>0改善2/4/4。B去掉任一Shadow组后的平均AP增量3.3540至6.9308pp均正。
- 实际召回/损伤：A/B/C相对L3宏召回分别-2.6512pp[-9.7386,1.7454]、-0.3299pp[-4.6318,4.7808]、+0.7252pp[-0.8251,2.4750]。A最差T40XDR AP-10.2568pp/召回-22.7492pp；B最差T32TML AP-3.9661pp/召回-8.8925pp；C最差AP T52UDG-1.1857pp、召回T35RPM-2.2006pp。B全组FPR点0.7335%/单侧95%上界1.2561%，Shadow层2.7008%/4.3736%，无Shadow层0.1827%/0.3398%；无Shadow点高于L3，不能以全组下降替代逐层风险。
- 结论/冻结晋级：A满足3/8、B 7/8、C 6/8，三枝均不晋级。B虽满足AP点增量、4/7严格改善、全部leave-one-out正、mIoU/误报点预算等条件，但宏实际召回低于L3，按原规则失败；AP区间跨零另表不确定性，不加作新门槛。C未达+1pp及4/7，A损伤明显。按协议收口，不修改阈值/窗口/公式/正则、不组合分枝或追加实验。仅7个正类development组的多枝探索，不作独立确认或机制证明；原H1失败、Phase66及68/69状态、65c未授权保持。
- D：185产品现有shapefile CRS已核验，DBF均仅FID；产品名/原ZIP成员SHA与cube对应有证据，但太阳方位/天顶角及cube CRS/仿射格网/像元大小/朝向缺可靠对应，未就绪。逐产品元数据表及来源保留；未补造、下载或投影，输入缺项不算几何机制无效。
- 交付/核验：本地outputs/phase70/phase70_20261010/README.md分数据结果/结论探讨，七行summary.csv、224行逐组CSV、42行配对差、185产品元数据表、模型/标准化/阈值/输入/特征/抽样锁及shared运行日志齐全。九份shared交付双机SHA一致；本地独立重算CSV宏指标/池化confusion算术、配对AP/召回与分层FPR bootstrap、全部8项晋级及555个分枝置换SHA一致。SSH/git正常，无数据/模型下载，无work_dirs同步或清理，无新确认集访问；2026-10-20预算保持。

## Phase71 / 2026-10-10 / B特异性与逐波段固定删除执行前
- 用户追加明确授权围绕B仅回答光谱特异性和逐波段收益/损伤，固定RGB_local与B_noB08/B_noB11/B_noB12四个新对照。原五输入始终保留，RGB追加同61窗口/同公式/同三维数的原生B04/B03/B02局部对比；删除仅对应B局部对比列，其他输入不改，均独立重新拟合。先复现原L3/B全部32组AP/confusion；原153fit/32development、样本/损失/惩罚/优化器与全部fit负类1%组平均FPR校准保持，四模型/阈值全落锁后才评价。无搜索/组合/新shuffle/后验阈值修复，四次串行拟合后止。
- 三项RGB同公式原生通道/有效邻居中心排除/删除列映射本地检查通过，语法通过。固定比较B-RGB及B-三个删除、各策略-L3，10000次seed71010组配对描述区间；全部32组原值与对L3/B差异交付，预先指定下降组T35RPM/T32TML/T40XDR完整检查。RGB对照只检验在原五输入条件下新增局部信息的特异性，删除是重新拟合条件贡献，不作物理因果证明或新晋级规则。
- 代码本地验证push后远端pull执行，新shared/phase71_20261010与本地outputs/phase71/phase71_20261010防覆盖。原Phase70 B未晋级、原H1失败/65c未授权保持，不访问确认/旧sealed，不加ALCD，不下载/新增分割前向或同步work_dirs；预算2026-10-20不变。

## Phase71 / 2026-10-10 / B特异性四个固定对照完成
- 执行/核验：冻结代码8b6a076先本地push远端pull，shared/phase71_20261010 PID565318完成EXIT_CODE=0。本地16检查/远端7检查通过；L3逐像元分数及L3/B全部32组AP/confusion复现，B本轮复现和正式评价分数SHA一致。原split/模型/recovery锁、153fit/32development边界、原1253376样本25473Shadow及五列/标签/索引SHA保持。新RGB特征仅原生TOA/影像有效邻域，全部185产品有限，无零有效邻居；原B特征/模型复用不重训。
- 仅四次串行浅层拟合RGB_local/B_noB08/B_noB11/B_noB12，38/27/31/30步均收敛；fit全局阈值-1.4447878599/-1.3018645048/-1.3086204529/-1.3173230886，完整fit负类全部组平均FPR均严格<=1%。原组等权BCE/0.001惩罚/优化器/样本及非Shadow胜者不变，四模型参数/标准化/阈值全落锁后才评价，未补跑/调参/下载/新增分割网络前向。
- 数据（L3/B/RGB_local/B_noB08/B_noB11/B_noB12）：宏AP73.6307/79.0048/74.4113/77.6122/79.0937/78.4761%；池化AP67.1820/72.2023/68.7761/71.9187/72.1898/72.2315%；宏实际召回64.9087/64.5788/61.7398/64.5938/64.5500/64.2601%；全组FPR0.8645/0.7335/0.7302/0.7358/0.7354/0.7362%；池化mIoU78.0235/78.1293/77.4593/78.1386/78.0972/78.0945%；零召回均0。完整分层FPR/精确率/IoU/漏影流向与分母见交付。
- 特异性：固定B-RGB宏AP+4.5934pp，10000次seed71010相关组配对描述95%CI[0.3490,10.2358]，4/2/1组按>1pp改善/稳定/下降；宏实际召回+2.8391pp，CI[-0.1470,6.1183]。支持在保留原五输入条件下，新增NIR/SWIR局部对比优于这一固定RGB局部亮度对照的开发排序信号，不等于纯RGB与全多光谱比较、独立确认或排除所有亮度/纹理混杂。全部区间本轮统一seed71010，原B/L3点值不变。
- 波段条件增量（B-删除版）：B08宏AP+1.3926pp，CI[-0.3305,4.4276]，实际召回-0.0150pp；B11宏AP-0.0889pp，CI[-0.3380,0.1024]，召回+0.0289pp；B12宏AP+0.5287pp，CI[-0.1349,1.3849]，召回+0.3187pp。B08 AP均值主要来自T38TNN+10.2549pp，其余六组均值-0.0845pp；B11七组差均在±1pp，原始B11仍保留，不能称B11无用；B12较大正AP增量在T38TNN+2.7144及T32NKF+1.2862。三条平均删除AP区间均跨零，不证明任何单波段普遍必要，重新拟合删除差异不是物理因果。
- 三下降组（删除版相对B，B08/B11/B12顺序）：T35RPM ΔAP -0.1988/-0.0871/+0.6190pp，Δ召回+0.0216/-0.1508/+0.4000pp；T32TML ΔAP+0.2806/+0.1528/-0.0147pp，Δ召回+0.2758/+0.3040/+0.3457pp；T40XDR ΔAP+0.0845/+0.1425/+0.0682pp，Δ召回-0.0596/-0.6292/-0.5696pp。所有删除版本在三个组的AP及实际召回仍低于L3，不能用单一删除消除损伤。T32TML主要新增Shadow→Surface（L3 16140到B 26049，Shadow→Cloud 636到527）；T40XDR漏影全流向Surface（45631到47942）；T35RPM Surface 50820到58129/Cloud11184到15137均增。仅定位误差去向，不推断地物或物理成因。
- 交付/止损：outputs/phase71/phase71_20261010/README.md按数据结果/结论探讨及两个问题报告，六行summary、192行逐组原始表、63行配对差、32行全部组对L3/B的AP/召回/FPR绝对及变化、18行三下降组细查、输入/特征/模型/阈值/样本锁、shared日志齐全。七份交付双机SHA一致，本地独立重算全部宏指标/池化confusion算术及9项配对和分层FPR区间，旧L3/B原值一致。SSH只读查询一次自动审批超时后允许重试成功，任务未重复启动，其他SSH/git正常。四个授权对照结束后停止，无新晋级规则/阈值搜索/组合删除/独立确认/65c；原H1失败及原B未晋级保持。无work_dirs同步或清理，2026-10-20预算保持。

## 2026-10-10 Phase72：独立确认协议与元数据准入核验（未启封）
- 目标：结束development优化，只制定B/L3/RGB_local一次独立确认协议；用户确认B−L3逐组平均召回及mIoU单侧95%下界各≥−1pp、B平均FPR单侧95%上界≤1%。主要B−L3 AP区间下界>0后才检验B−RGB AP；无新模型/特征/阈值。
- 交付：T32NKF_noB12既有结果单独补表，AP54.0810424112%、实际召回31.4156499790%、FPR0.9628978486%；相对B AP−1.2862121688pp、召回−1.6459823307pp。原Phase71模型/输入锁校验一致，完整B/L3/RGB参数、标准化、阈值及输入SHA保存在phase72_frozen_20261010/confirmation_lock.json。无重拟合。
- 元数据：原65c_final64组66产品、公开支持12组；只读取manifest/index/receipt等24份访问清单和原谱系依赖，未发现65c产品访问记录，6项谱系依赖SHA一致；固定组/产品不交叉，全部66成员与其他固定池及65c不同组之间缓冲WGS84包围框交叠均0。审计不证明不存在未记录/手工访问，条件于原footprint准确性；未读取确认像元/预测/指标。
- 执行/止损：SSH及Git网络正常，下载未重启；本轮无CNN前向、训练或评价。协议与冻结候选提交push后远端pull --ff-only；后续启动指令和执行代码检查前不启封。原H1支持门失败、Phase70 B未晋级保留，work_dirs自动同步/备份/清理继续禁止。数据结果与结论探讨分开，不把本轮准入核验称独立确认成功。
## 2026-10-10 Phase72执行授权（启封前）
- 用户明确授权执行Phase72；沿用7b2858b冻结协议与B/L3/RGB_local参数锁，不改窗口、输入、阈值或统计门。新增一次性执行器、相关组统计与4项合成数据测试；本地全部通过。execution_lock单独记录本次授权与代码SHA，原confirmation_lock保留制定协议时的evaluation_authorized=false历史字段，不改写原锁。
- 执行先重查访问元数据/谱系/输入/共享候选checkpoint并精确复现32组development三个策略，随后写启封收据再准备原65c_final64代表。仅两网络共128前向、三冻结决策，无拟合/训练；原H1及Phase70失败保留，禁读旧sealed，不同步work_dirs。最终结果仍按数据结果/结论探讨分开。
- 启封前执行修复01：首次主进程566522已退出、EXIT1，phase72数据目录尚不存在，未启封。原候选JSON本地881处CRLF使本地SHA17c8d6…与Git/远端LF SHA018253…不同；字典参数未变。保留原execution_attempt01_lock和控制日志，将执行锁绑定实际Git部署字节SHA，新控制目录retry01；不绕过校验、不改候选/统计/阈值。
- 启封前执行修复02：主进程566599已退出EXIT1，未创建phase72数据目录/启封收据。既有冻结输入锁路径含Windows反斜杠，Linux Path.name未提取basename；修复仅规范路径分隔符后核对同一原始SHA，保留attempt02锁与retry01日志，新控制目录retry02。不读取确认像元、不改冻结候选/统计。
