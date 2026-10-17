# 证据索引

| 阶段 | 比较 | 适配组数 | seed | 工作点 | 数据池 | 来源 |
|---|---|---|---|---|---|---|
| 119 | E/R/N单成员与双模型 | 0 | none | 0.5 | Catalogue/PixBox | phase119 summary/benchmark/resource receipts |
| 120 | C153监督/D153蒸馏 | 153 | 12001 | 0.5 | Catalogue/PixBox | phase120 endpoint receipts;124 historical background |
| 121 | C153/D153第二seed与两seed汇总 | 153 | 12001/12002 | 0.5; historical oracle diagnostic not reused | Catalogue/PixBox | phase121 comparisons;124 historical background |
| 122 | C100监督及固定R校准 | 100 | 12201 | 0.5/own Tcal;53 calibration | Catalogue/PixBox | phase122 locks/receipts;124 accepted tables |
| 123 | C100第二seed/两seed指标均值 | 100 | 12201/12301 | 0.5/own frozen Tcal | Catalogue/PixBox | phase123 receipts;124 intervals and evidence |
| 124 | 性能/工作点/成本整理与复核 | no new fit | no new seed | unchanged | separate pools | phase124 accepted documentation; no new experiment |

以下只核验本轮实际引用的小材料，不重复124全量274条交付审计，也不重新读取模型、影像、参考或分数数组。已有模型SHA与位置沿用124验收记录。

## 本轮材料与SHA

| role | path | sha256 |
|---|---|---|
| accepted124 material | src/research_results/phase124_20261017/README.md | 4f2a39c6b134707d80b66cc071ed0b334d22f2bcd5201abd8e94b43e841807e2 |
| accepted124 material | src/research_results/phase124_20261017/CLAIMS_FOR_PHASE125.md | 2eac3060715826a5fa54de58224c902f397e5104220ca5edaa1436e6a33860a1 |
| accepted124 material | src/research_results/phase124_20261017/Catalogue_group_evidence.csv | df5a7d009d78ba27c347c2ad4679d1f47460941fba25ca2e1a99977477108f02 |
| accepted124 material | src/research_results/phase124_20261017/PixBox_group_evidence.csv | 3c636b449ea3c2217e32bf168d513c20e616854fa07559f025e146bed3e7fded |
| accepted124 material | src/research_results/phase124_20261017/Catalogue_performance.csv | f71e78ec90e1d55608a7427da0f489732b0bf3319be4dc4354e7195d70a15448 |
| accepted124 material | src/research_results/phase124_20261017/PixBox_performance.csv | 7c93b8981d706699f8bfe67c2915aeeb0f0c83223e089340a7c500ddd5f94810 |
| accepted124 material | src/research_results/phase124_20261017/DEPLOYMENT_COST_AND_SOURCES.csv | 77e5c157454e1e009b1abe278bc5f4ccf426d495db6c99ea691ab62ba54449fa |
| accepted124 material | src/research_results/phase124_20261017/MISSING_AND_LIMITATIONS.csv | 785d4831d66857288c153dda5c591d4dde05a8adb011dac8dfda5130c7e9066c |
| accepted124 material | src/research_results/phase124_20261017/HISTORICAL_BACKGROUND_ONLY.csv | 38fc5edb80d6dbfae9d798ba36b5c1a55589304e7b775ee603505d3fb6aa3e8a |
| accepted124 material | src/research_results/phase124_20261017/MODEL_MEMBER_IDENTITY.csv | 7948e04402f1280c15e06771230b724ecd90b3df885956c66bbf90a4c7c2fcbb |
| accepted124 material | src/research_results/phase124_20261017/COST_METADATA_ACCEPTANCE.json | bdbfa9ead0a76d993fad21d9e6ea95ff9f32f932f772e923fd927ca8debf8a23 |
| accepted124 material | src/research_results/phase124_20261017/SUPPORT_AND_CHAIN_AUDIT.json | 94f1d7f263d33f604bb4136c474a4fceadb0465f3594ec922db787eeeeac95e2 |
| accepted124 material | src/research_results/phase124_20261017/Catalogue_INTERVALS_REUSED.json | 0c11cbecd7f72c57c0f1e227d5ca2871f2b6cd8d2591d2c62239aab48db64e42 |
| accepted124 material | src/research_results/phase124_20261017/PixBox_INTERVALS_REUSED.json | ba460975eaa9b2870a6728338a14c42ea50b06f88031aa9e918fef58842aad0e |
| accepted124 material | src/research_results/phase124_20261017/PRE_EXECUTION_LOCK.json | 8b14321857625e6ba60f721166def23b206708de21c7f07083d6faa4b7b9b782 |

## 权重身份（只引用，不复制）

| phase | arm | adapt_training_groups | calibration_groups | model_sha256 | local_location | controlled_shared_location |
|---|---|---|---|---|---|---|
| 120 | C | 153 | 0 | e642302e66239870f7af4b01d1fa784b1266d701913298d30f92401bfdcf32e7 | outputs/phase120/phase120_20261015/C_decoder_final.safetensors | /home/scv/shared/phase120_20261015/C_decoder_final.safetensors |
| 120 | D | 153 | 0 | 618edcb4915ce766fc3d1e797547f8f602890ab516d1a561ebe01a24c900bad9 | outputs/phase120/phase120_20261015/D_decoder_final.safetensors | /home/scv/shared/phase120_20261015/D_decoder_final.safetensors |
| 121 | C | 153 | 0 | 445d34e4d240281abfdf48c5a42df528c27880133f93b836ae69ad11e88fe1e4 | outputs/phase121/phase121_20261015/C_decoder_final.safetensors | /home/scv/shared/phase121_20261015/C_decoder_final.safetensors |
| 121 | D | 153 | 0 | 7871b7df353a154dde116e1214ea623b0b24d26cc68924e63f59712a6f84c8bf | outputs/phase121/phase121_20261015/D_decoder_final.safetensors | /home/scv/shared/phase121_20261015/D_decoder_final.safetensors |
| 122 | C | 100 | 53 | 59f88f462b85c6a2eb8725781085efbc474962e862dd0a2401f4b344d88b5b9e | outputs/phase122/phase122_20261015/C_decoder_final.safetensors | /home/scv/shared/phase122_20261015/C_decoder_final.safetensors |
| 123 | C | 100 | 53 | 2ad9b8dea32e8cb06e33656fc0a4518a412a85722432bcc21abf604d463238dd | outputs/phase123/phase123_20261016/C_decoder_final.safetensors | /home/scv/shared/phase123_20261016/C_decoder_final.safetensors |

原R的已声明基础权重SHA：044f2d6a3b4b59817f19d4ca080485f01c4bf9a6fddd053df184ee98144b46a2；本轮不把旧身份记录称为新的字节核验。E/R/N原始二进制复核范围沿用124缺项表。

## 阈值、输入、代码与完整结果

- threshold and support chain: [SUPPORT_AND_CHAIN_AUDIT.json](D:/Cloud-Adapter-light/src/research_results/phase124_20261017/SUPPORT_AND_CHAIN_AUDIT.json)；SHA256 `94f1d7f263d33f604bb4136c474a4fceadb0465f3594ec922db787eeeeac95e2`。
- training member identity: [MODEL_MEMBER_IDENTITY.csv](D:/Cloud-Adapter-light/src/research_results/phase124_20261017/MODEL_MEMBER_IDENTITY.csv)；SHA256 `7948e04402f1280c15e06771230b724ecd90b3df885956c66bbf90a4c7c2fcbb`。
- input and source SHA index: [INPUT_SHA_LOCK.json](D:/Cloud-Adapter-light/src/research_results/phase124_20261017/INPUT_SHA_LOCK.json)；SHA256 `ada6a0433148be1bada41b160b418f59edd5fe5e872d1e05177477fb8b0db7e0`。
- 124 code SHA: [PRE_EXECUTION_LOCK.json](D:/Cloud-Adapter-light/src/research_results/phase124_20261017/PRE_EXECUTION_LOCK.json)；SHA256 `8b14321857625e6ba60f721166def23b206708de21c7f07083d6faa4b7b9b782`。
- Catalogue original123 intervals: [Catalogue_INTERVALS_REUSED.json](D:/Cloud-Adapter-light/src/research_results/phase124_20261017/Catalogue_INTERVALS_REUSED.json)；SHA256 `0c11cbecd7f72c57c0f1e227d5ca2871f2b6cd8d2591d2c62239aab48db64e42`。
- PixBox original123 intervals: [PixBox_INTERVALS_REUSED.json](D:/Cloud-Adapter-light/src/research_results/phase124_20261017/PixBox_INTERVALS_REUSED.json)；SHA256 `ba460975eaa9b2870a6728338a14c42ea50b06f88031aa9e918fef58842aad0e`。
- cost sources: [DEPLOYMENT_COST_AND_SOURCES.csv](D:/Cloud-Adapter-light/src/research_results/phase124_20261017/DEPLOYMENT_COST_AND_SOURCES.csv)；SHA256 `77e5c157454e1e009b1abe278bc5f4ccf426d495db6c99ea691ab62ba54449fa`。
- unmeasured and unknown: [MISSING_AND_LIMITATIONS.csv](D:/Cloud-Adapter-light/src/research_results/phase124_20261017/MISSING_AND_LIMITATIONS.csv)；SHA256 `785d4831d66857288c153dda5c591d4dde05a8adb011dac8dfda5130c7e9066c`。

123完整阈值/输入/代码/分数封存身份继续按124的INPUT_SHA_LOCK索引引用；无新阈值、模型或后处理版本。C100的精确阈值为R=0.7693686485290528、C122=0.36361625790596014、C123=0.2778868973255158，按float64比较。
