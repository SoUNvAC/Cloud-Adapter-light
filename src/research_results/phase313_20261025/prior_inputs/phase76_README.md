# Phase76：三景cube对应验证

## 1 数据结果

执行完成，EXIT_CODE=0。六个精确原L1C对象全部通过长度、固定generation、镜像MD5及SHA256校验；新增443,120,797 bytes，低于冻结的1,000,000,000 bytes上限。只读取Phase75原三景fit cube和B11/B04原始影像，无标签、模型输出或确认像元访问。

每景80个固定候选，唯一合格候选均为行/列各内移65个20m像元、原方向d4_0；25/25选择点及24/24留出点均有完整支持。B11留出及1022×1022整图通过原误差门。B04两种实现均未通过原门。两波段整图双方有效率、有效性一致率均100%。

下表为整图TOA绝对误差。原界限：B11中位≤5e-6、P99≤2e-5；B04中位≤5e-5、P99≤2e-4。

| 场景 | B11中位 | B11 P99 | B04中位 | B04 P99 |
|---|---:|---:|---:|---:|
| T37LDL | 5.412e-09 | 3.891e-08 | 0.003675 | 0.042525 |
| T20PMQ | 6.342e-09 | 2.384e-08 | 0.002325 | 0.020800 |
| T46RBU | 5.865e-09 | 3.080e-08 | 0.004350 | 0.074900 |

B04完整格网重采样再裁切，与覆盖目标范围并加2个原生像元边缘的窗口重采样，在三景全部1,044,484格逐值完全相等，最大差值0。两者均采用冻结的纯中心双线性四邻点公式，无抗混叠，不使用GDAL缩小图像时可能不同的滤波变体。库和参数详见mapping_lock.json。未按残差选择软件、窗口、偏移或调整容差。

## 2 结论分析

B11原生20m像元在三景均支持65像元内裁、行向下/列向右的对应；误差约float32舍入量级。该证据仅覆盖三景和B11，不能传播到全185景或其他波段。

B04与发布cube的数值仍存在明显差异，无法用全格网与带边缘窗口的实现差异解释。高相关系数不构成像素等价证明。现有结果不能确定差异来自作者的插值滤波、采样中心约定、其他预处理还是其他原因；本轮不进一步搜索这些变体。

三景最终均为B11_verified_B04_pending。保留B11对应公式，但不把其称为已验证的全波段cube affine，不宣布六波段映射或云影投影就绪。地理北—投影格网—行列方向转换仍未验证。本轮没有训练或方法性能比较，旧研究停止结论不变。

## 文件索引与复核

- mapping_lock.json：首次JP2像元读取前冻结的输入、cube SHA、候选、公式、采样点、容差和软件版本。
- object_verification.json：六对象URL、长度、MD5/generation、SHA256；大JP2仅在远端shared。
- scene_status.csv / summary.json：逐景状态和完整数值；validity_audit.csv：双方有效性与不一致计数。
- B11_only_transforms.json：明确标记为B11范围的CRS、affine、角点/中心及行列公式，不是全波段通过证明。
- 每景目录：B11候选与选择点CSV、留出点CSV、空间分区诊断，B04两实现49点CSV和整图残差NPY。B04_implementation_difference.npy保存全部原始差值。
- artifact_hashes.json：远端审计产物SHA；LOCAL_VERIFIED.json：本地逐文件及Git blob/Phase75输入/cube成员SHA复核。
- PID、GIT_COMMIT、EXIT_CODE、console.log、error.log、existing_object_scan.json：shared控制日志。远端五项合成检查通过；error.log包含affine未来API弃用提醒，无运行错误。

报告SCP握手两次被远端关闭后，改用既有SSH压缩流成功取回，未重跑审计。仅传shared审计产物，不涉及work_dirs同步。远端结果：/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/phase76_20261010/。
