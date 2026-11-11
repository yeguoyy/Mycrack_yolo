# 独立维护的裂缝转换输入

- `manifest.csv`：原清单的独立副本，保留 5462 个样本及原来的同源照片划分；路径改为此目录下的文件。
- `labelme/outputs_RGB`、`V7/outputs_RGB`：原始照片。
- `labelme/outputs_Mask`、`V7/outputs_Mask`：原始彩色 mask，绿色裂缝 `(0,128,0)`，红色剥落不转换。
- `source_snapshot.json`：初始化时的清单 SHA256、样本数和拷贝信息。

图片及 mask 是实际文件副本，与项目外的源文件独立。转换脚本默认读取这里的清单；
以后新增或删除样本时在此清单维护，保留 source_group 不跨 train/val/test 的约束。
清单的 boxes、max_aspect 等统计沿用初始基础转换结果，作为抽样参考；V2 标注始终从 mask 重新计算。

完整输出：`prepare/prepare_crack_obb_v2.py --full`，输出到 `data/crack_obb_auto_v2/`。
