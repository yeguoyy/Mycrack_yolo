# 裂缝大段旋转框数据集 V2.1

完整 YOLO OBB 数据集，类别 `0: crack`。每一大段一个旋转框，明显改变走向才分割。

| 划分 | 图片 | 框 |
| --- | ---: | ---: |
| train | 3822 | 8958 |
| val | 818 | 1828 |
| test | 822 | 1890 |
| 合计 | 5462 | 12676 |

## 训练入口

训练参数 `data` 指向本目录的 `data.yaml`：

```python
data=r"D:\Code\wall_damage_identification_model\yolo\data\crack_obb_auto_v21\data.yaml"
```

本次只生成数据，没有修改训练脚本或启动新训练。原有 V1/V2 数据保留。

## 来源及规则

- 输入为独立维护的 `data/crack512/manifest.csv`，原图及 mask 位于同目录的 `labelme/`、`V7/`。
- 沿用输入清单的 train/val/test 及 source_group；147 个来源组没有跨集合。
- 只提取绿色裂缝 mask，过滤不足 12 像素的连通区域；共过滤 1172 像素。
- 使用 fine-crack-tracing 提取中心线和分支，本项目按平滑走向进行大段切分。
- 转向阈值 30°，持续至少 16 px 才切分，切分两侧弧长至少 64 px。轻微摆动不拆碎。
- 在同一连通区域内合并相接且方向接近的碎段，按各段全部保留像素拟合旋转矩形。
- 框扩展到像素边界后每侧留 2 px 余量，短边至少 8 px。
- 倾斜框超出原图时，输出图片副本补 RGB `(114,114,114)` 灰边并同步平移标签，保留框角度。
  本数据集 5339 张图片有补边，单侧最大 80 px；原图不改。标签按补边后的输出尺寸归一化。
- 旧 V2 标签只用于中间对照栏，不参与 V2.1 标签生成。

转换脚本为 `prepare/prepare_crack_obb_v2.py`，本次命令：

```powershell
& .\prepare\.venv\Scripts\python.exe .\prepare\prepare_crack_obb_v2.py --full --profile relaxed --split-angle 30 --output data/crack_obb_auto_v21
```

## 检查与文件

全量校验通过：四角坐标在范围内、矩形有效、无重复或缺失标签、无来源组划分泄漏。
写入并重新读取标签后，保留的 28623810 个裂缝像素覆盖率为 100%。
三个集合均通过 Ultralytics OBB 加载及样例 batch 检查；36 张预览的标签与此前预览逐字一致。
这些是数据几何及加载检查，训练效果需通过新训练评估。

- `images/`、`labels/`、`data.yaml`：训练数据与入口。
- `index.html`、`review/`：36 张代表样本的 V2 与 V2.1 对照图。
- `traces/`：36 张代表样本的中心线调试信息，训练不读取。
- `manifest.csv`：逐图统计、输入路径、原始/输出尺寸、补边量。
- `summary.json`：完整统计、参数及上游版本。
- `validation.json`：本次全量检查结果。
- `source_groups.csv`：来源组划分。
