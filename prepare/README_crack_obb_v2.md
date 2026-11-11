# 裂缝 mask → 大段 YOLO OBB（V2.1）

`prepare_crack_obb_v2.py` 默认使用 `--profile relaxed`（V2.1）：每个大段一个旋转框，
只有走向明显改变才切分，轻微弯曲不切碎。`--profile v2` 保留此前较细的 V2 规则。
fine-crack-tracing 1.0.0 负责中心线、分支和宽度；大段切分、像素分配、框拟合及导出为本项目代码。
原始图片、mask、现有 V1/V2 数据集及训练脚本均只读。

## 环境

上游代码固定在 `vendor/fine_crack_tracing/`，MIT 许可证、提交号和来源见该目录的
`LICENSE`、`UPSTREAM.json`。新增骨架化依赖放在 `prepare/.venv/`，继承现有
`pytorch_cu126` 的 numpy、SciPy 和 OpenCV，不更改训练环境。

首次在项目根目录配置：

```powershell
& 'D:\Anaconda3\envs\pytorch_cu126\python.exe' -m venv --system-site-packages prepare\.venv
& .\prepare\.venv\Scripts\python.exe -m pip install --no-cache-dir -r .\prepare\requirements_crack_obb_v2.txt
```

## 预览

独立输入清单为 `data/crack512/manifest.csv`。它从原清单复制后单独维护，保留样本和
train/val/test 同源分组，原图及 mask 已实际复制到同目录的 `labelme/` 和 `V7/` 下，
清单路径全部指向这些本地副本。脚本不再读取其他数据集的 CSV。
V2.1 的中间对照栏只读 `data/crack_obb_auto_v2/labels/`，这些旧标签不参与新标签生成；
可用 `--compare-root` 改对照数据集位置。旧 `--profile v2` 则从当前 mask 重算 V1 基础框作对照。
`source_snapshot.json` 记录初始快照校验信息。
若移动原始数据，可用 `--source-root D:\path\to\crack512` 重定位路径。

```powershell
& .\prepare\.venv\Scripts\python.exe .\prepare\prepare_crack_obb_v2.py
```

默认处理 36 张样本，包含此前展示的验证样例、低/高长宽比、多连通区域和空标签样本。
输出到 `output/crack_obb_v21_preview/`，已非空的目录会拒绝覆盖，可用 `--output` 指定新目录。

- `index.html`：可浏览的全部样本对照。左：原 mask 与中心线；中：V2；右：V2.1。
- `review/<split>/<sample>.jpg`：单图三栏对照；`review/overview.jpg`：分叉样例概览。
- `labels/<split>/`：归一化四角坐标的 YOLO OBB 标签；`images/`：输出图片，必要时在原图外补灰边。
- `traces/<split>/`：预览样本的中心线、宽度及图结构 JSON，仅用于检查，训练不读取。
  其中路径坐标属于原图；`training_image_padding_ltrb` 记录输出图片的左、上、右、下补边量。
- `manifest.csv`：逐图框数、保留 mask 覆盖率、框内背景像素、回退和审核标记。
- `summary.json`：配置、上游版本和汇总结果；`data.yaml` 明确标记预览子集。

指定样例：

```powershell
& .\prepare\.venv\Scripts\python.exe .\prepare\prepare_crack_obb_v2.py --samples IMG_6637_35 IMG_6637_36 --split-angle 30 --output output\crack_obb_v21_custom
```

## V2.1 几何规则

1. 严格提取 BGR `(0,128,0)`，红色剥落区域不转成裂缝；未知颜色报错。
2. 与 v1 相同，过滤不足 12 像素的连通区域；单独记录这些像素数量。
3. 只用骨架模式，禁用自动补断裂和填洞；短毛刺剪枝仅影响路径，所有保留 mask 像素仍参与分配。
4. 主干和可见分支分别提取；长度不足 `max(24 px, 3 × 局部中位宽度)` 的毛刺不单独建段，
   其像素仍保留。不因主干上出现分叉节点就强制切断主干。
5. 用 48 px 邻域估计平滑走向，与当前大段的参考方向比较。方向变化达到 `--split-angle`（默认 30°），
   且持续至少 16 px 才切分；切分两侧弧长至少 64 px（`--min-segment-length`）。
   轻微摆动、单像素毛刺和贴边都不触发额外切分。最短长度仅限制拆段，不删除真实短裂缝。
6. 仅在同一原始连通区域内按最近中心线分配像素。少于 12 像素的小段合入最近的有效段；
   无有效中心线的区域使用整个区域拟合，并记录 `component_without_centerline`。
7. 同一连通区域内相接、方向接近的段允许合并，普通段夹角阈值为转角阈值减 5°，
   合并框面积不超过分开框面积和的 1.35 倍；不足 32 px 的碎段放宽到 1.8 倍。
   不跨不连通区域合并，保留真实分支。
8. 对每个大段的全部 mask 像素拟合最小面积旋转矩形，扩到像素边界并每侧加 2 px 余量；
   最小短边为 8 px。框角度由裂缝走向决定，接近水平的裂缝仍会得到接近水平的框。
9. 倾斜框角点超出原图时，只在输出图片副本外添加必要的 RGB `(114,114,114)` 灰边，
   并同步平移框、预览中心线和 mask。保留旋转矩形，不改为大水平框，也不因贴边切碎。
   原图文件不改；输出尺寸可能略大于 512，训练缩放时原图的有效分辨率会略低。

坐标规则采用像素边界 `[0,W] × [0,H]`，除以**输出图片（含补边）的宽高**归一化。
`manifest.csv` 记录原始/输出尺寸和 `padding_ltrb`，便于追溯。
`coverage` 是框并集对面积过滤后 mask 的栅格覆盖率。背景减少仅反映框更紧，不能证明模型精度提升。
大段框可能比 V2 的细分框包含更多背景，这是减少碎框的几何取舍。

例如 `--split-angle 40` 会更晚切分，`--split-angle 25` 会更早切分。
V2.1 不使用 `--max-deviation`。应结合预览选定规则后固定参数。

旧 `--profile v2` 按分叉和曲线偏离切分（默认 8 px / 最短弧长 40 px），最小短边 2 px。
它沿用贴边分离及必要时退回水平框的旧逻辑，只供重现此前数据。

## 完整转换与检查

确认样本规则合适后运行：

```powershell
& .\prepare\.venv\Scripts\python.exe .\prepare\prepare_crack_obb_v2.py --full
& .\prepare\.venv\Scripts\python.exe .\prepare\test_crack_obb_v2.py
```

完整数据默认输出到新的 `data/crack_obb_auto_v21/`，继承独立输入 manifest 中的 source_group 和
train/val/test 划分，转换前检查同源照片不跨集合。脚本不会启动训练。
全量图片和标签始终全部输出；完整转换默认只为 36 个代表性样本保存对照图和 trace，
可通过 `--review-count 100` 增加为 100 张，避免每次产生几千张调试图。

生成后，训练的 `data` 应指向新目录下的 `data.yaml`；脚本不会自动改正在运行的训练。
V2.1 的对象定义是大段裂缝。应将两个待比较模型都在同一套标签上训练/验证；
不能把跨标签版本的 mAP 变化直接归因于模型改进。
