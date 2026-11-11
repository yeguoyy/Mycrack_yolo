# 普通 YOLO26n-OBB 与 DINOv2-OBB：V2.1 测试集对比

2026 年 11 月 11 日。两组均使用 V2.1 数据、448 输入和训练 batch=8。普通版在第 54 轮取得最佳模型、第 84 轮早停；DINOv2 版对应第 162 和 192 轮，均连续 30 轮未刷新最佳 mAP50–95。

## 同一测试集上的检测结果

对三份 `best.pt` 使用相同的 V2.1 test 集：822 张图片、1,890 个目标，输入 448×448，FP32、batch=8，保留置信度至少 0.01 的候选框计算 PR 和 mAP。Precision 与 Recall 为 Ultralytics 在各模型 F1 最佳点报告的值，不是固定 0.25 阈值下的结果。

| 模型权重 | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| 普通版，本次再微调早停 | 66.95% | 61.96% | 63.90% | 31.92% |
| DINOv2 版，本次早停 | 69.96% | 62.33% | 67.21% | 34.02% |
| 普通版，上一次原始 100 轮训练（参考） | 71.39% | 59.15% | 66.74% | 34.42% |

DINOv2 相比本次普通版的 mAP50 和 mAP50–95 分别高 3.31、2.10 个百分点；相比更强的上一轮普通版，分别高 0.48、低 0.39 个百分点。两种结构的最佳已观察结果接近，优势取决于评价指标，现有结果不支持 DINOv2 稳定提高综合定位表现。

## 预测延迟

RTX 4060 Laptop GPU，PyTorch 2.13.0+cu126，Ultralytics 8.4.178，FP32、448×448、batch=1、预测置信度阈值 0.25。每模型预热 20 次，使用同样 64 张已加载到内存的测试图交替预测两遍，每模型 128 次计时。端到端指从内存图片到预测结果，包括预处理、GPU 传输、推理和后处理，不含读盘、画框、显示和保存。

| 指标 | 普通版，本次早停 | DINOv2 版 |
| --- | ---: | ---: |
| 端到端平均延迟 | 15.65 ms | 25.31 ms |
| 端到端中位延迟 | 14.48 ms | 24.91 ms |
| 端到端 P95 | 22.48 ms | 27.83 ms |
| 模型推理阶段中位延迟 | 10.68 ms | 20.53 ms |
| 模型参数量（融合前） | 2.65 M | 23.54 M |
| best.pt 大小 | 5.68 MB | 47.49 MB |

DINOv2 的端到端中位延迟为普通版的 1.72 倍，模型推理阶段为 1.92 倍。未对上一轮普通版参考权重单独进行 batch=1 计时。延迟数据适用于本机及上述推理配置。

## 训练口径与选型

| 训练记录 | 普通版，本次早停 | DINOv2 版 |
| --- | ---: | ---: |
| 验证最佳 mAP50 / mAP50–95 | 65.064% / 33.511% | 64.891% / 34.318% |
| 最后轮 mAP50 / mAP50–95 | 63.068% / 32.828% | 64.079% / 33.422% |
| 单轮训练耗时中位数 | 54.72 秒 | 119.70 秒 |
| 实际优化器 / 基础学习率 | AdamW / 0.002 | MuSGD / 0.01 |

普通版从上一轮训练好的 `best.pt` 新开微调，DINOv2 从预训练主干和兼容的 YOLO 后续层开始训练，两组训练上限和学习率调度也不同。因此，这次测试比较的是现有可部署权重，不能把全部差值单独归因于主干结构。

默认部署建议优先考虑上一轮普通版权重：在本测试集上 mAP50–95 略高，普通结构的实测单图延迟和参数量明显更低。若业务对漏检尤其敏感，DINOv2 可保留为备选，进一步比较同等精确率下的召回率再选择工作阈值。

- [本次普通版训练日志](D:/Code/wall_damage_identification_model/yolo/runs/crack_yolo26n_448_v21-2/results.csv)
- [本次 DINOv2 训练日志](D:/Code/wall_damage_identification_model/yolo/runs/crack_yolo26n_dinov2_obb_v21/results.csv)
- [上一轮普通版 best.pt](D:/Code/wall_damage_identification_model/yolo/runs/crack_yolo26n_448_v21/weights/best.pt)
- [实测原始数据及每次计时](D:/Code/wall_damage_identification_model/yolo/reports/yolo_vs_dinov2_v21/benchmark.json)
- [复测脚本](D:/Code/wall_damage_identification_model/yolo/reports/yolo_vs_dinov2_v21/benchmark.py)
