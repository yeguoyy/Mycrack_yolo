"""验证接入 DINOv2 主干的裂缝 OBB 模型，结果保存到 output/val_dinov2/。"""

import sys
from pathlib import Path

import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from ultralytics.utils.files import increment_path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from dinov2.dinov2_backbone import DINOv2Pyramid


# 注册自定义模块，与 DINOv2 训练入口保持一致。
tasks.DINOv2Pyramid = DINOv2Pyramid


def main() -> None:
    # best.pt 已包含训练后的主干、适配器、Neck 和 Head 参数。
    # 验证时直接加载完整模型，不再单独加载 DINOv2 的预训练参数。
    pt = ROOT / "runs/crack_yolo26n_dinov2_obb/weights/best.pt"
    if not pt.is_file():
        raise FileNotFoundError(f"DINOv2 OBB 权重不存在：{pt}")
    model = YOLO(str(pt), task="obb")

    # 每次验证使用独立目录，图表和指标文件保存在同一处。
    # 目录已存在时自动选择 val_dinov2_2、val_dinov2_3 等新名称。
    save_dir = increment_path(ROOT / "output" / "val_dinov2", sep="_")

    metrics = model.val(
        data=str(ROOT / "data/crack_obb_auto_v1/data.yaml"),
        split="val",
        imgsz=448,
        batch=8,  # 与当前 DINOv2 训练的批次大小一致。
        # 与原版验证脚本一致，保留计算 PR 曲线和 mAP 的候选框。
        conf=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
        workers=4,
        plots=True,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,  # 上面已选择新目录，避免框架再次追加编号。
    )

    # 在本次验证目录保存总体指标，并记录使用的权重路径。
    report = (
        f"Weights: {pt}\n"
        f"Precision(P): {metrics.box.mp:.6f}\n"
        f"Recall(R): {metrics.box.mr:.6f}\n"
        f"mAP50: {metrics.box.map50:.6f}\n"
        f"mAP50-95: {metrics.box.map:.6f}\n"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "val_metrics.txt").write_text(report, encoding="utf-8")

    # 保存各类别指标；使用 UTF-8 BOM，便于 Excel 识别中文。
    (save_dir / "val_metrics.csv").write_text(
        metrics.to_csv(), encoding="utf-8-sig"
    )

    print(report)
    print(f"验证结果保存至：{save_dir}")


if __name__ == "__main__":
    # Windows 数据加载使用多进程，需要保留主入口保护。
    main()
