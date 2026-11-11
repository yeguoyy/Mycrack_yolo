import json
import random
from pathlib import Path

from ultralytics import YOLO
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.utils.plotting import plot_images
import torch


ROOT = Path(__file__).resolve().parents[1]


class CrackOBBValidator(OBBValidator):
    """显式使用像素坐标绘制 GT，避免长对角框的归一化坐标被误判。"""

    def plot_val_samples(self, batch, ni):
        # 旋转框的长边可以大于图片宽度，归一化 xywhr 的 w 因此可能大于 1.1。
        # plot_images 用这个阈值猜测坐标单位；这里仅对绘图副本显式反归一化。
        labels = dict(batch)
        labels["img"] = batch["img"].clone()
        labels["bboxes"] = batch["bboxes"].clone()
        h, w = batch["img"].shape[2:]
        labels["bboxes"][..., :4] *= labels["bboxes"].new_tensor([w, h, w, h])
        plot_images(
            labels=labels,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )


def main() -> None:
    data = "crack_obb_auto_v21"
    pt_name = "crack_yolo26n_448_v21"

    pt = ROOT / "runs"/ pt_name /"weights/best.pt"
    model = YOLO(str(pt))

    metrics = model.val(
        validator=CrackOBBValidator,
        data=str(ROOT / "data"/data/"data.yaml"),
        split="val",
        imgsz=448,
        # OBB 验证默认使用低置信度下限，保留计算 PR 曲线和 mAP 的候选框。
        conf=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
        workers=4,
        plots=True,
        project=str(ROOT / "output"),
        name="val-"+pt_name,
    )
    # 保存总体指标
    report = (
        f"Precision(P): {metrics.box.mp:.6f}\n"
        f"Recall(R): {metrics.box.mr:.6f}\n"
        f"mAP50: {metrics.box.map50:.6f}\n"
        f"mAP50-95: {metrics.box.map:.6f}\n"
    )

    save_dir = ROOT / "output"
    save_dir.mkdir(parents=True, exist_ok=True)

    (save_dir / "val_metrics.txt").write_text(
        report, encoding="utf-8"
    )

    # 保存各类别的指标，可以用 Excel 打开
    (save_dir / "val_metrics.csv").write_text(
        metrics.to_csv(), encoding="utf-8-sig"
    )

    print(report)


if __name__ == "__main__":
    main()
