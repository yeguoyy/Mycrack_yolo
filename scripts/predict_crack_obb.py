import json
import random
from pathlib import Path

from ultralytics import YOLO
import torch


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    pt = ROOT / "runs/crack_yolo26n/weights/best.pt"
    model = YOLO(str(pt))
    print(model)

    test_folder = ROOT / "data/crack_obb_auto_v1/images/test"
    image_extensions = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    image_paths = [
        path
        for path in test_folder.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    ]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {test_folder}")

    sampled_images = random.sample(image_paths, min(30, len(image_paths)))
    print(f"随机抽查 {len(sampled_images)} 张图片进行预测")
    model.predict(
        source=[str(path) for path in sampled_images],
        save=True,
        conf=0.35,
        imgsz=640,
        device="cuda" if torch.cuda.is_available() else "cpu",
        project=str(ROOT / "output"),
        name="predict",
    )

    # metrics = model.val(
    #     data=str(ROOT / "data/crack_obb_auto_v1/data.yaml"),
    #     split="val",
    #     imgsz=416,
    #     # OBB 验证默认使用低置信度下限，保留计算 PR 曲线和 mAP 的候选框。
    #     conf=0.01,
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    #     workers=4,
    #     plots=True,
    #     project=str(ROOT / "output"),
    #     name="val",
    # )



if __name__ == "__main__":
    # Windows 数据加载子进程会重新导入本文件，仅主进程执行预测和验证。
    main()
