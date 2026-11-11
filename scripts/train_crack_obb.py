"""使用预训练 Ultralytics OBB 模型微调墙面裂缝检测器。

运行前请先检查 data/crack_obb_auto_v1/review/ 中的自动标签预览，
尤其是弯曲、分叉裂缝和空标签图片。当前标签是由像素 mask 转出的草稿，
未经审核不要把训练指标当成可靠实验结果。

从项目根目录运行：
    python scripts/train_crack_obb.py

请使用 pytorch_cu126 环境。仅保留 --weights 和 --resume 两个命令行参数，
其余训练配置直接在 main() 中修改。
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


root = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """只保留指定初始化权重和恢复训练这两个常用选项。"""
    parser = argparse.ArgumentParser(description="微调墙面裂缝 OBB 检测模型。")
    parser.add_argument(
        "--weights",
        default=None,
        help="初始化权重路径；未指定时优先使用当前实验的 best.pt，否则使用预训练权重。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从当前实验的 last.pt 恢复训练及优化器状态；优先于 --weights。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 数据集配置中定义 train、val、test 路径及裂缝类别。
    data = root / "data" / "crack_obb_auto_v21" / "data.yaml"
    project = root / "runs"  # 训练日志、图表和新权重统一保存在这里。
    # 实验名也用于寻找 best/last；独立的448实验可改为 crack_yolo26n_448。
    name = "crack_yolo26n_448_v21"

    record_file = project / f"{name}_latest.txt"
    current_name = name
    if record_file.is_file():
        current_name = record_file.read_text(encoding="utf-8").strip() or name

    current_dir = project / current_name
    last_checkpoint = current_dir / "weights" / "last.pt"
    best_checkpoint = current_dir / "weights" / "best.pt"

    # 恢复训练 > 显式指定权重 > 当前实验 best.pt > 原始预训练权重。
    # 从原始权重建立新基线时，显式传入 --weights weights/yolo26n-obb.pt。
    if args.resume:
        weights = last_checkpoint
    elif args.weights is not None:
        weights = Path(args.weights).expanduser()
        if not weights.is_absolute():
            weights = root / weights  # 相对权重路径以 yolo/ 目录为基准。
        weights = weights.resolve()
    else:
        weights = best_checkpoint if best_checkpoint.is_file() else root / "weights" / "yolo26n-obb.pt"

    # 提前检查路径，避免加载模型后才发现输入文件缺失。
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")

    # 从选定的权重加载模型；本脚本要求使用 OBB（旋转框）模型。
    model = YOLO(str(weights))
    if model.task != "obb":
        raise ValueError(f"Expected an OBB checkpoint, got task={model.task!r}: {weights}")

    print(f"Resume training:        {args.resume}")
    print(f"OBB weights/checkpoint:  {weights}")
    print(f"Dataset config:         {data}")
    print(f"Training output base:   {project / name}")

    def remember_run(trainer):
        record_file.write_text(trainer.save_dir.name, encoding="utf-8")

    model.add_callback("on_model_save", remember_run)

    # 常用训练配置直接在这里修改；训练期间默认会执行验证。
    # resume=True 用于续接中断的训练，多数配置以检查点记录为准。
    model.train(
        data=str(data),
        epochs=100,          # 最多训练轮数；可能因早停提前结束。
        imgsz=448,          # 输入边长；后续统一448实验时改为448，并修改上面的实验名。
        batch=8,            # 每批图片数；显存不足时可改为4或2。
        device="0",         # 第0张 NVIDIA GPU；也可改为 "cpu"。
        workers=4,          # 数据加载子进程数；Windows多进程报错时可改为0。
        patience=30,        # 验证指标连续30轮未改善时早停；0表示关闭早停。
        project=str(project),
        name=name,          # 独立实验请使用新名称；目录已存在时会自动追加序号。
        seed=0,             # 固定随机种子，便于复现实验。
        resume=args.resume,  # 恢复 last.pt；与从 best.pt 开启新微调不同。
    )


if __name__ == "__main__":
    main()
