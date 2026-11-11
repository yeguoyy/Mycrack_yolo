import sys
from pathlib import Path

from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from ultralytics.models.yolo.obb.train import OBBTrainer

root = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(root))

from dinov2.dinov2_backbone import DINOv2Pyramid

# 注册到 YOLO 模型解析器所在的模块
tasks.DINOv2Pyramid = DINOv2Pyramid


# 继承 OBBTrainer，重写 get_model 方法
class DINOv2Trainer(OBBTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        # 按配置构建完整模型。
        # 如果传入了完整模型权重，父类会在这里加载。
        model = super().get_model(
            cfg=cfg,
            weights=weights,
            verbose=verbose,
        )

        # 未提供完整模型权重时，分别加载 DINOv2 主干和 YOLO 后续层的预训练权重。
        if weights is None:
            model.model[0].backbone.load_pretrained()
            # YOLO 后续层使用官方 OBB 预训练权重。
            source = YOLO(
                str(root / "weights" / "yolo26n-obb.pt")
            ).model

            transferred = 0
            # 当前结构中，目标第 4～16 层对应官方模型第 11～23 层。
            for target_index in range(4, 17):
                target_layer = model.model[target_index]  # 获取目标层模块。
                # 按相差 7 的层号获取官方对应层的状态字典；修改结构后需重新核对。
                source_state = source.model[target_index + 7].state_dict()
                target_state = target_layer.state_dict()  # 包含参数及 BN 运行统计等缓冲区。

                compatible = {# 保留匹配的状态张量
                    key: value
                    for key, value in source_state.items()
                    if key in target_state  # 对应层内的状态名称相同。
                       and value.shape == target_state[key].shape  # 状态张量的形状相同。
                }

                # 形状不匹配的分类输出层张量保留目标模型的初始化。
                target_layer.load_state_dict(compatible, strict=False)  # 复制兼容的预训练状态。
                transferred += len(compatible)  # 累计迁移的状态张量数，包含参数和缓冲区。

            print(f"YOLO 后续层已迁移 {transferred} 个状态张量")

        return model

if __name__ == "__main__":

    data = root / "data" / "crack_obb_auto_v21" / "data.yaml"
    project = root / "runs"  # 训练日志、图表和新权重统一保存在这里。
    name = "crack_yolo26n_dinov2_obb_v21"


    model_yaml = root / "structure" / "yolo26n-dinov2-obb.yaml"
    last_checkpoint = project / name / "weights" / "last.pt"
    best_checkpoint = project / name / "weights" / "best.pt"

    if best_checkpoint.is_file():
        print(f"使用已有模型权重：{best_checkpoint}")
        model = YOLO(str(best_checkpoint), task="obb")
    else:
        print("新建模型；训练建模时加载DINOv2预训练参数。")
        model = YOLO(str(model_yaml), task="obb") #从yaml文件创建模型

    model.train(
        trainer=DINOv2Trainer,  # 使用上面定义的初始化逻辑
        data=str(data),
        epochs=300,
        imgsz=448,
        batch=8,
        device="0",
        workers=4,  # 数据加载子进程数；Windows多进程报错时可改为0。
        patience=30,  # 验证指标连续30轮未改善时早停；0表示关闭早停。
        project=str(project),
        name=name,
        resume=False,          # 从选中的权重开启一轮新的微调
        deterministic=False    #允许 PyTorch 使用不保证每次结果完全一致的 GPU 算法
    )



