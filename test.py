import torch
import timm
from timm.models import load_checkpoint

# 1. 创建模型结构
backbone = timm.create_model(
    "vit_small_patch14_dinov2.lvd142m",
    pretrained=False,          # 本次从本地加载，不自动下载
    num_classes=0,             # 不使用分类头
    dynamic_img_size=True,     # 允许调整输入尺寸
)

# 2. 加载本地预训练参数：这里改成你的实际路径
load_checkpoint(
    backbone,
    r"D:\Code\wall_damage_identification_model\yolo\weights\dinov2_s14\pytorch_model.bin",
    strict=True,
)

backbone.eval()

# 3. 用零张量检查输出形状
x = torch.zeros(1, 3, 448, 448)

with torch.inference_mode():
    tokens = backbone.forward_features(x)

print(tokens.shape)