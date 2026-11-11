import torch
import torch.nn as nn
import timm
from torchvision import transforms as T
import torch.nn.functional as F
from .MultiScaleAdapter import MultiScaleAdapter
from pathlib import Path
from timm.models import load_checkpoint

class DINOv2Backbone(nn.Module):
    def __init__(self):
        """只创建网络结构；预训练参数由 DINOv2Trainer 按需加载。"""
        super().__init__()

        # 注册为子模块：PyTorch 会自动管理它的参数
        self.encoder = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
        )
        self.normalize=T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )


    def load_pretrained(self):
        """供训练器在没有完整模型权重时，加载 DINOv2 编码器预训练参数。"""
        weights_path = (Path(__file__).resolve().parents[1]/ "weights"/ "dinov2_s14"/ "pytorch_model.bin" )
        # 只向 DINOv2 编码器加载参数
        # strict=True 要求参数名称、数量和形状匹配
        result = load_checkpoint(
            self.encoder,
            str(weights_path),
            strict=True,
        )

        print("DINOv2预训练参数加载结果：", result)

    def forward(self, x):
        # 约定输入为 RGB、浮点数，数值范围为 [0, 1]
        batch, _, height, width = x.shape

        if height % 14 != 0 or width % 14 != 0:
            raise ValueError("当前版本要求输入高宽能被14整除")

        x = self.normalize(x)

        # [B, 1 + 图像token数量（1024）, 384]
        tokens = self.encoder.forward_features(x)  #编码器，经过transformer生成了32x32=1024个含自注意力语义信息的token（+1个CLS token），384是token的特征维度

        # 去掉 CLS 等前缀 token，保留图像位置 [B,1024, 384]
        patch_tokens = tokens[:, self.encoder.num_prefix_tokens:, :]

        # [B,1024,384] → [B,384,1024] → [B,384,H/14=32,W/14=32]  转换为原先的空间结构384是通道，32是高，32是宽
        feature_map = patch_tokens.transpose(1, 2).reshape(
            batch,
            self.encoder.num_features,
            height // 14,
            width // 14,
        )

        return feature_map


class DINOv2Pyramid(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = DINOv2Backbone()
        self.adapter = MultiScaleAdapter()

    def forward(self, x):
        # 记录原始输入尺寸，适配器最终按这个尺寸生成三尺度
        original_size = x.shape[-2:]
        height, width = original_size

        # 向上调整为14的倍数
        target_height = (height + 13) // 14 * 14
        target_width = (width + 13) // 14 * 14
        #只要输入尺寸不是14的倍数，就会进行插值调整
        if (target_height, target_width) != original_size:
            x = F.interpolate(
                x,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )

        feature = self.backbone(x)

        return self.adapter(
            feature,
            input_size=original_size,
        )
