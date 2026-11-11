
import torch
import torch.nn as nn
import torch.nn.functional as F

# 多尺度适配器(Multi-Scale Adapter) 用于将 DINOv2 编码器输出的特征图调整为 FPN 所需的多尺度特征图
class MultiScaleAdapter(nn.Module):
    def __init__(self):
        super().__init__()

        # 三个独立的通道投影，需要在后续训练中学习参数
        self.proj3 = nn.Conv2d(384, 128, kernel_size=1)
        self.proj4 = nn.Conv2d(384, 128, kernel_size=1)
        self.proj5 = nn.Conv2d(384, 256, kernel_size=1)

    def forward(self, feature, input_size):
        # input_size 是原本送入主干的图片高宽
        height, width = input_size

        if height % 32 != 0 or width % 32 != 0:
            raise ValueError("当前适配器要求图片高宽能被32整除")

        # 双线性插值 按目标位置对邻近的特征值做加权平均以调整空间尺寸（高宽），align_corners=False表示不对齐角点
        p3 = F.interpolate(
            feature,
            size=(height // 8, width // 8),
            mode="bilinear",
            align_corners=False,
        )

        p4 = F.interpolate(
            feature,
            size=(height // 16, width // 16),
            mode="bilinear",
            align_corners=False,
        )

        p5 = F.interpolate(
            feature,
            size=(height // 32, width // 32),
            mode="bilinear",
            align_corners=False,
        )

        #调整通道数
        p3 = self.proj3(p3)
        p4 = self.proj4(p4)
        p5 = self.proj5(p5)

        return [p3, p4, p5]