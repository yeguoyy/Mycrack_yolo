from pathlib import Path

from ultralytics import YOLO
import torch
import PIL

# 优先使用仓库中的预训练权重；本地缺失时使用 Ultralytics 的默认查找方式。
weights = Path(__file__).resolve().parents[1] / "weights" / "yolo26n-obb.pt"
model = YOLO(str(weights) if weights.is_file() else weights.name)
print(model)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)

# img = PIL.Image.open("D:\\Desktop\\OIP-C (2).webp")
# model.predict(source= img , save=True, device= device)

for result in model.predict(

source=0,

show=True,

stream=True,

conf=0.35,

imgsz=416,

device= device

):
    pass