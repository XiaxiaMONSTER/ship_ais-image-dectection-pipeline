from ultralytics import YOLO

# 加载你的训练好的模型或官方预训练模型
model = YOLO("/home/Tronlong/Desktop/rknn_model_zoo_and_models/yolov11n-day/5-best.pt")

# 获取所有预处理变换
transforms = model.transforms
print("完整预处理流程：")
print(transforms)

# 提取归一化参数
for transform in transforms.transforms:
    if hasattr(transform, 'mean') and hasattr(transform, 'std'):
        print(f"\n归一化参数：")
        print(f"mean: {transform.mean}")
        print(f"std: {transform.std}")