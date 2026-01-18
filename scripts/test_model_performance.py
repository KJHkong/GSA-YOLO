import torch
from ultralytics import YOLO
import time
from thop import profile  # 用于计算FLOPs和参数量

# 加载模型
MODEL_PATH = "/root/autodl-tmp/YOLOv8-Project-8.1/runs/hixray_gl/weights/best.pt"
model = YOLO(MODEL_PATH).model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 测试用的 dummy data (batch=1 和 batch=64)
img_size = 640
dummy_input_batch1 = torch.randn(1, 3, img_size, img_size).to(device)  # batch size 1
dummy_input_batch64 = torch.randn(64, 3, img_size, img_size).to(device)  # batch size 64

# 测量模型推理时间（FPS）计算
def measure_fps(batch_input):
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):  # 重复测试100次来平均性能
            model(batch_input)
    end_time = time.time()
    avg_time = (end_time - start_time) / 100
    fps = 1 / avg_time  # FPS = 1 / 平均推理时间
    return fps

# 测量 FPS
fps_batch1 = measure_fps(dummy_input_batch1)
fps_batch64 = measure_fps(dummy_input_batch64)
print(f"FPS for batch=1: {fps_batch1:.2f}")
print(f"FPS for batch=64: {fps_batch64:.2f}")

# 计算 FLOPs 和 参数量
flops, params = profile(model, inputs=(dummy_input_batch1,))  # 只使用batch=1计算，因为FLOPs与batch大小无关
print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
print(f"Parameters: {params / 1e6:.2f} M")

