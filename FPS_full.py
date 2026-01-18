import sys
import os

# 获取当前脚本所在目录并添加到系统路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
    
import torch
import time
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression

def measure_fps(model_path, img_size=640, conf_thres=0.25, iou_thres=0.45, num_tests=100):
    # 1. 加载模型
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # 2. 移动到 GPU 并设为评估模式
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.model.eval()
    print(f"Using device: {device}")

    # 3. 准备随机输入
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    # 4. 热身阶段 (Warm-up)
    print("Warming up...")
    with torch.no_grad():
        for _ in range(20):
            _ = model.model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # 5. 测试 Inference FPS (仅前向传播)
    print("Testing Inference FPS...")
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_tests):
            _ = model.model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        
    inference_time = (end_time - start_time) / num_tests
    inference_fps = 1 / inference_time

    # 6. 测试 Full Pipeline FPS (前向传播 + NMS)
    print("Testing Full Pipeline FPS (including NMS)...")
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_tests):
            # 第一步：推理
            preds = model.model(dummy_input)
            # 第二步：NMS 后处理
            _ = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

    full_pipeline_time = (end_time - start_time) / num_tests
    full_pipeline_fps = 1 / full_pipeline_time

    # 7. 打印结果
    print("\n" + "="*30)
    print(f"Results for model: {model_path}")
    print(f"Input size: {img_size}x{img_size}")
    print(f"Inference FPS: {inference_fps:.2f}")
    print(f"Full Pipeline FPS: {full_pipeline_fps:.2f}")
    print(f"Drop Ratio: {(1 - full_pipeline_fps/inference_fps)*100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    # 请修改为你的模型路径
    MODEL_PATH = '/root/autodl-tmp/YOLOv8-Project-8.1/runs/hixray_gl/weights/best.pt'
    measure_fps(MODEL_PATH)