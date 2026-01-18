import sys
import os

# 获取当前脚本所在目录并添加到系统路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from ultralytics import YOLO
import torch
import time
import os

# 模型路径
model_path = '/root/autodl-tmp/YOLOv8-Project-8.1/runs/hixray_gl/weights/best.pt'

# 检查模型文件是否存在
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found!")
    exit(1)

# 加载模型
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# 设置设备为GPU（若可用），否则回退到CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.model.to(device)
print(f"Using device: {device}")

# 1. 计算参数量 (parameters)
try:
    params = sum(p.numel() for p in model.model.parameters())
    print(f"Parameters: {params / 1e6:.2f} M")  # 以百万为单位显示
except Exception as e:
    print(f"Error calculating parameters: {e}")
    exit(1)

# 2. 计算FLOPs
try:
    input_size = (1, 3, 640, 640)  # YOLOv8n默认输入大小
    flops = model.profile(input_size=input_size, device=device)
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
except AttributeError:
    print("Ultralytics profile method not available, trying ptflops...")
    try:
        from ptflops import get_model_complexity_info
        flops, _ = get_model_complexity_info(model.model, (3, 640, 640), as_strings=False, print_per_layer_stat=False)
        print(f"FLOPs: {flops / 1e9:.2f} GMac ({2 * flops / 1e9:.2f} GFLOPs)")
    except ImportError:
        print("Error: ptflops not installed. Install with `pip install ptflops` or skip FLOPs calculation.")
        flops = None
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        flops = None

# 3. 计算FPS（在GPU上进行推理速度测试）
try:
    num_tests = 10  # 测试次数
    dummy_input = torch.randn(1, 3, 640, 640).to(device)  # 随机输入图像

    model.model.eval()
    with torch.no_grad():
        # 预热
        for _ in range(5):
            _ = model.model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        
        # 计时推理
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_tests):
            _ = model.model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        end_time = time.time()

    fps = num_tests / (end_time - start_time)
    print(f"FPS (on {device.type.upper()}): {fps:.2f}")
except Exception as e:
    print(f"Error calculating FPS: {e}")