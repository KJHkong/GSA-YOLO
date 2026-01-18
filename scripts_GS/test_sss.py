import os
import csv
import torch
from ultralytics import YOLO
import time
import torch.serialization
import ultralytics  # 确保导入 ultralytics

# -----------------------------
# 配置
# -----------------------------
YAML_CFG = "yolov8n.yaml"   # 原始结构
WEIGHTS = "runs/hixray_glsss/weights/ft_best.pt"  # 微调后的模型
DATA_YAML = "/root/autodl-tmp/YOLOv8-Project-8.1/HiXray/HiXray.yaml"
DEVICE = "cuda:0"
IMG_SIZE = 640
BATCH = 64
SAVE_DIR = "runs/hixray_glsss"
EXP_NAME = "test_glsss"
LOG_FILE = os.path.join(SAVE_DIR, "test_log.csv")

os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# 加载模型
# -----------------------------
print(f"🔍 Loading pruned model from {WEIGHTS}")
yolo = YOLO(YAML_CFG)

# 添加 DetectionModel 到安全全局类列表
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

# 进行模型加载
checkpoint = torch.load(WEIGHTS, map_location=DEVICE, weights_only=False)

# 手动加载剪枝后的权重
model = yolo.model.to(DEVICE).eval()
model_dict = model.state_dict()

# 遍历 checkpoint 权重，逐层加载
for name, param in checkpoint.items():
    if name in model_dict and param.size() == model_dict[name].size():
        model_dict[name].copy_(param)
    else:
        print(f"⚠️ Skipping layer: {name} due to mismatch or shape difference.")
        # 手动初始化被跳过的层
        if "cv2" in name or "cv3" in name:
            print(f"🔧 Initializing {name} with default weights.")
            if "weight" in name:
                model_dict[name].data.fill_(0)  # 手动将权重初始化为零
            elif "bias" in name:
                model_dict[name].data.fill_(0)  # 手动将偏置初始化为零

# -----------------------------
# 打印模型参数/FLOPs
# -----------------------------
print("\n📊 Model Info:")
try:
    model.info(verbose=True, imgsz=IMG_SIZE)
except Exception as e:
    print(f"⚠️ model.info() 不支持 FLOPs，错误信息: {e}")

# -----------------------------
# 在测试集上验证
# -----------------------------
print(f"🔍 Validating on test set...")
results = yolo.val(
    data=DATA_YAML,
    split="test",
    device=DEVICE,
    imgsz=IMG_SIZE,
    batch=BATCH,
    save=True,
    save_json=True,
    project=SAVE_DIR,
    name=EXP_NAME,
    verbose=True
)

# -----------------------------
# 调试：打印 DetMetrics 对象的内容
# -----------------------------
print(f"Results object type: {type(results)}")
print(f"Results object content: {dir(results)}")  # 打印 results 对象的所有属性
print(f"Results summary: {results.summary()}")  # 打印每个类别的总结
print(f"Results stats: {results.stats}")  # 打印详细的统计信息

# -----------------------------
# 打印结果
# -----------------------------
metrics = results.results_dict
print(f"Metrics dictionary: {metrics}")  # 打印 metrics 字典内容

precision = metrics.get("metrics/precision(B)", 0)
recall = metrics.get("metrics/recall(B)", 0)
map50 = metrics.get("metrics/mAP50(B)", 0)
map95 = metrics.get("metrics/mAP50-95(B)", 0)

print(f"\n✅ Final Test Results:")
print(f"Precision={precision:.4f}, Recall={recall:.4f}, "
      f"mAP50={map50:.4f}, mAP50-95={map95:.4f}")

# -----------------------------
# 计算推理 FPS (batch=1 和 batch=64)
# -----------------------------
def benchmark_fps(batch_size, n_runs=50, n_warmup=10):
    dummy = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
        start = time.time()
        for _ in range(n_runs):
            _ = model(dummy)
        end = time.time()
    fps = (n_runs * batch_size) / (end - start)
    return fps

fps_b1 = benchmark_fps(1)
fps_b64 = benchmark_fps(64)

print(f"⚡ Inference speed @batch=1:  {fps_b1:.2f} FPS")
print(f"⚡ Inference speed @batch=64: {fps_b64:.2f} FPS")

# -----------------------------
# 保存结果到 CSV
# -----------------------------
header = ["precision", "recall", "mAP50", "mAP50-95", "FPS_batch1", "FPS_batch64"]
file_exists = os.path.isfile(LOG_FILE)

with open(LOG_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(header)
    writer.writerow([precision, recall, map50, map95, fps_b1, fps_b64])

print(f"📊 Test results saved to {LOG_FILE}")
