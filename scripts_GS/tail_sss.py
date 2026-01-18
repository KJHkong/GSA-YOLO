# /root/autodl-tmp/YOLOv8-Project-8.1/scripts_GS/test_sss.py
import os
import csv
import time
import torch
from ultralytics import YOLO

# -----------------------------
# 配置
# -----------------------------
YAML_CFG = "yolov8n.yaml"   # 原始结构
WEIGHTS = "runs/hixray_glsss/weights/pruned_head.pt"  # 剪枝后的 state_dict
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
yolo.model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model = yolo.model.to(DEVICE).eval()

# 打印模型参数/FLOPs
print("\n📊 Model Info:")
try:
    model.info(verbose=True, imgsz=IMG_SIZE)
except:
    print("⚠️ model.info() 不支持 FLOPs")

# -----------------------------
# 在测试集上验证
# -----------------------------
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

metrics = results.results_dict
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
