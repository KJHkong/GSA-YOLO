import os
from ultralytics import YOLO

# -----------------------------
# 配置
# -----------------------------
MODEL_CFG = "yolov8n.yaml"  # ✅ 模型结构
WEIGHTS = "/root/autodl-tmp/YOLOv8-Project-8.1/runs/hixray_gl/weights/best.pt"
DATA_YAML = "/root/autodl-tmp/YOLOv8-Project-8.1/HiXray/HiXray.yaml"
DEVICE = 0
IMG_SIZE = 640
BATCH = 64
SAVE_DIR = "runs/hixray_testing"
EXP_NAME = "test_gl"

# -----------------------------
# 加载模型（结构 + 权重）
# -----------------------------
print(f"🔍 Loading model from {WEIGHTS}")
# model = YOLO(MODEL_CFG)
# model.load(WEIGHTS)
model = YOLO(WEIGHTS)
# -----------------------------
# 在测试集上验证
# -----------------------------
results = model.val(
    data=DATA_YAML,
    split="test",        # ✅ 指定测试集
    device=DEVICE,
    imgsz=IMG_SIZE,
    batch=BATCH,
    save=True,           # 保存预测可视化
    save_json=True,      # 保存 COCO 格式 JSON
    project=SAVE_DIR,
    name=EXP_NAME,
    verbose=True
)

# -----------------------------
# 打印结果
# -----------------------------
precision, recall, map50, map95 = results.box.mean_results()
print(f"\n✅ Final Test Results:")
print(f"Precision={precision:.4f}, Recall={recall:.4f}, "
      f"mAP50={map50:.4f}, mAP50-95={map95:.4f}")
