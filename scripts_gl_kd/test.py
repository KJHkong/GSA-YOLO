import os
from ultralytics import YOLO

# -----------------------------
# 配置
# -----------------------------
MODEL_CFG = "yolov8n.yaml"  # ✅ 模型结构
WEIGHTS = "/root/autodl-tmp/YOLOv8-Project-8.1/runs/hixray_gl_kd/weights/best.pt"  # 学生模型权重
TEACHER_WEIGHTS = "/root/autodl-tmp/YOLOv8-Project-8.1/runs/hixray_training/train_8.6_3090/weights/best.pt"  # 教师模型
DATA_YAML = "/root/autodl-tmp/YOLOv8-Project-8.1/HiXray/HiXray.yaml"
DEVICE = 0
IMG_SIZE = 640
BATCH = 64
SAVE_DIR = "runs/hixray_testing_kd"
EXP_NAME = "test_kd"

# -----------------------------
# 加载模型（学生和教师）
# -----------------------------
print(f"🔍 Loading student model from {WEIGHTS}")
student_model = YOLO(WEIGHTS).model.to(DEVICE)
teacher_model = YOLO(TEACHER_WEIGHTS).model.to(DEVICE).eval()

# -----------------------------
# 在测试集上验证
# -----------------------------
results = student_model.val(
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
# 打印结果
# -----------------------------
precision, recall, map50, map95 = results.box.mean_results()
print(f"\n✅ Final Test Results:")
print(f"Precision={precision:.4f}, Recall={recall:.4f}, "
      f"mAP50={map50:.4f}, mAP50-95={map95:.4f}")
