import torch
import torch.optim as optim
import yaml
import os
import csv
from ultralytics import YOLO
from torch.optim.lr_scheduler import CosineAnnealingLR
from custom_loss import CustomV8DetectionLoss
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from types import SimpleNamespace

# -----------------------------
# 配置
# -----------------------------
MODEL_PATH = "runs/hixray_glsss/weights/ft_best.pt"  # ← 新剪枝后的模型
DATA_YAML = "/root/autodl-tmp/YOLOv8-Project-8.1/HiXray/HiXray.yaml"
BATCH_SIZE = 64
IMG_SIZE = 640
DEVICE = "cuda:0"
EPOCHS = 35   # 微调30个epoch
LR = 0.002

SAVE_DIR = "runs/hixray_glsss"
WEIGHT_DIR = os.path.join(SAVE_DIR, "weights")
LOG_FILE = os.path.join(SAVE_DIR, "finetune_log.csv")
os.makedirs(WEIGHT_DIR, exist_ok=True)

# -----------------------------
# 数据加载
# -----------------------------
with open(DATA_YAML, "r") as f:
    data_cfg = yaml.safe_load(f)

train_cfg = get_cfg(overrides={"imgsz": IMG_SIZE, "batch": BATCH_SIZE, "device": DEVICE})
train_dataset = build_yolo_dataset(cfg=train_cfg, img_path=data_cfg["train"], batch=BATCH_SIZE, data=data_cfg, mode="train")
val_dataset = build_yolo_dataset(cfg=train_cfg, img_path=data_cfg["val"], batch=BATCH_SIZE, data=data_cfg, mode="val")

train_loader = build_dataloader(train_dataset, batch=BATCH_SIZE, workers=4, shuffle=True)
val_loader = build_dataloader(val_dataset, batch=BATCH_SIZE, workers=4, shuffle=False)

# -----------------------------
# 加载剪枝后的模型
# -----------------------------
print(f"🔍 Loading model from {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# 手动加载模型，检查是否包含 'model' 键
if 'model' in checkpoint:
    model = checkpoint['model'].to(DEVICE)  # 从 checkpoint 中提取 model
    print(f"✅ Loaded model from checkpoint.")
else:
    print(f"⚠️ 'model' not found in checkpoint. Initializing model weights from scratch.")
    # 使用原始配置文件加载 YOLO 模型
    yolo = YOLO("yolov8n.yaml")  # 请确认 yolo 配置文件路径是否正确
    model = yolo.model.to(DEVICE)  # 加载模型结构

if isinstance(model.args, dict):
    model.args = SimpleNamespace(**model.args)
if not hasattr(model.args, "box"): model.args.box = 7.5
if not hasattr(model.args, "cls"): model.args.cls = 0.5
if not hasattr(model.args, "dfl"): model.args.dfl = 1.5

model.criterion = CustomV8DetectionLoss(model)

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

# 使用 CosineAnnealingLR 进行学习率衰减
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "steps", "avg_loss", "precision", "recall", "mAP50", "mAP50_95"])

print("🚀 开始微调训练")
best_map = -1.0

for epoch in range(EPOCHS):
    model.train()
    loss_sum = 0.0
    steps = len(train_loader)

    for i, batch in enumerate(train_loader):
        imgs = batch["img"].to(DEVICE).float() / 255.0
        batch["img"] = imgs

        optimizer.zero_grad()
        preds = model(imgs)
        loss, _ = model.criterion(preds, batch)

        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

        if (i + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Step {i+1}/{steps} | Loss={loss.item():.4f}")

    avg_loss = loss_sum / steps
    print(f"Epoch {epoch+1} | Avg Loss={avg_loss:.4f}")

    tmp_path = os.path.join(WEIGHT_DIR, "tmp_val.pt")
    torch.save(model.state_dict(), tmp_path)

    tmp = YOLO("yolov8n.yaml")  # 同结构
    tmp.model.load_state_dict(torch.load(tmp_path, map_location=DEVICE))
    tmp = tmp.to(DEVICE)

    results = tmp.val(data=DATA_YAML, split="val", batch=BATCH_SIZE, imgsz=IMG_SIZE, device=DEVICE, verbose=False)
    precision, recall, map50, map95 = results.box.mean_results()
    del tmp

    print(f"Epoch {epoch+1} Val: P={precision:.4f}, R={recall:.4f}, mAP50={map50:.4f}, mAP50-95={map95:.4f}")

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, steps, avg_loss, precision, recall, map50, map95])

    last_path = os.path.join(WEIGHT_DIR, "ft_last.pt")
    best_path = os.path.join(WEIGHT_DIR, "ft_best.pt")
    yolo.save(last_path)
    if map95 > best_map:
        best_map = map95
        yolo.save(best_path)
        print(f"🌟 New best model saved at epoch {epoch+1} with mAP50-95={map95:.4f}")

    # 更新学习率
    scheduler.step()

print("🎯 微调完成！")
