# /root/autodl-tmp/YOLOv8-Project-8.1/scripts_GS/train_sss.py
import torch
import csv
import torch.optim as optim
import yaml
import os
from ultralytics import YOLO
from torch.optim.lr_scheduler import CosineAnnealingLR
from sss import sss_penalty  # 导入SSS正则化
from custom_loss import CustomV8DetectionLoss  # 导入自定义损失
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from types import SimpleNamespace

# -----------------------------
# 参数配置
# -----------------------------
MODEL_PATH = "/root/autodl-tmp/YOLOv8-Project-8.1/runs/hixray_gl/weights/best.pt"
DATA_YAML = "/root/autodl-tmp/YOLOv8-Project-8.1/HiXray/HiXray.yaml"
BATCH_SIZE = 64
IMG_SIZE = 640
DEVICE = "cuda:0"
EPOCHS = 85
SSS_LAMBDA = 1e-4
LR = 1e-4

SAVE_DIR = "runs/hixray_glsss"
WEIGHT_DIR = os.path.join(SAVE_DIR, "weights")
LOG_FILE = os.path.join(SAVE_DIR, "train_log.csv")
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
# 模型加载
# -----------------------------
yolo = YOLO(MODEL_PATH)
model = yolo.model.to(DEVICE)

if isinstance(model.args, dict):
    model.args = SimpleNamespace(**model.args)
if not hasattr(model.args, "box"): model.args.box = 7.5
if not hasattr(model.args, "cls"): model.args.cls = 0.5
if not hasattr(model.args, "dfl"): model.args.dfl = 1.5

# ✅ 自定义损失
model.criterion = CustomV8DetectionLoss(model)

# 解冻所有参数
for name, param in model.named_parameters():
    param.requires_grad = True

# 优化器 & 学习率调度器
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

# -----------------------------
# 初始化日志（覆盖写入）
# -----------------------------
with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "steps", "avg_det_loss", "avg_sss_penalty", "avg_total_loss",
                     "precision", "recall", "mAP50", "mAP50_95"])

# -----------------------------
# 训练循环
# -----------------------------
best_map = -1.0
print("🚀 开始 GPU 训练 SSS")

for epoch in range(EPOCHS):
    model.train()
    det_loss_sum, sss_sum, total_sum = 0.0, 0.0, 0.0
    steps = len(train_loader)

    for i, batch in enumerate(train_loader):
        imgs = batch["img"].to(DEVICE).float() / 255.0
        batch["img"] = imgs

        optimizer.zero_grad()
        preds = model(imgs)

        # 损失
        loss, _ = model.criterion(preds, batch)
        sss_penalty_value = sss_penalty(model, lambda_s=SSS_LAMBDA).to(DEVICE)
        total_loss = loss + sss_penalty_value

        # Debug 打印
        if i == 0:
            print(f"[DEBUG][Epoch {epoch+1} | Step {i+1}]")
            print(f"  loss.requires_grad={loss.requires_grad}, grad_fn={loss.grad_fn}")
            print(f"  sss_penalty.requires_grad={sss_penalty_value.requires_grad}, grad_fn={sss_penalty_value.grad_fn}")
            print(f"  total_loss.requires_grad={total_loss.requires_grad}, grad_fn={total_loss.grad_fn}")

        # ✅ 强制检查梯度
        assert total_loss.requires_grad, "❌ total_loss 没有梯度，请检查 custom_loss.py 或 sss.py"

        total_loss.backward()
        optimizer.step()

        # 记录
        det_loss_sum += loss.item()
        sss_sum += sss_penalty_value.item()
        total_sum += total_loss.item()

        if (i + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Step {i+1}/{steps} | "
                  f"DetLoss={loss.item():.4f} | SSS={sss_penalty_value.item():.6f} | Total={total_loss.item():.4f}")

    # 平均损失
    avg_det_loss = det_loss_sum / steps
    avg_sss_penalty = sss_sum / steps
    avg_total_loss = total_sum / steps

    # -------------------------
    # 验证（安全方式）
    # -------------------------
    print("验证中...")
    tmp_path = os.path.join(SAVE_DIR, "tmp_val.pt")
    torch.save(model.state_dict(), tmp_path)

    tmp = YOLO(MODEL_PATH)
    tmp.model.load_state_dict(torch.load(tmp_path, map_location=DEVICE))
    tmp = tmp.to(DEVICE)

    results = tmp.val(data=DATA_YAML, split="val", batch=BATCH_SIZE,
                      imgsz=IMG_SIZE, device=DEVICE, verbose=False)
    precision, recall, map50, map95 = results.box.mean_results()
    del tmp

    print(f"Epoch {epoch+1} Val Results: P={precision:.6f}, R={recall:.6f}, "
          f"mAP50={map50:.6f}, mAP50-95={map95:.6f}")

    # -------------------------
    # 保存日志（追加）
    # -------------------------
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, steps, avg_det_loss, avg_sss_penalty,
                         avg_total_loss, precision, recall, map50, map95])

    # -------------------------
    # 保存权重
    # -------------------------
    last_path = os.path.join(WEIGHT_DIR, "last.pt")
    best_path = os.path.join(WEIGHT_DIR, "best.pt")
    yolo.model = model
    yolo.save(last_path)

    if map95 > best_map:
        best_map = map95
        yolo.save(best_path)
        print(f"🌟 New best model saved at epoch {epoch+1} with mAP50-95={map95:.4f}")

    scheduler.step()
    print(f"[DEBUG] Epoch {epoch+1} LR={scheduler.get_last_lr()[0]:.6f}")

print("🎯 训练完成！best.pt 和 last.pt 已保存。")
