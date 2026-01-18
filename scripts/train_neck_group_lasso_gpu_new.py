import os
import csv
import torch
import torch.optim as optim
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from types import SimpleNamespace

from group_lasso import group_lasso_penalty
from custom_loss import CustomV8DetectionLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
# -----------------------------
# 参数配置
# -----------------------------
DATA_YAML = "/root/autodl-tmp/YOLOv8-Project-8.1/HiXray/HiXray.yaml"
#MODEL_PATH = "/root/autodl-tmp/YOLOv8-Project-8.1/runs/hixray_training/train_8.6_3090/weights/best.pt"
MODEL_PATH="/root/autodl-tmp/YOLOv8-Project-8.1/runs/hixray_gl/weights/best.pt"
BATCH = 64
IMG_SIZE = 640
DEVICE = "cuda:0"
LR = 1e-4
EPOCHS = 60     
LAMBDA_GL = 3e-3

SAVE_DIR = "runs/hixray_gl"
WEIGHT_DIR = os.path.join(SAVE_DIR, "weights")
LOG_FILE = os.path.join(SAVE_DIR, "train_log.csv")
os.makedirs(WEIGHT_DIR, exist_ok=True)

# -----------------------------
# 数据加载
# -----------------------------
with open(DATA_YAML, "r") as f:
    data_cfg = yaml.safe_load(f)

train_cfg = get_cfg(overrides={
    "imgsz": IMG_SIZE,
    "batch": BATCH,
    "device": DEVICE
})

train_dataset = build_yolo_dataset(cfg=train_cfg, img_path=data_cfg["train"],
                                   batch=BATCH, data=data_cfg, mode="train")
val_dataset   = build_yolo_dataset(cfg=train_cfg, img_path=data_cfg["val"],
                                   batch=BATCH, data=data_cfg, mode="val")

train_loader = build_dataloader(train_dataset, batch=BATCH, workers=4, shuffle=True)
val_loader   = build_dataloader(val_dataset, batch=BATCH, workers=4, shuffle=False)

# -----------------------------
# 模型 & 优化器 & 损失
# -----------------------------
yolo = YOLO(MODEL_PATH)
model = yolo.model.to(DEVICE)

# 🔓 强制解冻所有参数
for name, param in model.named_parameters():
    param.requires_grad = True

# 修复 model.args
if isinstance(model.args, dict):
    model.args = SimpleNamespace(**model.args)
if not hasattr(model.args, "box"): model.args.box = 7.5
if not hasattr(model.args, "cls"): model.args.cls = 0.5
if not hasattr(model.args, "dfl"): model.args.dfl = 1.5

# ✅ 强制用自定义损失
model.criterion = CustomV8DetectionLoss(model)

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
# -----------------------------
# 初始化日志
# -----------------------------
with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch", "steps", "avg_det_loss", "avg_gl_penalty", "avg_total_loss",
        "precision", "recall", "mAP50", "mAP50_95"
    ])

# -----------------------------
# 训练循环
# -----------------------------
print("🚀 开始 GPU 训练 Group Lasso")
best_map = -1.0

for epoch in range(EPOCHS):
    model.train()
    det_loss_sum, gl_sum, total_sum = 0.0, 0.0, 0.0
    steps = len(train_loader)

    for i, batch in enumerate(train_loader):
        imgs = batch["img"].to(DEVICE).float() / 255.0
        batch["img"] = imgs

        optimizer.zero_grad()
        preds = model(imgs)

        # ✅ 损失
        loss, _ = model.criterion(preds, batch)
        gl_penalty = group_lasso_penalty(model, lambda_gl=LAMBDA_GL).to(DEVICE)
        total = loss + gl_penalty

        # 🔍 DEBUG 打印
        if i == 0:
            print(f"[DEBUG][Epoch {epoch+1} | Step {i+1}]")
            print(f"  loss.requires_grad={loss.requires_grad}, grad_fn={loss.grad_fn}")
            print(f"  gl_penalty.requires_grad={gl_penalty.requires_grad}, grad_fn={gl_penalty.grad_fn}")
            print(f"  total.requires_grad={total.requires_grad}, grad_fn={total.grad_fn}")
            for name, param in model.named_parameters():
                if "model.15.cv1.conv.weight" in name:  # 👈 确认真实层名
                    print(f"  param.requires_grad={param.requires_grad}")

        total.backward()
        if i == 0:  # 每个 epoch 的第一个 batch 打印一次权重梯度
            for name, param in model.named_parameters():
                if "model.15.cv1.conv.weight" in name:
                    print(f"[CHECK][Epoch {epoch+1}] grad mean={param.grad.mean().item() if param.grad is not None else None}")
        optimizer.step()

        gl_sum += gl_penalty.item()
        total_sum += total.item()
        det_loss_sum += loss.item()

        if (i + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Step {i+1}/{steps} | "
                  f"DetLoss={loss.item():.4f} | GL={gl_penalty.item():.6f} | Total={total.item():.4f}")

    # 计算平均 loss
    avg_det_loss = det_loss_sum / steps
    avg_gl_penalty = gl_sum / steps
    avg_total_loss = total_sum / steps

    # -------------------------
    # ✅ 验证（用官方接口）
    # -------------------------
    print("验证中...")
    tmp_path = os.path.join(SAVE_DIR, "tmp_val.pt")
    torch.save(model.state_dict(), tmp_path)

    tmp = YOLO(MODEL_PATH)   # 先用原始结构初始化（比如 best.pt）
    tmp.model.load_state_dict(torch.load(tmp_path, map_location=DEVICE))
    tmp = tmp.to(DEVICE)

    results = tmp.val(
        data=DATA_YAML,
        split="val",
        batch=BATCH,
        imgsz=IMG_SIZE,
        device=DEVICE,
        verbose=False
    )
    precision, recall, map50, map95 = results.box.mean_results()
    del tmp

    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = True
    print(f"Epoch {epoch+1} Val Results: "
          f"P={precision:.6f}, R={recall:.6f}, "
          f"mAP50={map50:.6f}, mAP50-95={map95:.6f}")

    # -------------------------
    # 保存日志
    # -------------------------
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, steps,
                         avg_det_loss, avg_gl_penalty, avg_total_loss,
                         precision, recall, map50, map95])

    # -------------------------
    # 保存权重（保证可用 YOLO() 加载）
    # -------------------------
    last_path = os.path.join(WEIGHT_DIR, "last.pt")
    yolo.model = model  # 确保同步
    yolo.save(last_path)

    if map95 > best_map:
        best_map = map95
        best_path = os.path.join(WEIGHT_DIR, "best.pt")
        yolo.save(best_path)
        print(f"🌟 New best model saved at epoch {epoch+1} with mAP50-95={map95:.4f}")
    scheduler.step()
    print(f"[DEBUG] Epoch {epoch+1} LR={scheduler.get_last_lr()[0]:.6f}")
print("🎯 训练完成！best.pt 和 last.pt 已保存。")
