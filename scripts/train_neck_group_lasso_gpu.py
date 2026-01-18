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

# -----------------------------
# 参数配置
# -----------------------------
DATA_YAML = "/root/autodl-tmp/YOLOv8-Project-8.1/HiXray/HiXray.yaml"
MODEL_PATH = "/root/autodl-tmp/YOLOv8-Project-8.1/runs/hixray_training/train_8.6_3090/weights/best.pt"

BATCH = 64
IMG_SIZE = 640
DEVICE = "cuda:0"
LR = 3e-4
EPOCHS = 10
LAMBDA_GL = 1e-3

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
train_loader = build_dataloader(train_dataset, batch=BATCH, workers=4, shuffle=True)

# -----------------------------
# 模型 & 优化器 & 损失
# -----------------------------
yolo = YOLO(MODEL_PATH)
model = yolo.model.to(DEVICE)

# 修复 model.args
if isinstance(model.args, dict):
    model.args = SimpleNamespace(**model.args)
if not hasattr(model.args, "box"): model.args.box = 7.5
if not hasattr(model.args, "cls"): model.args.cls = 0.5
if not hasattr(model.args, "dfl"): model.args.dfl = 1.5

# ✅ 强制用自定义损失
model.criterion = CustomV8DetectionLoss(model)

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

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

        total.backward()
        if i == 0:  # 每个 epoch 的第一个 batch 打印一次
            for name, param in model.named_parameters():
                if "model.15.conv.weight" in name:  # 👈 换成你Neck层的名字
                    print(f"[DEBUG][Epoch {epoch+1}] {name} grad mean={param.grad.mean().item() if param.grad is not None else None}")
                    print(f"[DEBUG][Epoch {epoch+1}] {name} weight mean(before step)={param.data.mean().item()}")
        optimizer.step()
        if i == 0:
            for name, param in model.named_parameters():
                if "model.15.conv.weight" in name:
                    print(f"[DEBUG][Epoch {epoch+1}] {name} weight mean(after step)={param.data.mean().item()}")

        det_loss_sum += loss.item()
        gl_sum += gl_penalty.item()
        total_sum += total.item()

        if (i + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Step {i+1}/{steps} | "
                  f"DetLoss={loss.item():.4f} | GL={gl_penalty.item():.6f} | Total={total.item():.4f}")

    # 计算平均 loss
    avg_det_loss = det_loss_sum / steps
    avg_gl_penalty = gl_sum / steps
    avg_total_loss = total_sum / steps

    # -------------------------
    # ✅ 验证（用 yolo.val()）
    # -------------------------
    # ⚠️ 确保 yolo.model 和训练过的 model 同步
    yolo.model = model
    results = yolo.val(
        data=DATA_YAML,
        batch=BATCH,
        imgsz=IMG_SIZE,
        device=DEVICE,
        plots=False,
        verbose=False
    )

    precision, recall, map50, map95 = results.box.mean_results()
    print(f"Epoch {epoch+1}: "
          f"P={precision:.4f}, R={recall:.4f}, "
          f"mAP50={map50:.4f}, mAP50-95={map95:.4f}")

    # -------------------------
    # 保存日志
    # -------------------------
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, steps,
                         avg_det_loss, avg_gl_penalty, avg_total_loss,
                         precision, recall, map50, map95])

    # -------------------------
    # 保存权重（仅 state_dict）
    # -------------------------
    last_path = os.path.join(WEIGHT_DIR, "last.pt")
    yolo.save(last_path) 

    if map95 > best_map:
        best_map = map95
        best_path = os.path.join(WEIGHT_DIR, "best.pt")
        yolo.save(best_path) 
        print(f"🌟 New best model saved at epoch {epoch+1} with mAP50-95={map95:.4f}")

print("🎯 训练完成！best.pt 和 last.pt 已保存。")
