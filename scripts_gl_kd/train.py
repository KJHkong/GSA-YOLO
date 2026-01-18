# scripts_gl_kd/train_kd.py
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from types import SimpleNamespace
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.utils.loss import v8DetectionLoss
from torch.optim.lr_scheduler import CosineAnnealingLR

# =============================
# 关键：允许安全反序列化（方法1）
# =============================
import torch.nn as tnn
import ultralytics.nn.tasks as utasks

# 尽量多放一些常见模块进 allowlist，减少反复报错
torch.serialization.add_safe_globals([
    utasks.DetectionModel,
    tnn.modules.container.Sequential,
    tnn.Conv2d, tnn.BatchNorm2d, tnn.SiLU, tnn.ReLU, tnn.LeakyReLU,
    tnn.Upsample, tnn.MaxPool2d, tnn.ModuleList, tnn.Identity, tnn.Dropout,
])

# =============================
# KD 损失（分类蒸馏 + 检测损失）
# =============================
class KDLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5):
        """
        temperature: 蒸馏温度
        alpha: KD与检测损失的权重 (总损失 = alpha*KD + (1-alpha)*det)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, s_logits, t_logits, det_loss):
        T = self.temperature
        kd = self.kl(
            nn.functional.log_softmax(s_logits / T, dim=1),
            nn.functional.softmax(t_logits / T, dim=1)
        ) * (T * T)
        return self.alpha * kd + (1 - self.alpha) * det_loss

def extract_cls_logits(preds):
    """
    从 YOLOv8 前向输出中抽取分类 logits 用于蒸馏。
    YOLOv8 的每个预测行是 [x,y,w,h, cls1, cls2, ...]，所以 cls 从索引 4 开始（不是5!）
    preds 可能是张量或 (preds, aux) 形式的元组/列表，取 preds[0] 即主输出。
    返回形状：(B*N, num_classes)
    """
    if isinstance(preds, (list, tuple)):
        p = preds[0]
    else:
        p = preds
    # 期望形状 (B, N, 4+nc)
    if p.dim() == 3:
        cls = p[..., 4:]
        return cls.reshape(-1, cls.shape[-1])
    elif p.dim() == 2:
        return p[:, 4:]
    else:
        raise RuntimeError(f"Unexpected preds shape: {tuple(p.shape)}")

# =============================
# 配置
# =============================
DATA_YAML = "/root/autodl-tmp/YOLOv8-Project-8.1/HiXray/HiXray.yaml"

# 学生 = 你做完 GL 的 best.pt
STUDENT_WEIGHTS = "/root/autodl-tmp/YOLOv8-Project-8.1/runs/hixray_gl/weights/best.pt"
# 教师 = baseline best.pt
TEACHER_WEIGHTS = "/root/autodl-tmp/YOLOv8-Project-8.1/runs/hixray_training/train_8.6_3090/weights/best.pt"

DEVICE = "cuda:0"
BATCH = 64
IMG_SIZE = 640
LR = 1e-4
EPOCHS = 60
SAVE_DIR = "runs/hixray_gl_kd"
WEIGHT_DIR = os.path.join(SAVE_DIR, "weights")
LOG_FILE = os.path.join(SAVE_DIR, "train_log.csv")
os.makedirs(WEIGHT_DIR, exist_ok=True)

# =============================
# 数据
# =============================
with open(DATA_YAML, "r") as f:
    data_cfg = yaml.safe_load(f)

train_cfg = get_cfg(overrides={"imgsz": IMG_SIZE, "batch": BATCH, "device": DEVICE})
train_dataset = build_yolo_dataset(cfg=train_cfg, img_path=data_cfg["train"],
                                   batch=BATCH, data=data_cfg, mode="train")
val_dataset   = build_yolo_dataset(cfg=train_cfg, img_path=data_cfg["val"],
                                   batch=BATCH, data=data_cfg, mode="val")

train_loader = build_dataloader(train_dataset, batch=BATCH, workers=4, shuffle=True)
val_loader   = build_dataloader(val_dataset, batch=BATCH, workers=4, shuffle=False)

# =============================
# 模型：用 yaml 初始化结构，再手动加载 state_dict
# =============================
student_yolo = YOLO("yolov8n.yaml")
student = student_yolo.model.to(DEVICE)

# —— 安全方式加载 GL 后的权重（只取权重）——
ckpt = torch.load(STUDENT_WEIGHTS, map_location="cpu", weights_only=True)
state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
student.load_state_dict(state_dict, strict=False)

# 教师模型直接用 YOLO 加载（baseline 一般不带自定义模块，稳定）
teacher = YOLO(TEACHER_WEIGHTS).model.to(DEVICE).eval()

# 检测损失（官方）
criterion = v8DetectionLoss(student, tal_topk=10)

# 优化器 & 学习率
optimizer = optim.SGD(student.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

# KD loss
kd_loss_fn = KDLoss(temperature=2.0, alpha=0.5)

# =============================
# 日志
# =============================
with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "steps", "avg_total_loss", "precision", "recall", "mAP50", "mAP50_95"])

# =============================
# 训练
# =============================
best_map = -1.0

for epoch in range(EPOCHS):
    student.train()
    total_loss_sum = 0.0
    steps = len(train_loader)

    for i, batch in enumerate(train_loader):
        imgs = batch["img"].to(DEVICE).float() / 255.0
        batch["img"] = imgs

        optimizer.zero_grad()

        # 学生预测
        s_preds = student(imgs)
        det_loss, _ = criterion(s_preds, batch)

        # 教师预测（只前向，不反传）
        with torch.no_grad():
            t_preds = teacher(imgs)

        # 提取分类 logits 做 KD（cls 从索引 4 开始）
        s_logits = extract_cls_logits(s_preds)
        t_logits = extract_cls_logits(t_preds)

        # 组合损失
        loss = kd_loss_fn(s_logits, t_logits, det_loss)
        loss.backward()
        optimizer.step()
        total_loss_sum += loss.item()

        if (i + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Step {i+1}/{steps} | "
                  f"KD+Det Loss={loss.item():.4f}")

    avg_total_loss = total_loss_sum / steps

    # ========== 验证 ==========
    # 用 YOLO 对象跑 val：把 student 的 state_dict 暂存后再读入 YOLO 类
    tmp_sd_path = os.path.join(SAVE_DIR, "tmp_student_sd.pt")
    torch.save(student.state_dict(), tmp_sd_path)

    tmp = YOLO("yolov8n.yaml")
    tmp.model.load_state_dict(torch.load(tmp_sd_path, map_location=DEVICE))
    _ = tmp.to(DEVICE)  # 让内部模型在 DEVICE
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

    # 记录日志
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, steps, avg_total_loss, precision, recall, map50, map95])

    # 保存 best
    if map95 > best_map:
        best_map = map95
        # 保存为纯 state_dict 的 ckpt，测试时更容易加载
        torch.save({"model": student.state_dict()}, os.path.join(WEIGHT_DIR, "best.pt"))
        print(f"🌟 New best @ epoch {epoch+1}: mAP50-95={map95:.4f}")

    scheduler.step()
    print(f"[Epoch {epoch+1}] avg_total_loss={avg_total_loss:.4f} | P={precision:.4f} R={recall:.4f} mAP50={map50:.4f} mAP50-95={map95:.4f}")

print("🎯 KD 训练完成！best.pt 已保存到 runs/hixray_gl_kd/weights/")
