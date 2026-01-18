import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

from group_lasso import get_neck_convs  # 复用你之前写的函数

# -----------------------------
# 配置
# -----------------------------
WEIGHTS = "runs/hixray_gl/weights/best.pt"   # 或者换成 last.pt
DEVICE = "cuda:0"
SAVE_DIR = "runs/hixray_gl/plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# 加载模型
# -----------------------------
print(f"🔍 Loading model from {WEIGHTS}")
yolo = YOLO(WEIGHTS)
model = yolo.model.to(DEVICE)

# -----------------------------
# 提取 Neck 卷积
# -----------------------------
neck_convs = get_neck_convs(model)
print(f"✅ Found {len(neck_convs)} Neck Conv layers")

# -----------------------------
# 统计稀疏性
# -----------------------------
summary = []
threshold = 1e-3  # 小于这个值视为“接近零”

for name, module in neck_convs:
    w = module.weight.data
    norms = w.view(w.size(0), -1).norm(2, dim=1).cpu().numpy()

    # 统计信息
    mean_norm = np.mean(norms)
    min_norm = np.min(norms)
    max_norm = np.max(norms)
    zero_like = np.sum(norms < threshold)
    ratio = zero_like / len(norms)

    summary.append({
        "layer": name,
        "channels": len(norms),
        "mean_norm": mean_norm,
        "min_norm": min_norm,
        "max_norm": max_norm,
        "near_zero_ratio": ratio
    })

    # 保存直方图
    plt.figure(figsize=(6, 4))
    plt.hist(norms, bins=30, color="steelblue", alpha=0.7)
    plt.title(f"Sparsity distribution - {name}")
    plt.xlabel("Channel L2 norm")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.6)
    fname = os.path.join(SAVE_DIR, f"sparsity_{name.replace('.', '_')}.png")
    plt.savefig(fname, dpi=300)
    plt.close()

print(f"📊 Histograms saved in {SAVE_DIR}")

# -----------------------------
# 保存汇总表
# -----------------------------
df = pd.DataFrame(summary)
csv_path = os.path.join(SAVE_DIR, "sparsity_summary.csv")
df.to_csv(csv_path, index=False)

print(f"✅ Sparsity summary saved to {csv_path}")
print(df.head())
