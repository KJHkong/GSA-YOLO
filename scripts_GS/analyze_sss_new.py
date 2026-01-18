# /root/autodl-tmp/YOLOv8-Project-8.1/scripts_GS/analyze_sss.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from sss import get_head_convs  # 复用之前写的函数

# -----------------------------
# 配置
# -----------------------------
WEIGHTS = "runs/hixray_glsss/weights/best.pt"   # 建议用微调后的权重
DEVICE = "cuda:0"
SAVE_DIR = "runs/hixray_glsss/plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# 加载模型
# -----------------------------
print(f"🔍 Loading model from {WEIGHTS}")
yolo = YOLO(WEIGHTS)
model = yolo.model.to(DEVICE)

# -----------------------------
# 提取 Head 卷积
# -----------------------------
head_convs = get_head_convs(model)
print(f"✅ Found {len(head_convs)} Head Conv layers")

# -----------------------------
# 统计稀疏性
# -----------------------------
summary = []
threshold = 1e-3  # 小于这个值视为“接近零”

for name, module in head_convs:
    w = module.weight.data
    norms = w.view(w.size(0), -1).norm(1, dim=1).cpu().numpy()

    mean_norm = np.mean(norms)
    min_norm = np.min(norms)
    max_norm = np.max(norms)
    zero_like = np.sum(norms < threshold)
    ratio = zero_like / len(norms)

    # 提取分支类型和索引
    if "cv2" in name:
        layer_type = "cv2"
    elif "cv3" in name:
        layer_type = "cv3"
    else:
        layer_type = "other"
    layer_idx = -1
    for tok in name.replace("]", "").split("["):
        if tok.isdigit():
            layer_idx = int(tok)
            break

    summary.append({
        "layer": name,
        "layer_type": layer_type,
        "layer_idx": layer_idx,
        "channels": len(norms),
        "mean_norm": mean_norm,
        "min_norm": min_norm,
        "max_norm": max_norm,
        "near_zero_ratio": ratio,
        "sparsity_score": f"{ratio*100:.2f}%"
    })

    # 保存直方图
    plt.figure(figsize=(6, 4))
    plt.hist(norms, bins=30, color="tomato", alpha=0.7)
    plt.title(f"SSS sparsity distribution - {name}")
    plt.xlabel("Channel L1 norm")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.6)
    fname = os.path.join(SAVE_DIR, f"sss_sparsity_{name.replace('.', '_')}.png")
    plt.savefig(fname, dpi=300)
    plt.close()

    # 保存折线图
    plt.figure(figsize=(7, 4))
    plt.plot(range(len(norms)), np.sort(norms), marker="o", markersize=2)
    plt.title(f"Channel-wise L1 norm (sorted) - {name}")
    plt.xlabel("Channel index (sorted)")
    plt.ylabel("L1 norm")
    plt.grid(True, linestyle="--", alpha=0.6)
    fname_line = os.path.join(SAVE_DIR, f"sss_line_{name.replace('.', '_')}.png")
    plt.savefig(fname_line, dpi=300)
    plt.close()

print(f"📊 Histograms & line plots saved in {SAVE_DIR}")

# -----------------------------
# 保存汇总表
# -----------------------------
df = pd.DataFrame(summary)
csv_path = os.path.join(SAVE_DIR, "sss_sparsity_summary.csv")
df.to_csv(csv_path, index=False)

print(f"✅ SSS Sparsity summary saved to {csv_path}")
print(df.head())

# -----------------------------
# 生成分支级别平均 sparsity 对比图
# -----------------------------
pivot = df.groupby(["layer_type", "layer_idx"])["near_zero_ratio"].mean().reset_index()

plt.figure(figsize=(7, 5))
for ltype in ["cv2", "cv3"]:
    sub = pivot[pivot["layer_type"] == ltype]
    plt.plot(sub["layer_idx"], sub["near_zero_ratio"], marker="o", label=ltype)
plt.xlabel("Branch index")
plt.ylabel("Near-zero ratio")
plt.title("Branch-wise sparsity comparison (cv2 vs cv3)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig(os.path.join(SAVE_DIR, "sss_branch_sparsity.png"), dpi=300)
plt.close()

print(f"📊 Branch-level sparsity plot saved in {SAVE_DIR}")
