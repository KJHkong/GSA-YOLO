import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_FILE = "runs/hixray_gl/train_log.csv"
SAVE_DIR = "runs/hixray_gl/plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# 读取 CSV
df = pd.read_csv(LOG_FILE)

# -------------------------------
# 1. 训练 Loss 曲线
# -------------------------------
plt.figure(figsize=(8, 6))
plt.plot(df["epoch"], df["det_loss"], label="DetLoss")
plt.plot(df["epoch"], df["gl_penalty"], label="GL_penalty")
plt.plot(df["epoch"], df["total_loss"], label="TotalLoss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss with Group Lasso")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"), dpi=300)
plt.close()

# -------------------------------
# 2. 验证指标曲线
# -------------------------------
plt.figure(figsize=(8, 6))
plt.plot(df["epoch"], df["precision"], label="Precision")
plt.plot(df["epoch"], df["recall"], label="Recall")
plt.plot(df["epoch"], df["mAP50"], label="mAP50")
plt.plot(df["epoch"], df["mAP50_95"], label="mAP50-95")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("Validation Metrics")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, "val_metrics.png"), dpi=300)
plt.close()

print(f"✅ 可视化完成，曲线图已保存到 {SAVE_DIR}")
