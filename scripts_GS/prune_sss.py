# /root/autodl-tmp/YOLOv8-Project-8.1/scripts_GS/prune_sss.py
import os
import json
import csv
import torch
import torch.nn as nn
from ultralytics import YOLO

# ======================
# 配置
# ======================
WEIGHTS_IN  = "runs/hixray_glsss/weights/best.pt"
WEIGHTS_OUT = "runs/hixray_glsss/weights/pruned_head.pt"
REPORT_JSON = "runs/hixray_glsss/weights/prune_report.json"
REPORT_CSV  = "runs/hixray_glsss/weights/prune_table.csv"
DEVICE = "cuda:0"
PRUNE_RATIO = 0.25
MIN_KEEP    = 16

os.makedirs(os.path.dirname(WEIGHTS_OUT), exist_ok=True)

# ======================
# 工具函数
# ======================
def prune_out_channels(conv: nn.Conv2d, keep_idx: torch.Tensor):
    w = conv.weight.data.index_select(0, keep_idx)
    conv.out_channels = w.size(0)
    conv.weight = nn.Parameter(w)
    if conv.bias is not None:
        conv.bias = nn.Parameter(conv.bias.data.index_select(0, keep_idx))

def keep_from_conv(conv: nn.Conv2d, ratio: float):
    out_c = conv.weight.shape[0]
    n_keep = max(MIN_KEEP, int(out_c * (1.0 - ratio)))
    n_keep = min(n_keep, out_c)
    norms = conv.weight.view(out_c, -1).norm(1, dim=1)
    _, idx = torch.topk(norms, k=n_keep, largest=True)
    return idx.sort()[0], out_c, n_keep

# ======================
# 主逻辑
# ======================
if __name__ == "__main__":
    print(f"🔍 Loading model from {WEIGHTS_IN}")
    yolo = YOLO(WEIGHTS_IN)
    model = yolo.model.to(DEVICE).eval()

    detect = None
    for n, m in model.named_modules():
        if m.__class__.__name__.lower() == "detect":
            detect = m
            break
    assert detect is not None, "未找到 Detect 头部模块！"

    report = {"branches": []}
    table_rows = []

    with torch.no_grad():
        for i, (cv2_block, cv3_block) in enumerate(zip(detect.cv2, detect.cv3)):
            cv2_last = list(cv2_block.modules())[-1]
            cv3_last = list(cv3_block.modules())[-1]
            if not isinstance(cv2_last, nn.Conv2d) or not isinstance(cv3_last, nn.Conv2d):
                print(f"⚠️ branch[{i}] 未找到末端 Conv，跳过。")
                continue

            cv2_keep, cv2_old, cv2_new = keep_from_conv(cv2_last, PRUNE_RATIO)
            cv3_keep, cv3_old, cv3_new = keep_from_conv(cv3_last, PRUNE_RATIO)

            prune_out_channels(cv2_last, cv2_keep)
            prune_out_channels(cv3_last, cv3_keep)

            print(f"✂️ branch[{i}]: cv2_penult out: {cv2_old}->{cv2_new} | "
                  f"cv3_penult out: {cv3_old}->{cv3_new} | proj_out kept: (reg=64, cls=8)")

            report["branches"].append({
                "idx": i,
                "cv2_out_old": int(cv2_old), "cv2_out_new": int(cv2_new),
                "cv3_out_old": int(cv3_old), "cv3_out_new": int(cv3_new)
            })
            table_rows.append([f"branch{i}_cv2", cv2_old, cv2_new, cv2_old - cv2_new])
            table_rows.append([f"branch{i}_cv3", cv3_old, cv3_new, cv3_old - cv3_new])

    # 保存权重（state_dict）
    torch.save(model.state_dict(), WEIGHTS_OUT)
    print(f"✅ Pruned model weights saved: {WEIGHTS_OUT}")

    # 保存报告 JSON
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"📊 Report saved: {REPORT_JSON}")

    # 保存报告 CSV
    with open(REPORT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["branch", "old_out", "new_out", "pruned"])
        writer.writerows(table_rows)
    print(f"📊 Table saved: {REPORT_CSV}")
