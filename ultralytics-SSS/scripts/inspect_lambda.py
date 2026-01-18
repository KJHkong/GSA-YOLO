import argparse, torch, numpy as np
from ultralytics import YOLO
from ultralytics.nn.modules.sss import ScaleLayer
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument('--weights', type=str, required=True)
ap.add_argument('--tau', type=float, default=1e-3)
args = ap.parse_args()

m = YOLO(args.weights).model.eval()
total, below = 0, 0
print(f"[Inspect] tau={args.tau}")
print("layer_name, C, sparse%, mean|λ|, min|λ|, max|λ|, smallest_idx10")

for name, mod in m.named_modules():
    if isinstance(mod, nn.Sequential) and len(mod)==2 and isinstance(mod[1], ScaleLayer):
        lam = mod[1].lambda_.detach().abs().cpu().numpy()
        if lam.size == 0: 
            continue
        c = lam.size
        mask = (lam < args.tau)
        sratio = mask.mean() * 100.0
        total += c
        below += int(mask.sum())
        smallest_idx = np.argsort(lam)[:10].tolist()
        print(f"{name}, {c}, {sratio:.2f}%, {lam.mean():.6f}, {lam.min():.6f}, {lam.max():.6f}, {smallest_idx}")

overall = (below / total * 100.0) if total > 0 else 0.0
print(f"[Inspect] overall sparse ratio (<tau): {below}/{total} = {overall:.2f}%")
