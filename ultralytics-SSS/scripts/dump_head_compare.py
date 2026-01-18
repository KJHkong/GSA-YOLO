#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从两份权重（before/after）对比 YOLOv8 SSS 头部：
- cv2[i], cv3[i] 的 conv/bn/lambda_ 尺寸
导出 Markdown 和 CSV。
"""
import os, csv, argparse, sys, torch
sys.path.insert(0, "/root/autodl-tmp/YOLOv8-Project-8.1/ultralytics-SSS")  # 按你的安装路径修改
from ultralytics import YOLO
import torch.nn as nn

def _get_head(model: nn.Module):
    # 通常 head 在 model.model[-1]
    m = getattr(model, "model", None)
    if isinstance(m, nn.Sequential) and len(m) > 0:
        return m[-1]
    # 兜底：直接找含有 cv2/cv3 的模块
    for name, mod in model.named_modules():
        if hasattr(mod, "cv2") and hasattr(mod, "cv3"):
            return mod
    raise RuntimeError("Head not found (no cv2/cv3).")

def _row_from_branch(br_name: str, blk: nn.Module):
    # blk 形如 head.cv2[i] 或 head.cv3[i]，通常是 nn.Sequential 或 Conv 包装
    # 我们尽量找到：(conv, bn, lambda_)
    conv_out = conv_in = bn_out = lam_len = None

    # 支持 sequential: 可能是 [UConv, ScaleLayer] 或 [UConv]
    if isinstance(blk, nn.Sequential):
        for m in blk:
            # UConv 或 Conv 包装
            if hasattr(m, "conv") and isinstance(m.conv, nn.Conv2d):
                conv_out = m.conv.out_channels
                conv_in  = m.conv.in_channels
                # BN
                if hasattr(m, "bn") and isinstance(m.bn, nn.BatchNorm2d):
                    bn_out = m.bn.num_features
            # ScaleLayer(lambda_)
            if hasattr(m, "lambda_") and isinstance(m.lambda_, torch.nn.Parameter):
                lam_len = m.lambda_.numel()
    else:
        # 非 sequential（极少见）
        if hasattr(blk, "conv") and isinstance(blk.conv, nn.Conv2d):
            conv_out = blk.conv.out_channels
            conv_in  = blk.conv.in_channels
        if hasattr(blk, "bn") and isinstance(blk.bn, nn.BatchNorm2d):
            bn_out = blk.bn.num_features
        if hasattr(blk, "lambda_") and isinstance(blk.lambda_, torch.nn.Parameter):
            lam_len = blk.lambda_.numel()

    return [br_name, conv_out, conv_in, bn_out, lam_len]

def _collect_head_info(weight_path: str):
    y = YOLO(weight_path)
    model = y.model.eval()
    head = _get_head(model)
    rows = []
    for br in ["cv2", "cv3"]:
        seq = getattr(head, br, None)
        if seq is None:
            continue
        # seq 可能是 nn.ModuleList
        for i, mod in enumerate(seq):
            br_name = f"{br}[{i}]"
            rows.append(_row_from_branch(br_name, mod))
    return rows

def _to_dict(rows):
    # {branch: {key:val}}
    D = {}
    for r in rows:
        br, co, ci, bo, ll = r
        D[br] = {"conv_out": co, "conv_in": ci, "bn_out": bo, "lambda_len": ll}
    return D

def main(before, after, outdir):
    os.makedirs(outdir, exist_ok=True)

    rows_b = _collect_head_info(before)
    rows_a = _collect_head_info(after)

    Db = _to_dict(rows_b)
    Da = _to_dict(rows_a)

    # 汇合所有出现过的分支
    branches = sorted(set(list(Db.keys()) + list(Da.keys())),
                      key=lambda s: (0 if s.startswith("cv2") else 1, int(s[s.find("[")+1:s.find("]")])))

    # 导出 CSV/MD
    csv_path = os.path.join(outdir, "head_compare.csv")
    md_path  = os.path.join(outdir, "head_compare.md")

    with open(csv_path, "w", newline="") as cf, open(md_path, "w") as mf:
        cw = csv.writer(cf)
        cw.writerow(["branch",
                     "before.conv_out","after.conv_out",
                     "before.conv_in","after.conv_in",
                     "before.bn_out","after.bn_out",
                     "before.lambda","after.lambda"])
        mf.write("| Branch | conv_out (B→A) | conv_in (B→A) | bn_out (B→A) | lambda_len (B→A) |\n")
        mf.write("|---|---:|---:|---:|---:|\n")

        for br in branches:
            B = Db.get(br, {})
            A = Da.get(br, {})
            coB, coA = B.get("conv_out"), A.get("conv_out")
            ciB, ciA = B.get("conv_in"),  A.get("conv_in")
            boB, boA = B.get("bn_out"),   A.get("bn_out")
            llB, llA = B.get("lambda_len"), A.get("lambda_len")

            cw.writerow([br, coB, coA, ciB, ciA, boB, boA, llB, llA])
            mf.write(f"| {br} | {coB}→{coA} | {ciB}→{ciA} | {boB}→{boA} | {llB}→{llA} |\n")

    print(f"[OK] CSV: {csv_path}")
    print(f"[OK] MD : {md_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="剪枝前权重路径")
    ap.add_argument("--after",  required=True, help="剪枝后权重路径")
    ap.add_argument("--outdir", required=True, help="输出目录")
    args = ap.parse_args()
    main(args.before, args.after, args.outdir)
