#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 prune_report.json 生成两个表：
- Markdown: prune_table.md
- CSV:      prune_table.csv
并打印整体 head 通道稀疏率
"""
import os, json, argparse, csv
from collections import defaultdict

def main(report_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(report_path, "r") as f:
        rep = json.load(f)

    # 兼容：items 里每条为一个 branch 记录（你目前的格式）
    rows = []
    total_old, total_pruned = 0, 0

    for it in rep.get("items", []):
        # 预期键：branch, old_out, new_out, tau, idx(可选)
        branch = it.get("branch")  # 形如 "cv2[0]" 或 "cv3[2]"
        old_out = int(it.get("old_out", 0))
        new_out = int(it.get("new_out", 0))
        tau = it.get("tau", None)
        pruned = old_out - new_out

        rows.append([branch, old_out, new_out, pruned, tau])

        total_old += old_out
        total_pruned += pruned

    # 排序，让 cv2[0], cv2[1], cv2[2], cv3[0]... 更好看
    def _key(r):
        b = r[0]
        if b is None:
            return (999, 999)
        # b 例子: "cv2[1]"
        try:
            name, idx = b.split("[")
            idx = int(idx.rstrip("]"))
        except Exception:
            name, idx = b, 999
        order = 0 if "cv2" in name else 1
        return (order, idx)

    rows.sort(key=_key)

    # 生成 Markdown
    md_path = os.path.join(outdir, "prune_table.md")
    with open(md_path, "w") as f:
        f.write("| Branch | Old Out | New Out | Pruned | Tau |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |\n")

        rate = (total_pruned / max(1, total_old)) * 100.0
        f.write(f"\n**Head 通道稀疏率**：{rate:.2f}%  ({total_pruned}/{total_old})\n")

    # 生成 CSV
    csv_path = os.path.join(outdir, "prune_table.csv")
    with open(csv_path, "w", newline="") as cf:
        cw = csv.writer(cf)
        cw.writerow(["branch", "old_out", "new_out", "pruned", "tau"])
        for r in rows:
            cw.writerow(r)

    print(f"[OK] Markdown: {md_path}")
    print(f"[OK] CSV     : {csv_path}")
    print(f"Head 通道稀疏率: { (total_pruned/max(1,total_old))*100:.2f}%  ({total_pruned}/{total_old})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="path to prune_report.json")
    ap.add_argument("--outdir", required=True, help="where to save tables")
    args = ap.parse_args()
    main(args.report, args.outdir)
