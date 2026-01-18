#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
import numpy as np
import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.nn.modules.sss import ScaleLayer
from ultralytics.nn.modules.conv import Conv as UConv
from ultralytics.nn.modules.pad import ChannelPad  # 模块化 ChannelPad（你已按前面步骤加入）


# -------------------- name-based module utils --------------------
def _named_module_dict(model: nn.Module):
    return dict(model.named_modules())

def _get_child(parent: nn.Module, token: str):
    # 优先 _modules
    if token in parent._modules:
        return parent._modules[token]
    # 再尝试数字索引
    try:
        idx = int(token)
        if isinstance(parent, (nn.Sequential, nn.ModuleList)):
            return parent[idx]
    except Exception:
        pass
    # 最后 getattr
    return getattr(parent, token)

def _set_child(parent: nn.Module, token: str, new_m: nn.Module):
    if token in parent._modules:
        parent._modules[token] = new_m
        return
    try:
        idx = int(token)
        if isinstance(parent, (nn.Sequential, nn.ModuleList)):
            parent[idx] = new_m
            return
    except Exception:
        pass
    setattr(parent, token, new_m)

def _resolve_parent_child(model: nn.Module, full_name: str):
    parts = full_name.split(".")
    parent = model
    for t in parts[:-1]:
        parent = _get_child(parent, t)
    key = parts[-1]
    return parent, key


# -------------------- locate head --------------------
def _locate_head(model: nn.Module):
    if hasattr(model, "model"):  # Ultralytics YOLO 主容器
        cand = model.model[-1]
        if hasattr(cand, "cv2") and hasattr(cand, "cv3"):
            return cand, f"model.{len(model.model)-1}"
    for n, m in model.named_modules():
        if hasattr(m, "cv2") and hasattr(m, "cv3"):
            return m, n
    raise RuntimeError("Cannot locate YOLO head (no module with attributes cv2 & cv3).")


# -------------------- pruning helpers --------------------
def _sorted_unique_cpu(idx: torch.Tensor) -> torch.Tensor:
    idx = idx.detach().cpu().view(-1)
    if idx.numel() == 0:
        return idx
    idx = torch.unique(idx)
    idx, _ = torch.sort(idx)
    return idx

def _clamp_keep(old_c: int, prune_idx_cpu: torch.Tensor, min_keep: int) -> torch.Tensor:
    k = old_c - int(prune_idx_cpu.numel())
    if k >= min_keep:
        return prune_idx_cpu
    need = min_keep - k
    if need <= 0:
        return prune_idx_cpu
    return prune_idx_cpu[:-need] if need < prune_idx_cpu.numel() else prune_idx_cpu[:0]

def _shrink_bn_out(bn: nn.BatchNorm2d, keep_idx_cpu: torch.Tensor) -> nn.BatchNorm2d:
    assert isinstance(bn, nn.BatchNorm2d)
    dev = bn.weight.device
    keep = keep_idx_cpu.to(dev)
    C = keep.numel()
    new_bn = nn.BatchNorm2d(
        C, eps=bn.eps, momentum=bn.momentum,
        affine=bn.affine, track_running_stats=bn.track_running_stats
    ).to(dev)
    with torch.no_grad():
        if bn.affine:
            new_bn.weight.copy_(bn.weight[keep])
            new_bn.bias.copy_(bn.bias[keep])
        if bn.track_running_stats:
            new_bn.running_mean.copy_(bn.running_mean[keep])
            new_bn.running_var.copy_(bn.running_var[keep])
    return new_bn

def _prune_uconv_out(uconv: UConv, keep_idx_cpu: torch.Tensor):
    assert isinstance(uconv, UConv) and hasattr(uconv, "conv") and isinstance(uconv.conv, nn.Conv2d)
    dev = uconv.conv.weight.device
    keep = keep_idx_cpu.to(dev)
    # Conv out
    w = uconv.conv.weight.index_select(0, keep)
    uconv.conv.weight = nn.Parameter(w)
    if uconv.conv.bias is not None:
        b = uconv.conv.bias.index_select(0, keep)
        uconv.conv.bias = nn.Parameter(b)
    # BN out
    if hasattr(uconv, "bn") and isinstance(uconv.bn, nn.BatchNorm2d):
        uconv.bn = _shrink_bn_out(uconv.bn, keep_idx_cpu)


# -------------------- find last Scale + its producer UConv via forward hooks --------------------
@torch.no_grad()
def _find_tail_pairs_by_runtime(model: nn.Module, head, head_name: str, example: torch.Tensor):
    """
    返回每个分支的 (scale_name, scale_module, uconv_name, uconv_module, C_orig)
    利用前向 hook 记录调用顺序，定位“分支内最后执行的 ScaleLayer”，并回溯到最近的 UConv。
    """
    mod_dict = _named_module_dict(model)
    call_log = []  # (name, kind, ch, idx_in_log)

    # 为所有 UConv 和 Scale 注册 hook，记录调用顺序和输出通道数
    hooks = []
    idx_counter = {"i": 0}

    def make_hook(name, kind):
        def _hook(mod, inp, out):
            idx = idx_counter["i"]; idx_counter["i"] += 1
            if isinstance(out, (list, tuple)):
                t = out[0]
            else:
                t = out
            ch = int(t.shape[1]) if isinstance(t, torch.Tensor) and t.ndim == 4 else -1
            call_log.append((name, kind, ch, idx))
        return _hook

    for name, m in mod_dict.items():
        if isinstance(m, ScaleLayer):
            hooks.append(m.register_forward_hook(make_hook(name, "Scale")))
        elif isinstance(m, UConv):
            hooks.append(m.register_forward_hook(make_hook(name, "UConv")))

    # 走一次前向
    _ = model(example.clone())

    # 清理 hook
    for h in hooks:
        h.remove()

    # 对每个分支找“最后的 Scale”，再向前找最近的 UConv
    results = []  # list of dicts for each branch
    for list_name, blist in [("cv2", head.cv2), ("cv3", head.cv3)]:
        for i in range(len(blist)):
            prefix = f"{head_name}.{list_name}.{i}"
            sub = [x for x in call_log if x[0].startswith(prefix)]
            # 找最后一个 Scale
            scales = [(j, rec) for j, rec in enumerate(sub) if rec[1] == "Scale"]
            if not scales:
                # 该分支没有 Scale（不符合 SSS 假设），跳过
                continue
            last_scale_idx, last_scale_rec = max(scales, key=lambda it: it[1][3])  # 按全局执行顺序 idx 最大
            # 在它之前找最近的 UConv
            uconv_cand = None
            for j in range(last_scale_idx - 1, -1, -1):
                if sub[j][1] == "UConv":
                    uconv_cand = sub[j]; break
            if uconv_cand is None:
                continue

            s_name, _, s_ch, _ = last_scale_rec
            u_name, _, u_ch, _ = uconv_cand

            s_mod = mod_dict[s_name]
            u_mod = mod_dict[u_name]
            C_orig = int(u_mod.conv.weight.shape[0])
            # 一致性检查：lambda 长度应等于 uconv out_ch
            assert int(s_mod.lambda_.numel()) == C_orig, f"lambda({s_mod.lambda_.numel()}) != uconv.out({C_orig}) at {s_name}"

            results.append({
                "branch_name": f"{head_name}.{list_name}[{i}]",
                "list_name": list_name, "index": i,
                "scale_name": s_name, "scale": s_mod,
                "uconv_name": u_name, "uconv": u_mod,
                "C_orig": C_orig
            })
    return results


# -------------------- main --------------------
def main(weights, imgsz=640, tau=None, ratio=None, out="pruned.pt", report="prune_report.json",
         min_keep=16, dry_run=False):
    # 1) load
    y = YOLO(weights)
    model: nn.Module = y.model
    model.eval()

    # 2) locate head
    head, head_name = _locate_head(model)
    if not isinstance(head.cv2, nn.ModuleList) or not isinstance(head.cv3, nn.ModuleList):
        raise RuntimeError("head.cv2 or head.cv3 is not a ModuleList.")
    if len(head.cv2) != len(head.cv3):
        raise RuntimeError(f"len(cv2)={len(head.cv2)} != len(cv3)={len(head.cv3)}")

    # 3) runtime 分析，找到每个分支的“尾端 Scale + 其 UConv”
    device = next(model.parameters()).device
    example = torch.randn(1, 3, imgsz, imgsz, device=device)
    # 先热身一次，确保 head 分支被触达
    _ = model(example.clone())
    tails = _find_tail_pairs_by_runtime(model, head, head_name, example)
    assert len(tails) > 0, "No tail Scale/UConv pairs found. Did you inject SSS into head branches?"

    # 4) 计算全局阈值 tau（若给 ratio）
    if ratio is not None and tau is None:
        lambdas = []
        for t in tails:
            lam = t["scale"].lambda_.detach().abs().cpu().numpy().reshape(-1)
            lambdas.append(lam)
        a = np.concatenate(lambdas) if lambdas else np.array([])
        assert a.size > 0, "No lambda values collected from tails."
        keep = max(1, int((1.0 - float(ratio)) * a.size))
        tau = float(np.partition(a, keep - 1)[keep - 1])
        print(f"[PRUNE] ratio={ratio} -> τ={tau:.6f} (keep {keep}/{a.size})")
    assert tau is not None, "Either --tau or --ratio must be provided."

    # 5) 逐分支裁剪（UConv 出通道 + BN、Scale.lambda），并把 ChannelPad **插到 Scale 后面**
    params_before = sum(p.numel() for p in model.parameters())
    report_items, total_pruned = [], 0

    for t in tails:
        s_name, s_mod = t["scale_name"], t["scale"]
        u_name, u_mod = t["uconv_name"], t["uconv"]
        C = int(t["C_orig"])

        lam = s_mod.lambda_.detach().abs().cpu().view(-1)
        prune_idx = _sorted_unique_cpu((lam < tau).nonzero(as_tuple=False).view(-1))
        prune_idx = _clamp_keep(C, prune_idx, min_keep)
        keep_idx = torch.tensor([j for j in range(C) if j not in set(prune_idx.tolist())], dtype=torch.long)

        old_c, new_c = C, int(keep_idx.numel())
        if new_c == old_c:
            note = "no change (tau too low or min_keep too high)"
        else:
            note = "tail pruned; ChannelPad inserted right after Scale"

        if not dry_run:
            # 剪 UConv 出通道 + BN
            _prune_uconv_out(u_mod, keep_idx)
            # 缩短 Scale.lambda_
            s_mod.lambda_ = nn.Parameter(s_mod.lambda_.detach()[keep_idx])

            # 用 Sequential(Scale, ChannelPad) 替换原 Scale 模块，插入到**分支内部**
            parent, key = _resolve_parent_child(model, s_name)
            old_scale = _get_child(parent, key)
            pad = ChannelPad(keep_idx, c_orig=old_c)
            new_node = nn.Sequential(old_scale, pad)
            _set_child(parent, key, new_node)

        report_items.append({
            "branch": t["branch_name"],
            "scale_module": s_name,
            "uconv_module": u_name,
            "old_out": old_c,
            "new_out": new_c,
            "pruned_idx": prune_idx.tolist(),
            "note": note
        })
        total_pruned += (old_c - new_c)
        print(f"[PRUNE] {t['branch_name']}: {old_c}->{new_c} (pad after {s_name})")

    # 6) 简单自检：再跑一次前向，确保 shape 一致
    try:
        _ = model(example.clone())
    except Exception as e:
        raise RuntimeError(f"Sanity check forward failed after pruning: {e}")

    # 7) 保存与报告
    params_after = sum(p.numel() for p in model.parameters())
    if not dry_run:
        torch.save({"model": model}, out)
        print(f"[PRUNE] saved pruned module to: {out}")

    with open(report, "w") as f:
        json.dump({
            "summary": {
                "tau": float(tau),
                "min_keep": int(min_keep),
                "pairs_found": len(tails),
                "total_channels_pruned": int(total_pruned),
                "params_before": int(params_before),
                "params_after": int(params_after),
                "ratio_params": float(params_after / max(1, params_before))
            },
            "items": report_items
        }, f, indent=2)
    print(f"[PRUNE] report saved to: {report}")
    print(f"[PARAMS] {params_before} -> {params_after}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prune YOLOv8 head (SSS) by runtime-tail with in-branch ChannelPad.")
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--ratio", type=float, default=None)  # e.g. 0.25
    ap.add_argument("--out", type=str, default="pruned.pt")
    ap.add_argument("--report", type=str, default="prune_report.json")
    ap.add_argument("--min_keep", type=int, default=16)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    main(**vars(args))
