# scripts/benchmark_flops_fps.py
import os, time, argparse, torch
from contextlib import nullcontext

# 用你的 ultralytics-SSS
import sys
sys.path.insert(0, "/root/autodl-tmp/YOLOv8-Project-8.1/ultralytics-SSS")
from ultralytics import YOLO

def try_import_thop():
    try:
        from thop import profile
        return profile
    except Exception:
        return None

def get_model(w, allow_fuse=False):
    m = YOLO(w).model.eval()
    if allow_fuse and hasattr(m, "fuse"):
        try:
            m = m.fuse()
        except Exception:
            pass
    return m

@torch.inference_mode()
def bench_fps(m, imgsz=640, iters=200, warmup=50, batch=1, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = m.to(device)
    x = torch.randn(batch, 3, imgsz, imgsz, device=device)
    # warmup
    for _ in range(warmup):
        _ = m(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = m(x)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    ms = dt * 1000
    fps = batch / dt
    return ms, fps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True, help="原始模型权重（可 fuse）")
    ap.add_argument("--pruned", required=True, help="剪枝后模型权重（不要 fuse）")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    profile = try_import_thop()
    if profile is None:
        print("⚠️ 未安装 thop，跳过 FLOPs 计算。pip install thop 可启用。")

    # 原模型
    m_orig = get_model(args.orig, allow_fuse=True)
    # 剪枝模型
    m_pruned = get_model(args.pruned, allow_fuse=False)

    # FLOPs
    if profile is not None:
        x = torch.randn(1,3,args.imgsz,args.imgsz)
        flops_o, params_o = profile(m_orig, inputs=(x,), verbose=False)
        print(f"[ORIG]  FLOPs: {flops_o/1e9:.3f} G, Params: {params_o/1e6:.3f} M")
        flops_p, params_p = profile(m_pruned, inputs=(x,), verbose=False)
        print(f"[PRUN]  FLOPs: {flops_p/1e9:.3f} G, Params: {params_p/1e6:.3f} M")
    else:
        # 至少把参数量报一下
        p_o = sum(p.numel() for p in m_orig.parameters())
        p_p = sum(p.numel() for p in m_pruned.parameters())
        print(f"[ORIG]  Params: {p_o/1e6:.3f} M")
        print(f"[PRUN]  Params: {p_p/1e6:.3f} M")

    # FPS（batch=1 和 batch=16 来各测一次）
    for b in [1, 16]:
        ms_o, fps_o = bench_fps(m_orig, imgsz=args.imgsz, batch=b)
        ms_p, fps_p = bench_fps(m_pruned, imgsz=args.imgsz, batch=b)
        print(f"[FPS] batch={b} | ORIG: {ms_o:.2f} ms/img, {fps_o:.1f} FPS | PRUN: {ms_p:.2f} ms/img, {fps_p:.1f} FPS")

if __name__ == "__main__":
    main()
