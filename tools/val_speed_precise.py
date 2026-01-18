from ultralytics import YOLO

def run(model_path, data, imgsz=640, batch=1, device=0):
    model = YOLO(model_path)
    res = model.val(data=data, imgsz=imgsz, batch=batch, device=device, verbose=False)
    sp = getattr(res, "speed", {}) or {}
    pre  = float(sp.get("preprocess", 0.0))
    inf  = float(sp.get("inference", 0.0))
    post = float(sp.get("postprocess", 0.0))
    total = inf + post
    fps = 1000.0 / total if total > 0 else 0.0
    print(f"[{model_path}] batch={batch}, imgsz={imgsz}")
    print(f"Speed: {pre:.4f} ms preprocess, {inf:.4f} ms inference, {post:.4f} ms postprocess per image")
    print(f"FPS  : {fps:.2f}")

if __name__ == "__main__":
    # 你可以按需修改下面四行路径/批大小
    BASE = "/root/autodl-tmp/YOLOv8-Project-8.1"
    data = f"{BASE}/HiXray/HiXray.yaml"

    # baseline
    run(f"{BASE}/runs/hixray_training/train_8.6_3090/weights/best.pt", data, imgsz=640, batch=1,  device=0)
    run(f"{BASE}/runs/hixray_training/train_8.6_3090/weights/best.pt", data, imgsz=640, batch=64, device=0)

    # 稀疏剪枝后
    run(f"{BASE}/runs/hixray_finetune/ft_stage2_tail_e15/weights/best.pt", data, imgsz=640, batch=1,  device=0)
    run(f"{BASE}/runs/hixray_finetune/ft_stage2_tail_e15/weights/best.pt", data, imgsz=640, batch=64, device=0)
