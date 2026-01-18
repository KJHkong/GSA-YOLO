# /root/autodl-tmp/YOLOv8-Project-8.1/scripts_GS/sss.py
import torch
import torch.nn as nn

def get_head_convs(model):
    """
    提取YOLOv8 Head部分的Conv2d层
    - YOLOv8的Head通常在 model.22、model.23、model.24（具体需确认）
    """
    head_convs = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if name.startswith("model.22") or \
               name.startswith("model.23") or \
               name.startswith("model.24"):
                head_convs.append((name, module))
    return head_convs


def sss_penalty(model, lambda_s=1e-4):
    """
    对Head的Conv2d层施加SSS稀疏正则化
    - 训练时保持梯度连通性
    - 使用L1范数作为稀疏约束
    """
    penalties = []
    for name, module in get_head_convs(model):
        w = module.weight   # ⚠️ 不要用 .data/.detach()，保持梯度
        group_norms = w.view(w.size(0), -1).norm(1, dim=1)  # L1范数 [out_channels]
        penalties.append(group_norms.sum())

    if len(penalties) == 0:
        # 返回 dummy 张量，保持梯度链
        return next(model.parameters()).sum() * 0.0

    return lambda_s * sum(penalties)


if __name__ == "__main__":
    # 🔍 简单测试（只用于检查，不影响训练）
    from ultralytics import YOLO
    DEVICE = "cuda:0"
    model_path = "/root/autodl-tmp/YOLOv8-Project-8.1/runs/hixray_gl/weights/best.pt"
    yolo = YOLO(model_path)
    model = yolo.model.to(DEVICE)

    head_convs = get_head_convs(model)
    print(f"✅ Found {len(head_convs)} head Conv2d layers")

    penalty = sss_penalty(model, lambda_s=1e-4)
    # 这里只是打印 → detach 避免污染梯度
    print(f"SSS penalty (λ=1e-4): {penalty.detach().cpu().item():.6f}")
