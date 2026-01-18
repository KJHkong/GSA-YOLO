import torch
import torch.nn as nn

def get_neck_convs(model):
    """提取YOLOv8n Neck部分的Conv2d层"""
    neck_convs = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if name.startswith("model.12") or \
               name.startswith("model.15") or \
               name.startswith("model.16") or \
               name.startswith("model.18") or \
               name.startswith("model.19") or \
               name.startswith("model.21"):
                neck_convs.append((name, module))
    return neck_convs


def group_lasso_penalty(model, lambda_gl=1e-5):
    """对Neck的Conv2d层施加Group Lasso正则化"""
    penalties = []
    for name, module in get_neck_convs(model):
        w = module.weight
        group_norms = w.view(w.size(0), -1).norm(2, dim=1)  # [out_channels]
        penalties.append(group_norms.sum())

    if len(penalties) == 0:
        # 返回一个 dummy 张量，保持梯度连通性
        return next(model.parameters()).sum() * 0.0

    return lambda_gl * sum(penalties)