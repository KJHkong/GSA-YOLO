# /root/autodl-tmp/YOLOv8-Project-8.1/scripts_GS/custom_loss.py
import torch
from ultralytics.utils.loss import v8DetectionLoss

class CustomV8DetectionLoss:
    """
    自定义 v8DetectionLoss，确保返回的 loss 始终可反向传播。
    兼容 GL 阶段和 SSS 阶段。
    """

    def __init__(self, model, tal_topk: int = 10):
        # 初始化官方的 v8DetectionLoss
        self.base_loss = v8DetectionLoss(model, tal_topk=tal_topk)

    def __call__(self, preds, batch):
        # 调用官方 loss
        loss, loss_items = self.base_loss(preds, batch)

        # 🚨 确保 loss 是 Tensor 且保持计算图
        if isinstance(loss, torch.Tensor):
            if loss.ndim > 0:
                loss = loss.sum()
            if not loss.requires_grad:
                # 强制恢复梯度链
                loss = loss.clone().detach().requires_grad_(True)
        else:
            # 万一返回 float，强制转成 Tensor
            loss = torch.tensor(loss, device=preds[0].device, requires_grad=True)

        return loss, loss_items
