# import torch
# import torch.nn as nn
# from ultralytics.utils.loss import v8DetectionLoss

# class CustomV8DetectionLoss:
#     """
#     自定义 v8DetectionLoss，去掉 .detach()，保证 loss 可反向传播。
#     """

#     def __init__(self, model, tal_topk: int = 10):
#         # 用官方的 v8DetectionLoss 初始化
#         self.base_loss = v8DetectionLoss(model, tal_topk=tal_topk)

#     def __call__(self, preds, batch):
#         # 调用原版 loss
#         loss, loss_items = self.base_loss(preds, batch)

#         # 🚨 确保 loss 可导：去掉 detach + 变成标量
#         if isinstance(loss, torch.Tensor):
#             if loss.ndim > 0:
#                 loss = loss.sum()
#             # 强制 requires_grad=True
#             if not loss.requires_grad:
#                 loss = loss.clone().detach().requires_grad_(True)

#         return loss, loss_items
import torch
from ultralytics.utils.loss import v8DetectionLoss

class CustomV8DetectionLoss:
    """
    自定义 v8DetectionLoss，确保返回的 loss 可反向传播。
    """

    def __init__(self, model, tal_topk: int = 10):
        self.base_loss = v8DetectionLoss(model, tal_topk=tal_topk)

    def __call__(self, preds, batch):
        loss, loss_items = self.base_loss(preds, batch)

        # 🚨 保证 loss 是 tensor，保持计算图
        if isinstance(loss, torch.Tensor):
            if loss.ndim > 0:
                loss = loss.sum()
        else:
            # 万一返回 float，转 tensor
            loss = torch.tensor(loss, device=preds[0].device, requires_grad=True)

        return loss, loss_items
