import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureLoss(nn.Module):
    def __init__(self):
        super(FeatureLoss, self).__init__()

    def forward(self, f_s, f_t):
        """
        f_s: 学生模型的特征图列表 [batch, channel, h, w]
        f_t: 教师模型的特征图列表
        """
        loss = 0
        for s, t in zip(f_s, f_t):
            # 如果通道数不一致，通常需要一个 1x1 卷积对齐（这里假设已对齐）
            loss += F.mse_loss(s, t)
        return loss