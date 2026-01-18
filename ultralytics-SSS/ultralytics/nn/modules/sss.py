import torch
import torch.nn as nn

class ScaleLayer(nn.Module):
    """
    Per-channel learnable scale placed AFTER BatchNorm2d.
    Parameter name 'lambda_' is used by our sparsity code.
    """
    def __init__(self, num_channels: int):
        super().__init__()
        self.lambda_ = nn.Parameter(torch.ones(num_channels))

    def forward(self, x):
        return x * self.lambda_.view(1, -1, 1, 1)  # [B,C,H,W]
