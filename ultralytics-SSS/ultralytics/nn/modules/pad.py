# ultralytics/nn/modules/pad.py
import torch
import torch.nn as nn

class ChannelPad(nn.Module):
    """
    无参数的通道恢复层：将被裁剪后的 C_kept 通道拷贝回原始 C_orig 的位置，其余位置填 0。
    keep_idx 是 length=C_kept 的 LongTensor，表示保留通道在 [0..C_orig-1] 的原始索引。
    """
    def __init__(self, keep_idx, c_orig: int):
        super().__init__()
        keep_idx = torch.as_tensor(keep_idx, dtype=torch.long).view(-1)
        self.register_buffer("keep_idx", keep_idx)
        self.c_orig = int(c_orig)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C_kept, H, W] -> y: [N, C_orig, H, W]
        N, _, H, W = x.shape
        y = x.new_zeros((N, self.c_orig, H, W))
        y.index_copy_(1, self.keep_idx.to(x.device), x)
        return y
