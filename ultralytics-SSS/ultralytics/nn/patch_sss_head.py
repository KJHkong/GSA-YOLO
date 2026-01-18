import torch.nn as nn
from ultralytics.nn.modules.sss import ScaleLayer
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect

def _patch_module_after_bn(module: nn.Module):
    """Replace Conv(BN,act) with Sequential(Conv, ScaleLayer) recursively."""
    for name, child in list(module.named_children()):
        if isinstance(child, Conv) and hasattr(child, "bn") and child.bn is not None:
            c = child.bn.num_features
            seq = nn.Sequential(child, ScaleLayer(c))  # conv->bn->act -> scale
            setattr(module, name, seq)
        else:
            _patch_module_after_bn(child)

def add_scale_into_detect_head(model: nn.Module):
    """Inject ScaleLayer only inside Detect head."""
    for m in model.modules():
        if isinstance(m, Detect):
            _patch_module_after_bn(m)
