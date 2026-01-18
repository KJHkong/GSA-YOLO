#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复带有 __main__.ChannelPad 的剪枝权重：
- 在运行时为反序列化注入一个同名的 ChannelPad 到 __main__，
  该类来自 ultralytics.nn.modules.pad.ChannelPad（我们已模块化）。
- 成功加载后立刻按模块路径重新保存，得到通用可加载的 ckpt。

用法示例：
python scripts/fix_channelpad_checkpoint.py \
  --src /path/to/pruned_head.pt \
  --dst /path/to/pruned_head_fixed.pt \
  [--verify]
"""

import sys
import types
from pathlib import Path
import argparse
import torch

def _ensure_repo_pythonpath():
    """
    尝试自动把本地 ultralytics-SSS 加入 sys.path（相对当前脚本位置推断）。
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[1]  # .../YOLOv8-Project-8.1/ultralytics-SSS/scripts -> parents[1] = .../YOLOv8-Project-8.1
    udir = repo_root / "ultralytics-SSS"
    if udir.exists():
        sys.path.insert(0, str(udir))

def _alias_main_channelpad():
    """
    在 __main__ 模块下注册一个名为 ChannelPad 的类，指向模块化实现：
    ultralytics.nn.modules.pad.ChannelPad
    """
    # 确保 ultralytics-SSS 在路径中
    _ensure_repo_pythonpath()

    # 再导入模块化的 ChannelPad
    try:
        from ultralytics.nn.modules.pad import ChannelPad  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "无法导入 ultralytics.nn.modules.pad.ChannelPad。"
            "请确认你已创建并导出该类：\n"
            "  - 文件：ultralytics-SSS/ultralytics/nn/modules/pad.py\n"
            "  - __init__.py 中：from .pad import ChannelPad"
        ) from e

    # 创建/获取 __main__ 模块，并挂载 ChannelPad
    if "__main__" not in sys.modules:
        sys.modules["__main__"] = types.ModuleType("__main__")
    # 将模块化 ChannelPad 绑定到 __main__ 命名空间
    sys.modules["__main__"].ChannelPad = sys.modules["ultralytics.nn.modules.pad"].ChannelPad

def _load_yolo(weights_path: str):
    """
    用 ultralytics 的 YOLO 加载权重（在别名注入后）。
    """
    from ultralytics import YOLO
    return YOLO(weights_path)

def main(src: str, dst: str, verify: bool = False):
    _alias_main_channelpad()

    # 加载旧 ckpt（包含 __main__.ChannelPad）
    y = _load_yolo(src)
    model = y.model

    # 立刻按模块路径重新保存
    torch.save({"model": model}, dst)
    print(f"[FIX] Re-saved checkpoint to: {dst}")

    if verify:
        # 验证：用 ultralytics 再加载一次新权重，确保通用可读
        y2 = _load_yolo(dst)
        _ = sum(p.numel() for p in y2.model.parameters())
        print("[VERIFY] Successfully loaded fixed checkpoint.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix __main__.ChannelPad in pruned checkpoint.")
    parser.add_argument("--src", required=True, help="Path to old pruned checkpoint (with __main__.ChannelPad)")
    parser.add_argument("--dst", required=True, help="Path to save fixed checkpoint")
    parser.add_argument("--verify", action="store_true", help="Verify by reloading the fixed checkpoint")
    args = parser.parse_args()
    main(args.src, args.dst, args.verify)
