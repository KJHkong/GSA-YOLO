import torch
import torch.nn as nn
from ultralytics import YOLO

def global_l1_pruning(model_path, target_gflops=8.0):
    """
    执行全局 L1 范数剪枝，并尝试匹配目标 GFLOPs。
    """
    # 1. 加载模型
    model = YOLO(model_path)
    print(f"Original Model Loaded: {model_path}")

    # 2. 收集所有卷积层的 L1 norm
    all_norms = []
    prunable_layers = []
    
    for name, m in model.model.named_modules():
        # 仅针对卷积层进行通道级 L1 计算
        if isinstance(m, nn.Conv2d):
            # 计算每个输出通道权重绝对值的和 (L1 Norm)
            # weights shape: [out_ch, in_ch, k, k]
            norm = torch.norm(m.weight.data.view(m.weight.data.shape[0], -1), p=1, dim=1)
            all_norms.append(norm)
            prunable_layers.append(name)

    # 将所有 norm 合并为一个长向量并排序
    all_norms_vec = torch.cat(all_norms)
    all_norms_sorted, _ = torch.sort(all_norms_vec)

    # 3. 确定剪枝阈值
    # 通过循环微调这个 ratio 来精准对齐 8.0G
    pruning_ratio = 0.125  # 示例
    threshold_idx = int(len(all_norms_sorted) * pruning_ratio)
    threshold = all_norms_sorted[threshold_idx]
    print(f"Global L1 Threshold determined: {threshold:.6f}")

    # 4. 执行掩码操作 (Masking)
    num_pruned = 0
    total_channels = len(all_norms_vec)

    for name, m in model.model.named_modules():
        if isinstance(m, nn.Conv2d):
            weight_copy = m.weight.data.view(m.weight.data.shape[0], -1)
            norm = torch.norm(weight_copy, p=1, dim=1)
            
            # 创建掩码：norm 小于阈值的设为 0
            mask = (norm > threshold).float()
            
            # 应用掩码到权重和偏置
            m.weight.data.mul_(mask.view(-1, 1, 1, 1))
            if m.bias is not None:
                m.bias.data.mul_(mask)
            
            num_pruned += (mask == 0).sum().item()

    print(f"Pruning Complete: {num_pruned}/{total_channels} channels masked.")
    print(f"Estimated Pruning Ratio: {num_pruned/total_channels:.2%}")

    # 5. 保存模拟剪枝后的权重
    save_path = model_path.replace('.pt', '_global_l1.pt')
    model.save(save_path)
    print(f"Pruned model saved to: {save_path}")
    
    return model

if __name__ == "__main__":
    # 使用你的 Baseline 路径
    MODEL_PATH = '/root/autodl-tmp/YOLOv8-Project-8.1/runs/train/weights/best.pt'
    # 执行剪枝
    pruned_model = global_l1_pruning(MODEL_PATH, target_gflops=8.0)
    