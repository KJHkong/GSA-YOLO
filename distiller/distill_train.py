import torch
from ultralytics.utils.loss import v8DetectionLoss # 假设使用 v8 损失

class GSA_Distill_Trainer:
    def __init__(self, student_model, teacher_model, mode='Ada-KD'):
        self.student = student_model
        self.teacher = teacher_model
        self.mode = mode # 'Ada-KD', 'Feature-KD', 'Box-KD'
        self.teacher.eval() # 教师模型保持在 eval 模式

    def get_distill_loss(self, batch, student_outputs):
        with torch.no_grad():
            teacher_outputs = self.teacher(batch['img'])

        if self.mode == 'Ada-KD':
            # 你的核心代码：计算类别概率分布的 KL 散度
            p_s = F.log_softmax(student_outputs['cls'], dim=-1)
            p_t = F.softmax(teacher_outputs['cls'], dim=-1)
            return F.kl_div(p_s, p_t, reduction='batchmean')
        
        elif self.mode == 'Feature-KD':
            # 对比方法 1：特征图 MSE 损失
            return F.mse_loss(student_outputs['features'], teacher_outputs['features'])
            
        elif self.mode == 'Box-KD':
            # 对比方法 2：边界框回归损失
            return F.smooth_l1_loss(student_outputs['bboxes'], teacher_outputs['bboxes'])
            
        return 0.0