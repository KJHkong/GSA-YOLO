class BoxLoss(nn.Module):
    def __init__(self):
        super(BoxLoss, self).__init__()

    def forward(self, pred_bboxes_s, pred_bboxes_t):
        """
        pred_bboxes_s: 学生预测的框 [batch, num_boxes, 4]
        pred_bboxes_t: 教师预测的框
        """
        # 通常只对教师模型置信度较高的框进行蒸馏
        loss = F.smooth_l1_loss(pred_bboxes_s, pred_bboxes_t)
        return loss