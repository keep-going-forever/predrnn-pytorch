import torch
import torch.nn as nn
import torch.nn.functional as F

class RadarLoss(nn.Module):
    def __init__(self):
        super(RadarLoss, self).__init__()

    def forward(self, pred, target):
        # 计算每一个像素点的MAE
        mse = (pred - target) ** 2
        pred = pred * 70
        target = target * 70

        # 初始化权重矩阵
        weights = torch.zeros_like(mae)

        # 根据真实值的分档进行权重分配，并计算softmax权重
        for mask in [(target < 20), (target >= 20) & (target < 30), (target >= 30) & (target < 40), (target >= 40)]:
            if mask.sum() > 0:
                mae_segment = mse[mask]
                # 排除掉非当前类别的零值
                if mae_segment.numel() > 0:
                    # 对当前类别的有效MAE值进行softmax
                    weights_segment = F.softmax(mae_segment, dim=0)
                    # 将权重分配回原始权重矩阵中
                    weights[mask] = weights_segment

        # 计算加权后的MAE损失
        weighted_mse = weights * mse
        loss = weighted_mae.mean()

        return loss


