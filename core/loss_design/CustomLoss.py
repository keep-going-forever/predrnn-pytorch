import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, config):
        super(CustomLoss, self).__init__()
        self.config = config
        self.stability_value = config.stability_value  # 避免分母为零的稳定值
        self.threshold = config.threshold  # 用于区分大雨强和小雨强的dbz阈值

    def forward(self, predictions, targets):
        # 假设predictions和targets的形状都是 (batch_size, seq, h, w, c)
        batch_size, seq, h, w, c = predictions.shape

        # 计算所有区域的均方误差 E_avg
        E_avg = torch.mean((predictions - targets) ** 2)

        # 计算大雨强区域的均方误差 E_heavy (MSE计算)
        heavy_mask = targets >= self.threshold  # 大雨强区域掩码
        if torch.sum(heavy_mask) == 0:
            E_heavy = torch.tensor(0.0, device=predictions.device)
        else:
            E_heavy = torch.mean((predictions[heavy_mask] - targets[heavy_mask]) ** 2)

        # 计算整体MSE
        overall_mse = torch.mean((predictions - targets) ** 2)

        # 计算损失函数
        loss = E_heavy / (E_avg + self.stability_value) + self.threshold * overall_mse

        return loss


# 示例用法
if __name__ == "__main__":
    # 假设输入的预测值和真实值的维度是 [2, 10, 64, 64, 3]，即 batch_size=2, time_steps=10, height=64, width=64, channels=3
    predictions = torch.rand(2, 10, 64, 64, 3)  # 示例预测值
    targets = torch.rand(2, 10, 64, 64, 3)  # 示例真实值

    # 实例化自定义损失类并计算损失
    custom_loss = CustomLoss(stability_value=1e-6, threshold=0.7)  # 设置阈值为0.7
    loss = custom_loss(predictions, targets)
    print(f"损失值: {loss.item()}")
