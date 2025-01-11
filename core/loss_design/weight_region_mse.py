import torch
import torch.nn as nn
import torch.nn.functional as F


class weight_region_mse(nn.Module):
    def __init__(self, config, reduction='mean'):
        """
        初始化分区域 MSE 损失函数

        Parameters:
        - window_size (int): 每个区域的窗口大小
        - reduction (str): 损失的计算方式，'mean', 'sum' 或 'none'
        """
        super(weight_region_mse, self).__init__()
        self.configs = config
        self.reduction = reduction
        self.window_size = config.window_size

    def forward(self, input, target):
        """
        计算分区域 MSE 损失并返回最终的损失值

        Parameters:
        - input (Tensor): 预测图像，形状为 [batch_size, seq, h, w,c]
        - target (Tensor): 目标图像，形状为 [batch_size, seq, h, w,c]
        Returns:
        - Tensor: 计算出的损失值，形状依赖于 reduction 参数
        """
        batch_size, seq, h, w, c = input.shape
        total_loss = 0.0  # 累加所有区域的损失

        # 遍历每个样本和时间步
        for b in range(batch_size):
            for s in range(seq):
                for i in range(0, h, self.window_size):
                    for j in range(0, w, self.window_size):
                        # 提取当前窗口区域
                        img1_window = input[b, s, i:i + self.window_size, j:j + self.window_size, :]
                        img2_window = target[b, s, i:i + self.window_size, j:j + self.window_size, :]

                        # 计算该区域内每个像素的MSE损失并进行加权
                        mse_loss = F.mse_loss(img1_window, img2_window, reduction='none')  # 每个区域使用 'none' 计算MSE

                        # 根据真实值范围加权
                        weight = torch.ones_like(img2_window)
                        weight[img2_window*70 > 40] = 30
                        weight[(img2_window*70 <= 40) & (img2_window*70 > 30)] = 20
                        weight[(img2_window*70 <= 30) & (img2_window*70 > 20)] = 10
                        weight[img2_window*70 <= 20] = 5

                        weighted_mse = mse_loss * weight
                        total_loss += weighted_mse.mean()  # 平均加权后的MSE损失

        # 根据 reduction 参数计算最终的损失
        if self.reduction == 'mean':
            return total_loss / (batch_size * seq)  # 平均损失，关注每个区域
        elif self.reduction == 'sum':
            return total_loss  # 返回损失的总和
        elif self.reduction == 'none':
            return total_loss / (batch_size * seq)  # 返回每个区域的平均损失
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

