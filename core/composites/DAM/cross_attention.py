import torch
import torch.nn as nn
from typing import Tuple

class CrossSelfAttentionMemory(nn.Module):
    def __init__(self, in_channel, num_hidden, width, kernel_size, stride, layer_norm):
        super(CrossSelfAttentionMemory, self).__init__()

        self.num_hidden = num_hidden
        self.padding = kernel_size // 2

        # 初始化 Query、Key 和 Value 层，根据是否使用 LayerNorm
        if layer_norm:
            self.query = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
            self.key = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
            self.value = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
            self.z = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
        else:
            self.query = nn.Conv2d(in_channel, num_hidden, kernel_size=1,
                                      stride=1, padding=0, bias=False)
            self.key = nn.Conv2d(in_channel, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)
            self.value = nn.Conv2d(in_channel, num_hidden, kernel_size=1,
                                     stride=1, padding=0, bias=False)
            self.z = nn.Conv2d(num_hidden, num_hidden, kernel_size=1,
                                 stride=1, padding=0, bias=False)

    def forward(self, current_frame,pred_frame) -> torch.Tensor:
        """
        输入:
            ns (torch.Tensor): NS 模块的输出, 形状为 [batch, in_channel, H, W]
            m (torch.Tensor): Memory 状态, 形状为 [batch, num_hidden, H, W]

        返回:
            torch.Tensor: 更新后的 Memory 状态, 形状为 [batch, num_hidden, H, W]
        """
        batch_size, _, H, W = current_frame.shape

        # 计算 Query、Key 和 Value
        q = self.query(current_frame).view(batch_size, self.num_hidden, H * W).transpose(1, 2)
        k = self.key(pred_frame).view(batch_size, self.num_hidden, H * W)
        v = self.value(pred_frame).view(batch_size, self.num_hidden, H * W)

        # 计算注意力得分
        attention = torch.softmax(torch.bmm(q, k), dim=-1)

        # 计算 Z_m
        z = torch.matmul(attention, v.transpose(1, 2))
        z = z.transpose(1, 2).view(batch_size, self.num_hidden, H, W)

        # 使用卷积更新 Memory 状态
        new = self.z(z)

        return new,attention
