import torch
import torch.nn as nn

import torch
import torch.nn as nn
from typing import Tuple

class SelfAttentionMemory(nn.Module):
    def __init__(self, in_channel, num_hidden, width, kernel_size, stride, layer_norm):
        super(SelfAttentionMemory, self).__init__()

        self.num_hidden = num_hidden
        self.padding = kernel_size // 2

        # 初始化 Query、Key 和 Value 层，根据是否使用 LayerNorm
        if layer_norm:
            self.query_ns = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
            self.key_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
            self.value_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
            self.z_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
        else:
            self.query_ns = nn.Conv2d(in_channel, num_hidden, kernel_size=1,
                                      stride=1, padding=0, bias=False)
            self.key_m = nn.Conv2d(num_hidden, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)
            self.value_m = nn.Conv2d(num_hidden, num_hidden, kernel_size=1,
                                     stride=1, padding=0, bias=False)
            self.z_m = nn.Conv2d(num_hidden, num_hidden, kernel_size=1,
                                 stride=1, padding=0, bias=False)

    def forward(self, ns: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        输入:
            ns (torch.Tensor): NS 模块的输出, 形状为 [batch, in_channel, H, W]
            m (torch.Tensor): Memory 状态, 形状为 [batch, num_hidden, H, W]

        返回:
            torch.Tensor: 更新后的 Memory 状态, 形状为 [batch, num_hidden, H, W]
        """
        batch_size, _, H, W = ns.shape


        # 计算 Query、Key 和 Value
        q_ns = self.query_ns(ns).view(batch_size, self.num_hidden, H * W).transpose(1, 2)
        k_m = self.key_m(m).view(batch_size, self.num_hidden, H * W)
        v_m = self.value_m(m).view(batch_size, self.num_hidden, H * W)

        # 计算注意力得分
        attention_m = torch.softmax(torch.bmm(q_ns, k_m), dim=-1)

        # 计算 Z_m
        z_m = torch.matmul(attention_m, v_m.transpose(1, 2))
        z_m = z_m.transpose(1, 2).view(batch_size, self.num_hidden, H, W)

        # 使用卷积更新 Memory 状态
        new_m = self.z_m(z_m)

        return new_m

