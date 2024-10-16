import torch
import torch.nn as nn
from typing import Tuple
from core.composites.DAM.attention import SelfAttentionMemory

import torch
import torch.nn as nn
from typing import Tuple

class DamCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, kernel_size, stride, layer_norm):
        super(DamCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = kernel_size // 2
        self._forget_bias = 1.0  # 加入忘记偏置项

        # 初始化卷积层
        if layer_norm:
            self.W_C = nn.Sequential(
                nn.Conv2d(in_channel * 2 + num_hidden * 2, num_hidden * 3,
                          kernel_size=kernel_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.W_M = nn.Sequential(
                nn.Conv2d(in_channel * 2 + num_hidden * 2, num_hidden * 3,
                          kernel_size=kernel_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
        else:
            self.W_C = nn.Conv2d(in_channel * 2 + num_hidden * 2, num_hidden * 3,
                                 kernel_size=kernel_size, stride=stride, padding=self.padding, bias=False)
            self.W_M = nn.Conv2d(in_channel * 2 + num_hidden * 2, num_hidden * 3,
                                 kernel_size=kernel_size, stride=stride, padding=self.padding, bias=False)

        self.W_o = nn.Conv2d(in_channel * 3 + num_hidden * 3, num_hidden,
                             kernel_size=kernel_size, stride=stride, padding=self.padding, bias=False)
        self.W_h = nn.Conv2d(in_channel * 3 + num_hidden * 3, num_hidden,
                             kernel_size=1, stride=1, padding=0, bias=False)

        # 引入 SelfAttentionMemory
        self.attention = SelfAttentionMemory(in_channel, num_hidden, width, kernel_size, stride, layer_norm)

    def forward(self, x_t: torch.Tensor, h_t: torch.Tensor,
                c_t: torch.Tensor, m_t: torch.Tensor, x_pre: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _, H, W = x_t.shape

        # 初始化 x_pre 如果未提供
        if x_pre is None:
            x_pre = torch.zeros_like(x_t).to("cuda:0")

        # 计算 C 门控
        combined = torch.cat([x_t, (x_t - x_pre), c_t, h_t], dim=1)
        C_gate = self.W_C(combined)
        i, f, g = torch.split(C_gate, self.num_hidden, dim=1)
        i, f, g = torch.sigmoid(i), torch.sigmoid(f + self._forget_bias), torch.tanh(g)
        new_c = f * c_t + i * g

        # 使用 SelfAttentionMemory 更新 M 门控
        m_attention = self.attention(x_t - x_pre, m_t)
        combined_m = torch.cat([x_t, (x_t - x_pre), m_attention, new_c], dim=1)
        M_gate = self.W_M(combined_m)
        i_, f_, g_ = torch.split(M_gate, self.num_hidden, dim=1)
        i_, f_, g_ = torch.sigmoid(i_), torch.sigmoid(f_ + self._forget_bias), torch.tanh(g_)
        new_m = f_ * m_t + i_ * g_

        # 计算输出 O 和新的隐藏状态 H
        combined_o = torch.cat([x_t, x_pre, (x_t - x_pre), new_c, new_m, h_t], dim=1)
        o = torch.sigmoid(self.W_o(combined_o))
        new_h = o * torch.tanh(self.W_h(combined_o))

        return new_h, new_c, new_m

