import torch
import torch.nn as nn
from typing import Tuple
from core.composites.DAM.common_attention import CommonSelfAttentionMemory
class NewDamCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, kernel_size, stride, layer_norm):
        super(NewDamCell, self).__init__()

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
            self.W_O = nn.Sequential(
                nn.Conv2d(in_channel * 2 + num_hidden * 2, num_hidden,
                          kernel_size=kernel_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
        else:
            self.W_C = nn.Conv2d(in_channel * 2 + num_hidden * 2, num_hidden * 3,
                                 kernel_size=kernel_size, stride=stride, padding=self.padding, bias=False)
            self.W_O = nn.Conv2d(in_channel * 2 + num_hidden * 2, num_hidden,
                                 kernel_size=kernel_size, stride=stride, padding=self.padding, bias=False)



        # 引入 SelfAttentionMemory
        self.attention = CommonSelfAttentionMemory(num_hidden, num_hidden, width, kernel_size, stride, layer_norm)

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

        o = torch.tanh(self.W_O(combined))

        new_h = o * torch.tanh(new_c)


        new_H, new_memory, attention_h=self.attention(new_h, m_t)



        return new_H,new_c, new_memory