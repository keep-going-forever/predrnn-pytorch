import torch
import torch.nn as nn
from typing import Tuple
from core.composites.DAM.origin_attention import OriginSelfAttentionMemory
from core.composites.DAM.cross_attention import CrossSelfAttentionMemory
class NewDamCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, kernel_size, stride, layer_norm):
        super(NewDamCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = kernel_size // 2
        self._forget_bias = 1.0  # 加入忘记偏置项

        # 初始化卷积层
        if layer_norm:
            self.W_C = nn.Sequential(
                nn.Conv2d(in_channel + num_hidden * 2, num_hidden * 3,
                          kernel_size=kernel_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.W_M = nn.Sequential(
                nn.Conv2d(in_channel + num_hidden * 2, num_hidden * 3,
                          kernel_size=kernel_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.W_O = nn.Sequential(
                nn.Conv2d(in_channel + num_hidden * 4, num_hidden,
                          kernel_size=kernel_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
            self.W_H = nn.Sequential(
                nn.Conv2d(num_hidden * 3, num_hidden,
                          kernel_size=1, padding='same', bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
        else:
            self.W_C = nn.Conv2d(in_channel + num_hidden * 2, num_hidden * 3,
                                 kernel_size=kernel_size, stride=stride, padding=self.padding, bias=False)
            self.W_M = nn.Conv2d(in_channel + num_hidden * 2, num_hidden * 3,
                                 kernel_size=kernel_size, stride=stride, padding=self.padding, bias=False)
            self.W_O = nn.Conv2d(in_channel + num_hidden * 4, num_hidden,
                                 kernel_size=kernel_size, stride=stride, padding=self.padding, bias=False)
            self.W_H = nn.Conv2d(num_hidden * 3, num_hidden,
                                 kernel_size=1, padding='same', bias=False)



        # 引入 SelfAttentionMemory
        # self.attention = OriginSelfAttentionMemory(in_channel*2, num_hidden, width, kernel_size, stride, layer_norm)
        self.attention = CrossSelfAttentionMemory(in_channel, num_hidden, width, kernel_size, stride, layer_norm)

    def forward(self, x_t: torch.Tensor, h_t: torch.Tensor,
                c_t: torch.Tensor, m_t: torch.Tensor, x_pre: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _, H, W = x_t.shape

        # 初始化 x_pre 如果未提供
        if x_pre is None:
            x_pre = torch.zeros_like(x_t).to("cuda:0")
        # conbined_attention = torch.cat([x_t, x_pre], dim=1)

        attention,score = self.attention(x_t,x_pre)
        # 计算 C 门控
        combined = torch.cat([x_t, attention, h_t], dim=1)
        C_gate = self.W_C(combined)
        i, f, g = torch.split(C_gate, self.num_hidden, dim=1)
        i, f, g = torch.sigmoid(i), torch.sigmoid(f + self._forget_bias), torch.tanh(g)
        new_c = f * c_t + i * g

        combined_m = torch.cat([x_t, attention, m_t], dim=1)
        M_gate = self.W_M(combined_m)
        i_m, f_m, g_m = torch.split(M_gate, self.num_hidden, dim=1)
        #这里注意修改了什么
        i_m, f_m, g_m = torch.sigmoid(i_m), torch.sigmoid(f_m + self._forget_bias), torch.tanh(g_m)
        new_m = f_m * m_t + i_m * g_m

        combined_o = torch.cat([x_t, attention,h_t,new_c, new_m], dim=1)
        O_gate = torch.tanh(self.W_O(combined_o))
        combined_h = torch.cat([attention,new_c, new_m], dim=1)
        H_gate = torch.tanh(self.W_H(combined_h))

        new_h = O_gate*H_gate




        return new_h,new_c, new_m