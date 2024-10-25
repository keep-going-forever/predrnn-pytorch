import torch
import torch.nn as nn
from typing import Tuple

class CommonSelfAttentionMemory(nn.Module):
    def __init__(self, input_dim, hidden_dim, width, kernel_size, stride, layer_norm) -> None:
        super(CommonSelfAttentionMemory, self).__init__()

        # 添加 LayerNorm 的条件逻辑
        if layer_norm:
            self.query_h = nn.Sequential(
                nn.Conv2d(input_dim, hidden_dim, 1, padding="same"),
                nn.LayerNorm([hidden_dim, width, width])
            )
            self.key_h = nn.Sequential(
                nn.Conv2d(input_dim, hidden_dim, 1, padding="same"),
                nn.LayerNorm([hidden_dim, width, width])
            )
            self.value_h = nn.Sequential(
                nn.Conv2d(input_dim, input_dim, 1, padding="same"),
                nn.LayerNorm([input_dim, width, width])
            )
            self.z_h = nn.Sequential(
                nn.Conv2d(input_dim, input_dim, 1, padding="same"),
                nn.LayerNorm([input_dim, width, width])
            )

            self.key_m = nn.Sequential(
                nn.Conv2d(input_dim, hidden_dim, 1, padding="same"),
                nn.LayerNorm([hidden_dim, width, width])
            )
            self.value_m = nn.Sequential(
                nn.Conv2d(input_dim, input_dim, 1, padding="same"),
                nn.LayerNorm([input_dim, width, width])
            )
            self.z_m = nn.Sequential(
                nn.Conv2d(input_dim, input_dim, 1, padding="same"),
                nn.LayerNorm([input_dim, width, width])
            )

            self.w_z = nn.Sequential(
                nn.Conv2d(input_dim * 2, input_dim * 2, 1, padding="same"),
                nn.LayerNorm([input_dim * 2, width, width])
            )
            self.w = nn.Sequential(
                nn.Conv2d(input_dim * 3, input_dim * 3, 1, padding="same"),
                nn.LayerNorm([input_dim * 3, width, width])
            )

        else:
            # 不使用 LayerNorm 时的层初始化
            self.query_h = nn.Conv2d(input_dim, hidden_dim, 1, padding="same")
            self.key_h = nn.Conv2d(input_dim, hidden_dim, 1, padding="same")
            self.value_h = nn.Conv2d(input_dim, input_dim, 1, padding="same")
            self.z_h = nn.Conv2d(input_dim, input_dim, 1, padding="same")

            self.key_m = nn.Conv2d(input_dim, hidden_dim, 1, padding="same")
            self.value_m = nn.Conv2d(input_dim, input_dim, 1, padding="same")
            self.z_m = nn.Conv2d(input_dim, input_dim, 1, padding="same")

            self.w_z = nn.Conv2d(input_dim * 2, input_dim * 2, 1, padding="same")
            self.w = nn.Conv2d(input_dim * 3, input_dim * 3, 1, padding="same")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, h, m) -> Tuple:
        """
        返回:
            Tuple(torch.Tensor, torch.Tensor): 新的 Hidden 层和新的 Memory 模块.
        """
        batch_size, _, H, W = h.shape

        # hidden attention
        k_h = self.key_h(h)
        q_h = self.query_h(h)
        v_h = self.value_h(h)

        k_h = k_h.view(batch_size, self.hidden_dim, H * W)
        q_h = q_h.view(batch_size, self.hidden_dim, H * W).transpose(1, 2)
        v_h = v_h.view(batch_size, self.input_dim, H * W)

        attention_h = torch.softmax(torch.bmm(q_h, k_h), dim=-1)
        z_h = torch.matmul(attention_h, v_h.permute(0, 2, 1))
        z_h = z_h.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        z_h = self.z_h(z_h)

        # memory attention
        k_m = self.key_m(m)
        v_m = self.value_m(m)

        k_m = k_m.view(batch_size, self.hidden_dim, H * W)
        v_m = v_m.view(batch_size, self.input_dim, H * W)

        attention_m = torch.softmax(torch.bmm(q_h, k_m), dim=-1)
        z_m = torch.matmul(attention_m, v_m.permute(0, 2, 1))
        z_m = z_m.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        z_m = self.z_m(z_m)

        # 通道拼接 Zh 和 Zm
        Z = torch.cat([z_h, z_m], dim=1)
        Z = self.w_z(Z)

        # 通道拼接 Z 和 h
        W = torch.cat([Z, h], dim=1)
        W = self.w(W)

        # 计算 gates 和新的 memory 状态
        mi_conv, mg_conv, mo_conv = torch.chunk(W, chunks=3, dim=1)
        input_gate = torch.sigmoid(mi_conv)
        g = torch.tanh(mg_conv)
        new_M = (1 - input_gate) * m + input_gate * g
        output_gate = torch.sigmoid(mo_conv)
        new_H = output_gate * new_M

        return new_H, new_M, attention_h
