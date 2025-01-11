import torch
import torch.nn as nn
from core.layers.CausalLSTMCell import CausalLSTMCell
class RNNpp(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNNpp, self).__init__()

        self.configs = configs
        self.input_dim = configs.patch_size * configs.patch_size * configs.img_channel
        self.hidden_list = num_hidden
        self.num_layers = num_layers
        self.frame_size = configs.img_width // configs.patch_size
        cell_list = []

        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.input_dim if i == 0 else self.hidden_list[i - 1]
            cell_list.append(
                CausalLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.hidden_list[self.num_layers - 1], self.input_dim,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]

        # print("freames",frames.size())
        batch = frames.shape[0]
        seq=frames.shape[1]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.hidden_list[i], height, width]).to(DEVICE)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.hidden_list[0], height, width]).to(DEVICE)

        for t in range(seq - 1):
            if t < 10:
                net = frames[:, t]
            else:
                net = mask_true[:, t - 10] * frames[:, t] + \
                      (1 - mask_true[:, t - 10]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                # print(i)
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                # print("完成")

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2,3,4).contiguous()
        # loss = self.MSE_criterion(next_frames, frames[:, 1:])
        # print("每一个的loss",loss)
        return next_frames