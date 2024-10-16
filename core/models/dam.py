import torch
import torch.nn as nn
from core.layers.DamCell import DamCell

class dam(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(dam, self).__init__()

        self.configs = configs
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel

        # 计算宽度和高度
        width = configs.img_width // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        # 初始化 ns_sam_conv_cell 列表
        self.cell_list = nn.ModuleList([
            DamCell(
                in_channel=self.frame_channel if i == 0 else num_hidden[i - 1],
                num_hidden=num_hidden[i],
                width=width,
                kernel_size=configs.filter_size,
                stride=configs.stride,
                layer_norm=configs.layer_norm
            ) for i in range(num_layers)
        ])

        # 定义输出卷积层
        self.conv_last = nn.Conv2d(
            in_channels=num_hidden[-1],
            out_channels=self.frame_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, frames_tensor, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        pre_net = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)


        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)


        for t in range(self.configs.total_length - 1):
            # reverse schedule sampling
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen

            else:
                if t < self.configs.input_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs.input_length]) * x_gen
            if t == 0:
                pre = net.clone().detach()
                pre_net.append(pre)
                h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)
                pre = h_t[0].clone().detach()
                pre_net.append(pre)
                for i in range(1, self.num_layers):
                    h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                    if i < self.num_layers - 1:
                        pre = h_t[i].clone().detach()
                        pre_net.append(pre)
            else:
                pre = frames[:, t - 1].clone().detach()
                pre_net[0] = pre
                if t < self.configs.input_length:
                    pred_net = pre_net[0]
                else:
                    # net=net.float()
                    pred_net = mask_true[:, t - self.configs.input_length] * pre_net[0] + (
                                1 - mask_true[:, t - self.configs.input_length]) * next_frames[-2]
                h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory,pred_net)
                for i in range(1, self.num_layers):
                    # print("layer:", i)
                    h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory,pre_net[i])
                    if i < self.num_layers - 1:
                        pre = h_t[i - 1].clone().detach()
                        pre_net[i] = pre



            # 生成下一帧
            x_gen = self.conv_last(h_t[-1])
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        return next_frames, loss
