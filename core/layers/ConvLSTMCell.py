class ConvLSTMCell(nn.Module):
    def __init__(
            self, in_channel, num_hidden, width, filter_size, stride, layer_norm
    ) -> None:
        super(ConvLSTMCell, self).__init__()

        self.input_channels = in_channel
        self.hidden_channels = num_hidden
        self.kernel_size = kernel_size
        self.H = width
        self.W = width

        self.padding = filter_size // 2

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, padding=self.padding,stride=stride)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, padding=self.padding,stride=stride)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, padding=self.padding,stride=stride)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, padding=self.padding,stride=stride)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, padding=self.padding,stride=stride)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, padding=self.padding,stride=stride)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, padding=self.padding,stride=stride)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, padding=self.padding,stride=stride)

        self.Wci = nn.parameter.Parameter(torch.zeros(self.hidden_channels, self.H,self.W, dtype=torch.float,requires_grad=True)).to(DEVICE)
        self.Wcf = nn.parameter.Parameter(torch.zeros(self.hidden_channels, self.H,self.W, dtype=torch.float,requires_grad=True)).to(DEVICE)
        self.Wco = nn.parameter.Parameter(torch.zeros(self.hidden_channels, self.H,self.W, dtype=torch.float,requires_grad=True)).to(DEVICE)

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc