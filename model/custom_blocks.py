import torch.nn as nn


def get_activation(activation):
    if activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'relu':
        return nn.ReLU(inplace=True)
    else:
        raise ValueError("Undefined Activation")
    

def get_downsample(downsample):
    if downsample == 'avgpool':
        return nn.AvgPool2d((2,2))
    else:
        raise ValueError("Undefined Downsample")


def get_normalization(normalization, num_features):
    if normalization == 'IN':
        return nn.InstanceNorm2d(num_features)
    else:
        raise ValueError("Undefined Normalization")


class ConvolutionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, normalization, activation, padding, stride, downsample=None):
        super(ConvolutionBlock, self).__init__()
        self.normalization = get_normalization(normalization=normalization, num_features=in_dim)
        self.activation = get_activation(activation=activation)
        if downsample is not None:
            self.downsample = get_downsample(downsample=downsample)
        self.convolution = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding, stride=stride)
        self.downsampling = False if downsample is None else True

    def forward(self,x):
        x = self.normalization(x)
        x = self.activation(x)
        x = self.convolution(x)
        if self.downsampling:
            x = self.downsample(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, normalization, activation, padding, stride, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv_blocks = nn.Sequential(
            ConvolutionBlock(in_dim, out_dim, kernel_size, normalization, activation, padding, stride, downsample=None),
            ConvolutionBlock(out_dim, out_dim, kernel_size, normalization, activation, padding, stride, downsample=None)
        )
        if downsample is not None:
            self.downsample = get_downsample(downsample)
        self.downsampling = False if downsample is None else True
        self.channel_fit_conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.shortcut_channel_fit = True if in_dim != out_dim else False
        
    def forward(self, x):
        _x = x

        if self.shortcut_channel_fit:
            x = self.channel_fit_conv(x)
        
        _x = self.conv_blocks(_x)

        if self.downsampling:
            x = self.downsample(x)
            _x = self.downsample(_x)

        out = x + _x
        
        return out
        
        
        