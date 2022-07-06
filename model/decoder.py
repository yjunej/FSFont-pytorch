import torch.nn as nn

from model.custom_blocks import ResidualBlock, ConvolutionBlock
from torch.nn.utils.parametrizations import spectral_norm

class Decoder(nn.Module):
    def __init__(self, input_channels=256):
        super(Decoder, self).__init__()

        self.res_block_1 = ResidualBlock(in_dim=input_channels,  out_dim=256,   kernel_size=3, normalization='IN', activation='relu', padding=1, stride=1, downsample=None)
        self.res_block_2 = ResidualBlock(in_dim=256,  out_dim=256,   kernel_size=3, normalization='IN', activation='relu', padding=1, stride=1, downsample=None)
        self.res_block_3 = ResidualBlock(in_dim=256,  out_dim=256,   kernel_size=3, normalization='IN', activation='relu', padding=1, stride=1, downsample=None)
    
        self.conv_block_1 = ConvolutionBlock(in_dim=256, out_dim=128, kernel_size=3, normalization='IN', activation='relu', padding=1, stride=1, downsample=None)

        self.conv_block_2 = ConvolutionBlock(in_dim=128, out_dim=64,  kernel_size=3, normalization='IN', activation='relu', padding=1, stride=1, downsample=None)
        self.conv_block_3 = ConvolutionBlock(in_dim=64,  out_dim=32,  kernel_size=3, normalization='IN', activation='relu', padding=1, stride=1, downsample=None)
        self.conv_block_4 = ConvolutionBlock(in_dim=32,  out_dim=1,   kernel_size=3, normalization='IN', activation='relu', padding=1, stride=1, downsample=None)

        self.upsample_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.output_activation = nn.Sigmoid()
    
    def forward(self, x):

        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)

        x = spectral_norm(self.conv_block_1(x))
        x = self.upsample_1(x)
        x = spectral_norm(self.conv_block_2(x))
        x = self.upsample_2(x)
        x = spectral_norm(self.conv_block_3(x))
        x = self.upsample_3(x)
        x = spectral_norm(self.conv_block_4(x))

        out = self.output_activation(x)

        return out

        


