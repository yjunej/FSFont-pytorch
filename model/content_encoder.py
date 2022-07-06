import torch
import torch.nn as nn

from model.custom_blocks import ConvolutionBlock

class ContentEncoder(nn.Module):
    def __init__(self, input_channels=1):
        super(ContentEncoder, self).__init__()
        
        self.conv_block_1 = ConvolutionBlock(in_dim=input_channels,  out_dim=32,   kernel_size=3, normalization='IN', activation='relu', padding=1, stride=1, downsample=None)
        self.conv_block_2 = ConvolutionBlock(in_dim=32, out_dim=64,   kernel_size=3, normalization='IN', activation='relu', padding=1, stride=2, downsample=None)
        self.conv_block_3 = ConvolutionBlock(in_dim=64, out_dim=128,  kernel_size=3, normalization='IN', activation='relu', padding=1, stride=2, downsample=None)
        self.conv_block_4 = ConvolutionBlock(in_dim=128, out_dim=256, kernel_size=3, normalization='IN', activation='relu', padding=1, stride=2, downsample=None)
        self.conv_block_5 = ConvolutionBlock(in_dim=256, out_dim=256, kernel_size=3, normalization='IN', activation='relu', padding=1, stride=1, downsample=None)
    
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        out = self.conv_block_5(x)

        return out


    
    
    

    