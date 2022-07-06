import torch

import torch.nn as nn

from model.custom_blocks import ResidualBlock, ConvolutionBlock


class Discriminator(nn.Module):
    def __init__(self, input_channels=1, num_content_labels=11172, num_style_labels=5000):
        super(Discriminator, self).__init__()

        self.conv_block_1 = ConvolutionBlock(in_dim=input_channels, out_dim=32,  kernel_size=3, normalization='IN', activation='relu', padding=1, stride=2, downsample=None)

        self.res_block_1 = ResidualBlock(in_dim=32,  out_dim=64,   kernel_size=3, normalization='IN', activation='relu', padding=1, stride=1, downsample='avgpool')
        self.res_block_2 = ResidualBlock(in_dim=64,  out_dim=128,   kernel_size=3, normalization='IN', activation='relu', padding=1, stride=1, downsample='avgpool')
        self.res_block_3 = ResidualBlock(in_dim=128,  out_dim=256,   kernel_size=3, normalization='IN', activation='relu', padding=1, stride=1, downsample='avgpool')
        self.res_block_4 = ResidualBlock(in_dim=256,  out_dim=256,   kernel_size=3, normalization='IN', activation='relu', padding=1, stride=1, downsample=None)
        self.res_block_5 = ResidualBlock(in_dim=256,  out_dim=512,   kernel_size=3, normalization='IN', activation='relu', padding=1, stride=1, downsample=None)

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.content_projection_embed = nn.Embedding(num_content_labels, 512)
        self.style_projection_embed = nn.Embedding(num_style_labels, 512)
    
        self.content_fc = nn.Linear(512, 1)
        self.style_fc = nn.Linear(512, 1)

    def forward(self, x, content_y, style_y):

        x = self.conv_block_1(x)

        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.res_block_4(x)
        x = self.res_block_5(x)

        x = self.adaptive_avg_pool(x).flatten(start_dim=1)

        content_out = self.content_fc(x)
        content_embed = self.content_projection_embed(content_y)
        content_out = content_out + torch.sum(x * content_embed, dim=1, keepdims=True)

        style_out = self.style_fc(x)
        style_embed = self.style_projection_embed(style_y)
        style_out = style_out + torch.sum(x * style_embed, dim=1, keepdims=True)

        return content_out, style_out

        


