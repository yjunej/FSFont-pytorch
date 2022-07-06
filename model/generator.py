import torch

import torch.nn as nn

from content_encoder import ContentEncoder
from reference_encoder import ReferenceEncoder
from style_aggregation_module import StyleAggregationModule

from einops import rearrange

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.content_encoder = ContentEncoder(input_channels=1)
        self.reference_encoder = ReferenceEncoder(input_channels=1)
        self.sam = StyleAggregationModule(num_heads=8)
    
    def forward(self, content_img, reference_img_list, target_img):
        content_feature_map = self.content_encoder(content_img)
        reference_feature_map_list = [self.reference_encoder(reference_img) for reference_img in reference_img_list]
        target_feature_map = [self.reference_encoder(target_img)]

        style_map = self.sam(content_feature_map, reference_feature_map_list)
        sr_style_map = self.sam(content_feature_map, target_feature_map)

        return style_map, sr_style_map