import torch
import torch.nn as nn

from model.decoder import Decoder
from model.content_encoder import ContentEncoder
from model.reference_encoder import ReferenceEncoder
from model.style_aggregation_module import StyleAggregationModule

class Generator(nn.Module):
    def __init__(self, input_channels=1, latent_hidden_dim=256, latent_output_dim=512):
        super(Generator, self).__init__()
        self.content_encoder = ContentEncoder(input_channels=input_channels)
        self.reference_encoder = ReferenceEncoder(input_channels=input_channels)
        self.sam = StyleAggregationModule(num_heads=8, hidden_dim=latent_hidden_dim)
        self.decoder = Decoder(input_channels=latent_output_dim)
    
    def forward(self, content_img, reference_img_list, target_img):
        content_feature_map = self.content_encoder(content_img)
        reference_feature_map_list = [self.reference_encoder(reference_img) for reference_img in reference_img_list]
        target_feature_map = [self.reference_encoder(target_img)]

        style_map = self.sam(content_feature_map, reference_feature_map_list)
        sr_style_map = self.sam(content_feature_map, target_feature_map)
        
        output_img = self.decoder(style_map)
        reconstructed_img = self.decoder(sr_style_map)

        return output_img, reconstructed_img
