import torch

import torch.nn as nn

from einops import rearrange


class StyleAggregationModule(nn.Module):
    def __init__(self, num_heads=8):
        super(StyleAggregationModule, self).__init__()

        self.num_heads = num_heads
        self.q_proj = nn.Linear(512, 512 * num_heads)
        self.k_proj = nn.Linear(512, 512 * num_heads)
        self.v_proj = nn.Linear(512, 512 * num_heads)
        self.softmax = nn.Softmax(dim=-1)
        self.multihead_concat_fc = nn.Linear(512 * num_heads, 512)


    def forward(self, content_feature_map, reference_map_list):
        B, C, H, W = content_feature_map.shape
        K = len(reference_map_list)

        content_sequence = rearrange(content_feature_map, 'b c h w -> b (h w) c')
        q = self.q_proj(content_sequence) # b (h w) (c m)
        q = rearrange(q, 'b (h w) (c m) -> (b m) (h w) c', h=H, m=self.num_heads)

        reference_map = torch.stack(reference_map_list, axis=1) # B, K, C, H, W
        reference_sequence = rearrange(reference_map, 'b k c h w -> b (k h w) c')
        k = self.k_proj(reference_sequence) # b (k h w) (c m)
        k = rearrange(k, 'b (k h w) (c m) -> (b m) (k h w) c', m=self.num_heads, k=K, h=H)

        v = self.v_proj(reference_sequence) # b (k h w) (c m)
        v = rearrange(v, 'b (k h w) (c m) -> (b m) (k h w) c', m=self.num_heads, k=K, h=H)

        score = torch.bmm(q, k.permute(0,2,1)) # (b m) (h w) (k h w)
        score /= C ** (self.num_heads / 2) 
        attention_score = self.softmax(score)
        multihead_result = torch.bmm(attention_score, v) # (b m) (h w) c
        multihead_result = rearrange(multihead_result, '(b m) (h w) c -> b m (h w) c', m=self.num_heads, h=H, w=W, c=C)
        multihead_result = torch.cat([multihead_result[:,i,:] for i in range(self.num_heads)], axis=-1)
        result = self.multihead_concat_fc(multihead_result)
        result = rearrange(result, 'b (h w) c -> b c h w', h=H, c=C)
        result = torch.cat([result, content_feature_map], axis=1) # b 2c h w

        return result









        