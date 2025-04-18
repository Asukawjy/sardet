import torch
import torch.nn.functional as F
from timm.models.layers.create_act import create_act_layer, get_act_layer
from timm.models.layers.helpers import make_divisible
from timm.models.layers.mlp import ConvMlp
from timm.models.layers.norm import LayerNorm2d
from torch import nn as nn
from mmyolo.registry import MODELS
 
@MODELS.register_module()
class GlobalContext(nn.Module):
 
    def __init__(self, in_channels, use_attn=True, fuse_add=False, fuse_scale=True, init_last_zero=False,
                 rd_ratio=1./8, rd_channels=None, rd_divisor=1, act_layer=nn.ReLU, gate_layer='sigmoid'):
        super(GlobalContext, self).__init__()
        act_layer = get_act_layer(act_layer)
 
        self.conv_attn = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True) if use_attn else None
 
        if rd_channels is None:
            rd_channels = make_divisible(in_channels * rd_ratio, rd_divisor, round_limit=0.)
        if fuse_add:
            self.mlp_add = ConvMlp(in_channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
        else:
            self.mlp_add = None
        if fuse_scale:
            self.mlp_scale = ConvMlp(in_channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
        else:
            self.mlp_scale = None
 
        self.gate = create_act_layer(gate_layer)
        self.init_last_zero = init_last_zero
        self.reset_parameters()
 
    def reset_parameters(self):
        if self.conv_attn is not None:
            nn.init.kaiming_normal_(self.conv_attn.weight, mode='fan_in', nonlinearity='relu')
        if self.mlp_add is not None:
            nn.init.zeros_(self.mlp_add.fc2.weight)
 
    def forward(self, x):
        B, C, H, W = x.shape
 
        if self.conv_attn is not None:
            attn = self.conv_attn(x).reshape(B, 1, H * W)  # (B, 1, H * W)
            attn = F.softmax(attn, dim=-1).unsqueeze(3)  # (B, 1, H * W, 1)
            context = x.reshape(B, C, H * W).unsqueeze(1) @ attn
            context = context.view(B, C, 1, 1)
        else:
            context = x.mean(dim=(2, 3), keepdim=True)
 
        if self.mlp_scale is not None:
            mlp_x = self.mlp_scale(context)
            x = x * self.gate(mlp_x)
        if self.mlp_add is not None:
            mlp_x = self.mlp_add(context)
            x = x + mlp_x
 
        return x
 
if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    gc = GlobalContext(512)
    output=gc(input)
    print(output.shape)