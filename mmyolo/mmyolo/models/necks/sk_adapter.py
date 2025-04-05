import torch.nn as nn
from mmyolo.registry import MODELS

@MODELS.register_module()
class SKAdapter(nn.Module):
    """添加SKAttention的适配器Neck"""
    
    def __init__(self,
                 in_channels=256,
                 reduction=8):
        super().__init__()
        # 获取SKAttention模块
        self.sk_attention = MODELS.build(
            dict(type='SKAttention', 
                 in_channels=in_channels,
                 reduction=reduction))
    
    def forward(self, inputs):
        """Forward function"""
        assert isinstance(inputs, list)
        outs = []
        for x in inputs:
            out = self.sk_attention(x)
            outs.append(out)
        return tuple(outs)