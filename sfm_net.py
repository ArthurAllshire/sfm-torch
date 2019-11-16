import torch
import torch.nn as nn
import torch.nn.functional as F

camera_intrinsics = (0.5, 0.5, 1.0)
structure_image_sz = (384, 128)
n_conv_layers = 11
n_deconv_layers = 5
kernel_size = 3
n_deconv_layers = 5
base_channels = 32

class SFMConvNet(nn.Module):
    """ConvNet."""
    def __init__(self, input_channels, use_skips=True, ret_conv=False):
        super(SFMConvNet, self).__init__()
        
        self.use_skips = use_skips
        self.ret_conv = ret_conv
        
        # number of channels goes from input_channles in onput to
        channels_conv_layers = [input_channels] + [base_channels * 2**(i//2) for i in range(1, n_conv_layers+1)]
        strides_per_layer = [1 if i%2 == 0 else 2 for i in range(n_conv_layers)]
        
        self.conv = [
            nn.Conv2d(channels_conv_layers[i],
                      channels_conv_layers[i+1],
                      kernel_size,
                      padding=1,
                      stride=strides_per_layer[i])
            for i in range(n_conv_layers)
        ]
        
        # number of channels goes from 1024 on input to 32 (base_channels) on output
        channels_deconv_layers = [base_channels * 2**i for i in reversed(range(n_deconv_layers+1))]
        self.deconv = [
            nn.ConvTranspose2d(channels_deconv_layers[i],
                               channels_deconv_layers[i+1],
                               kernel_size,
                               padding=1,
                               output_padding=1,
                               stride=2)
            for i in range(n_deconv_layers)
        ]
        
    
    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.conv):
            x = F.relu(layer(x))
            # store intermediate representations to use in skip connections later
            # only store every second layer (before dimenson reduction from stride 2)
            # and don't store the last layer
            if self.use_skips and i%2 == 0 and not i == len(self.conv)-1:
                skips.append(x)
        skips.reverse()

        if self.ret_conv:
            conv_out = x
        for i, layer in enumerate(self.deconv):
            x = layer(x)
            x += skips[i]
        
        if self.ret_conv:
            return x, conv_out
        return x

class StructureNet(nn.Module):
    """StructureNet from the paper."""
    
    def __init__(self, input_channels):
        super(StructureNet, self).__init__()
        
        self.conv_net = SFMConvNet(input_channels, use_skips=True, ret_conv=False)
        
        self.linear = nn.Linear(base_channels, 1)
    
    def forward(self, x):
        input_size = x.size()
        
        x = self.conv_net(x)
        x = x.view(-1, *input_size[2:4], base_channels)
        x = F.relu(self.linear(x))
        x = x.view(-1, 1, *input_size[2:4])
        return x