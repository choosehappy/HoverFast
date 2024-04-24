# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import torch.nn.functional as F


class HoverUNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False,
                 batch_norm=False, up_mode='upconv'):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(HoverUNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(MSUNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
            prev_channels = 2**(wf+i)
        
        temp=prev_channels

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)
        
        prev_channels=temp
        
        self.up_path_m = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_m.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.last_m = nn.Conv2d(prev_channels, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)
                
        y=x.clone()

        for i, up in enumerate(self.up_path_m):
            y = up(y, blocks[-i-1])

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])
        

        return self.last(x),self.last_m(y)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, kernel=3):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=kernel,
                               padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        block.append(nn.ReLU())

        block.append(nn.Conv2d(out_size, out_size, kernel_size=kernel,
                               padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        block.append(nn.ReLU())
        
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return out

class MSUNetConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, padding, batch_norm):
        super(MSUNetConvBlock, self).__init__()
        self.conv_3 = UNetConvBlock(ch_in, ch_out,padding,batch_norm,kernel=3)
        self.conv_7 = UNetConvBlock(ch_in, ch_out,3*padding,batch_norm,kernel=7)
        self.conv = nn.Conv2d(ch_out * 2, ch_out, kernel_size=1)

    def forward(self, x):
        x3 = self.conv_3(x)
        x7 = self.conv_7(x)
        x = torch.cat((x3, x7), dim=1)
        x = self.conv(x)
        return x


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = MSUNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer: torch.Tensor, target_size: list[int]) -> torch.Tensor:
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x: torch.Tensor, bridge: torch.Tensor) -> torch.Tensor:
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
