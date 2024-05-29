# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import torch.nn.functional as F


class HoverFast(nn.Module):
    """
    Implementation of HoverFast based on the HoverNet and U-net architectures.

    HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in 
    Multi-Tissue Histology Images (Graham et al., 2019).
    Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015).
    MSU-Net: Multi-Scale U-Net for 2D Medical Image Segmentation.

    This model uses convolutional blocks from MSU-net and supports various configurations.

    Parameters:
    in_channels (int): Number of input channels.
    n_classes (int): Number of output channels.
    depth (int): Depth of the network.
    wf (int): Number of filters in the first layer is 2**wf.
    padding (bool): If True, apply padding such that the input shape is the same as the output.
    batch_norm (bool): Use BatchNorm after layers with an activation function.
    up_mode (str): One of 'upconv' or 'upsample'. 'upconv' uses transposed convolutions for learned upsampling. 'upsample' uses bilinear upsampling.
    conv_block (str): One of 'unet' or 'msunet'. Specifies which model's convolutional block to use.
    """
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False,
                 batch_norm=False, up_mode='upconv',conv_block="msunet"):

        super(HoverFast, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        assert conv_block in ('unet', 'msunet')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append((MSUNetConvBlock if conv_block=='msunet' else UNetConvBlock)(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
            prev_channels = 2**(wf+i)
        
        temp=prev_channels

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm, conv_block))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)
        
        prev_channels=temp
        
        self.up_path_m = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_m.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm, conv_block))
            prev_channels = 2**(wf+i)

        self.last_m = nn.Conv2d(prev_channels, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        """
        Forward pass through the HoverFast model.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        tuple[torch.Tensor, torch.Tensor]: Output tensors from the main and auxiliary paths.
        """

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
    """
    Basic convolutional block for U-Net.

    Consists of two convolutional layers with ReLU activation, with optional BatchNorm.

    Parameters:
    in_size (int): Number of input channels.
    out_size (int): Number of output channels.
    padding (bool): If True, apply padding such that the input shape is the same as the output.
    batch_norm (bool): Use BatchNorm after layers with an activation function.
    kernel (int): Size of the convolutional kernel. Default is 3.
    """
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
        """
        Forward pass through the UNetConvBlock.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor.
        """
        out = self.block(x)
        return out

class MSUNetConvBlock(nn.Module):
    """
    Multi-Scale U-Net convolutional block.

    Consists of two parallel convolutional paths with different kernel sizes,
    followed by concatenation and a final convolution.

    Parameters:
    ch_in (int): Number of input channels.
    ch_out (int): Number of output channels.
    padding (bool): If True, apply padding such that the input shape is the same as the output.
    batch_norm (bool): Use BatchNorm after layers with an activation function.
    """

    def __init__(self, ch_in, ch_out, padding, batch_norm):
        super(MSUNetConvBlock, self).__init__()
        self.conv_3 = UNetConvBlock(ch_in, ch_out,padding,batch_norm,kernel=3)
        self.conv_7 = UNetConvBlock(ch_in, ch_out,3*padding,batch_norm,kernel=7)
        self.conv = nn.Conv2d(ch_out * 2, ch_out, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the MSUNetConvBlock.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor.
        """

        x3 = self.conv_3(x)
        x7 = self.conv_7(x)
        x = torch.cat((x3, x7), dim=1)
        x = self.conv(x)
        return x

class UNetUpBlock(nn.Module):
    """
    U-Net upsampling block.

    Consists of an upsampling layer followed by a convolutional block.

    Parameters:
    in_size (int): Number of input channels.
    out_size (int): Number of output channels.
    up_mode (str): One of 'upconv' or 'upsample'. 'upconv' uses transposed convolutions for learned upsampling. 'upsample' uses bilinear upsampling.
    padding (bool): If True, apply padding such that the input shape is the same as the output.
    batch_norm (bool): Use BatchNorm after layers with an activation function.
    conv_block (str): One of 'unet' or 'msunet'. Specifies which model's convolutional block to use.
    """

    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, conv_block):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = (MSUNetConvBlock if conv_block=='msunet' else UNetConvBlock)(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer: torch.Tensor, target_size: list[int]) -> torch.Tensor:
        """
        Center crop a tensor to a target size.

        Parameters:
        layer (torch.Tensor): Input tensor to be cropped.
        target_size (list[int]): Target size [height, width].

        Returns:
        torch.Tensor: Cropped tensor.
        """

        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x: torch.Tensor, bridge: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNetUpBlock.

        Parameters:
        x (torch.Tensor): Input tensor.
        bridge (torch.Tensor): Tensor from the corresponding downsampling block.

        Returns:
        torch.Tensor: Output tensor.
        """
                
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out