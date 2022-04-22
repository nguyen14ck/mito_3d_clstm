import torch.nn as nn
from torch.nn.modules.conv import Conv2d
from .decoder.modules import Flatten, Activation



class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)

# class SegmentationHead(nn.Module):

#     def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
#         super().__init__()
#         # conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
#         # upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
#         self.conv2dup = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,  kernel_size=kernel_size, padding=kernel_size//2, stride=2)
#         self.activation = Activation(activation)
#         # super().__init__(conv2d, upsampling, activation)

#     def forward(self, x):
#         new_size = list(x.shape)
#         new_size[-1] *= 2
#         new_size[-2] *= 2
#         x = self.conv2dup(x, output_size=new_size)
#         x = self.activation(x)
#         return x

class SegmentationHeadTranspose(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        super().__init__()
        # conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        # upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.conv2dup = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,  kernel_size=kernel_size, padding=kernel_size//2, stride=2)
        self.activation = Activation(activation)
        # super().__init__(conv2d, upsampling, activation)

    def forward(self, x):
        new_size = list(x.shape)
        new_size[-1] *= 2
        new_size[-2] *= 2
        x = self.conv2dup(x, output_size=new_size)
        x = self.activation(x)
        return x
        

class SegmentationHeadTranspose_2(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        # upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.conv2dup = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels,  kernel_size=kernel_size, padding=kernel_size//2, stride=2)
        self.activation = Activation(activation)
        # super().__init__(conv2d, upsampling, activation)

    def forward(self, x):
        new_size = list(x.shape)
        new_size[-1] *= 2
        new_size[-2] *= 2
        x = self.conv2d(x)
        x = self.conv2dup(x, output_size=new_size)
        x = self.activation(x)
        return x


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)

class SegmentationHead3D(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        # upsampling = nn.UpsamplingBilinear3d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        upsampling = nn.Upsample(scale_factor=upsampling, mode='trilinear', align_corners=True) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv3d, upsampling, activation)


class SegmentationHeadTranspose3D_2(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        # upsampling = nn.UpsamplingBilinear3d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        # upsampling = nn.Upsample(scale_factor=upsampling, mode='trilinear', align_corners=True) if upsampling > 1 else nn.Identity()

        self.conv3dup = nn.ConvTranspose3d(in_channels=out_channels, out_channels=out_channels,  kernel_size=kernel_size, padding=kernel_size//2, stride=2)
        
        self.activation = Activation(activation)
        # super().__init__(conv3d, upsampling, activation)

    def forward(self, x):
        new_size = list(x.shape)
        new_size[-1] *= 2
        new_size[-2] *= 2
        x = self.conv3d(x)
        x = self.conv3dup(x, output_size=new_size)
        x = self.activation(x)
        return x


class ClassificationHead3D(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool3d(1) if pooling == 'avg' else nn.AdaptiveMaxPool3d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)


