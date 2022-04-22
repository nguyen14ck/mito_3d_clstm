# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks import ADN
from monai.networks.layers.convutils import same_padding, stride_minus_kernel_padding
from monai.networks.layers.factories import Conv, replication_pad_factory

# from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
# from monai.utils import alias, export

from ._utils import pad_tensor_list_equal
import re
# from copy import deepcopy
# import torch
# import torch.nn as nn
import torch.nn.functional as Fn
# from torchvision.models.resnet import ResNet
# from torchvision.models.resnet import BasicBlock
# from torchvision.models.resnet import Bottleneck
# from pretrainedmodels.models.torchvision_models import pretrained_settings
from ._base import EncoderMixin


class Convolution(nn.Sequential):
    """
    Constructs a convolution with normalization, optional dropout, and optional activation layers::

        -- (Conv|ConvTrans) -- (Norm -- Dropout -- Acti) --

    if ``conv_only`` set to ``True``::

        -- (Conv|ConvTrans) --

    Args:
        dimensions: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 3.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: determine the dimensions of dropout. Defaults to 1.

            - When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            - When dropout_dim = 2, Randomly zeroes out entire channels (a channel is a 2D feature map).
            - When dropout_dim = 3, Randomly zeroes out entire channels (a channel is a 3D feature map).

            The value of dropout_dim should be no no larger than the value of `dimensions`.
        dilation: dilation rate. Defaults to 1.
        groups: controls the connections between inputs and outputs. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        conv_only: whether to use the convolutional layer only. Defaults to False.
        is_transposed: if True uses ConvTrans instead of Conv. Defaults to False.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension. Defaults to None.
        output_padding: controls the additional size added to one side of the output shape.
            Defaults to None.

    See also:

        :py:class:`monai.networks.layers.Conv`
        :py:class:`monai.networks.blocks.ADN`

    """

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        strides: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        adn_ordering: str = "NDA",
        act: Optional[Union[Tuple, str]] = "PRELU",
        norm: Optional[Union[Tuple, str]] = "INSTANCE",
        dropout: Optional[Union[Tuple, str, float]] = None,
        dropout_dim: Optional[int] = 1,
        dilation: Union[Sequence[int], int] = 1,
        groups: int = 1,
        bias: bool = True,
        conv_only: bool = False,
        is_transposed: bool = False,
        padding: Optional[Union[Sequence[int], int]] = None,
        output_padding: Optional[Union[Sequence[int], int]] = None,
    ) -> None:
        super().__init__()
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_transposed = is_transposed
        if padding is None:
            padding = same_padding(kernel_size, dilation)
        conv_type = Conv[Conv.CONVTRANS if is_transposed else Conv.CONV, dimensions]

        if is_transposed:
            if output_padding is None:
                output_padding = stride_minus_kernel_padding(1, strides)
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                bias=bias,
                dilation=dilation,
            )
        else:
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )

        self.add_module("conv", conv)

        if not conv_only:
            self.add_module(
                "adn",
                ADN(
                    ordering=adn_ordering,
                    in_channels=out_channels,
                    act=act,
                    norm=norm,
                    norm_dim=dimensions,
                    dropout=dropout,
                    dropout_dim=dropout_dim,
                ),
            )


class ResidualUnit(nn.Module):
    """
    Residual module with multiple convolutions and a residual connection.

    Args:
        dimensions: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 3.
        subunits: number of convolutions. Defaults to 2.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: determine the dimensions of dropout. Defaults to 1.

            - When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            - When dropout_dim = 2, Randomly zero out entire channels (a channel is a 2D feature map).
            - When dropout_dim = 3, Randomly zero out entire channels (a channel is a 3D feature map).

            The value of dropout_dim should be no no larger than the value of `dimensions`.
        dilation: dilation rate. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        last_conv_only: for the last subunit, whether to use the convolutional layer only.
            Defaults to False.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension. Defaults to None.

    See also:

        :py:class:`monai.networks.blocks.Convolution`

    """

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        strides: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        subunits: int = 2,
        adn_ordering: str = "NDA",
        act: Optional[Union[Tuple, str]] = "PRELU",
        norm: Optional[Union[Tuple, str]] = "INSTANCE",
        dropout: Optional[Union[Tuple, str, float]] = None,
        dropout_dim: Optional[int] = 1,
        dilation: Union[Sequence[int], int] = 1,
        bias: bool = True,
        last_conv_only: bool = False,
        padding: Optional[Union[Sequence[int], int]] = None,
        ratio=1,
    ) -> None:
        super().__init__()
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.conv = nn.Sequential()
        self.conv = nn.ModuleList()
        self.residual = nn.Identity()
        if not padding:
            padding = same_padding(kernel_size, dilation)
        schannels = in_channels
        sstrides = strides
        subunits = max(1, subunits)
        self.ratio = ratio

        for su in range(subunits):
            conv_only = last_conv_only and su == (subunits - 1)
            unit = Convolution(
                dimensions,
                schannels,
                out_channels,
                strides=sstrides,
                kernel_size=kernel_size,
                adn_ordering=adn_ordering,
                act=act,
                norm=norm,
                dropout=dropout,
                dropout_dim=dropout_dim,
                dilation=dilation,
                bias=bias,
                conv_only=conv_only,
                padding=padding,
            )

            # self.conv.add_module(f"unit{su:d}", unit)
            self.conv.append(unit)

            # after first loop set channels and strides to what they should be for subsequent units
            schannels = out_channels
            sstrides = 1

        # apply convolution to input to change number of output channels and size to match that coming from self.conv
        if np.prod(strides) != 1 or in_channels != out_channels:
            rkernel_size = kernel_size
            rpadding = padding

            if np.prod(strides) == 1:  # if only adapting number of channels a 1x1 kernel is used with no padding
                rkernel_size = 1
                rpadding = 0

            conv_type = Conv[Conv.CONV, dimensions]
            self.residual = conv_type(in_channels, out_channels, rkernel_size, strides, rpadding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x.shape
        # torch.Size([1, 1, 224, 224, 80])
        # res.shape
        # torch.Size([1, 32, 112, 112, 40])
        # self.residual
        # Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        # self.conv[i]
        # Convolution(
        # (conv): Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        # (adn): ADN(
        #     (N): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        #     (D): Dropout(p=0.0, inplace=False)
        #     (A): PReLU(num_parameters=1)
        # )
        # )
        # x.shape
        # torch.Size([1, 32, 112, 112, 40])
        # self.conv[i]
        # Convolution(
        # (conv): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        # (adn): ADN(
        #     (N): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        #     (D): Dropout(p=0.0, inplace=False)
        #     (A): PReLU(num_parameters=1)
        # )
        # )
        # x.shape
        # torch.Size([1, 32, 112, 112, 40])





        output_list = []
        res: torch.Tensor = self.residual(x)  # create the additive residual from x
        # cx: torch.Tensor = self.conv(x)  # apply x to sequence of operations
        for i in range (len(self.conv)):
            x = self.conv[i](x)
            output_list.append(x)

        output_list = pad_tensor_list_equal(output_list, combine=2)
        keep_len = round(output_list.shape[0] * self.ratio)
        if self.ratio > 0:
            output_list = output_list[:keep_len]
        else:
            output_list = output_list[-keep_len:]


        output = x + res # add the residual to the output

        return output, output_list


class EncoderMonaiCLstm(nn.Module, EncoderMixin):
    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #     # self._out_channels = out_channels
    #     # self._depth = depth
    #     # self._in_channels = 3
        
    #     self.model = None
    #     # del self.classifier


    def __init__(self, backbone_name, dimensions, in_channels, out_channels, channels, strides,
                        kernel_size=3, num_res_units=3, act='PRELU', norm=None, dropout=0, ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.spatial_dims = None
        self.block_config = None
        self.classify_classes = None

        # self._out_channels = out_channels
        # self._depth = depth
        # self._in_channels = 3
        
        self.model = None
        # del self.classifier

        self.backbone_name = backbone_name
        # self.layer_list = layer_list

        self.device = torch.device("cuda") # all devices

        # MONAI
        # model = monai.networks.nets.UNet(
        #         dimensions=3,
        #         in_channels=1,
        #         out_channels=3,
        #         channels=(16, 32, 64, 128, 256),
        #         strides=(2, 2, 2, 2),
        #         num_res_units=3,
        #     )
        self.dimensions = dimensions
        self.in_channels = in_channels
        # self.out_channels = out_channels # for decoder
        self.channels = channels #layer list
        self.strides = strides
        self.kernel_size = kernel_size
        # self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.ratio = ratio
        
    

        self.layers = []
        for i in range(len(channels)):
            if i == 0:
                conv = self._get_down_layer(in_channels, channels[i], strides[i], is_top=False)
            elif i < len(channels) - 1:
                conv = self._get_down_layer(channels[i-1], channels[i], strides[i], is_top=False)  # create layer in downsampling path
            else:
                # conv = self._get_down_layer(channels[i-1], channels[i], 1, is_top=False) # bottom layer
                conv = self._get_down_layer(channels[i-1], channels[i], 2, is_top=False) # for compatibility
            self.layers.append(conv)

        self.layers = nn.ModuleList(self.layers) 


    
    
    def _get_down_layer(
        self, in_channels: int, out_channels: int, strides: int, is_top: bool
    ) -> Union[ResidualUnit, Convolution]:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                ratio=self.ratio,
            )
        else:
            return Convolution(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )



    def make_dilated(self, stage_list, dilation_list):
        raise ValueError("DenseNet encoders do not support dilated mode "
                         "due to pooling operation for downsampling!")


    def get_stages(self):
        return [
            [nn.Identity()],
            [self.model.layer0.conv1, self.model.layer0.bn1, self.model.layer0.relu1],
            [self.model.layer0.pool, self.model.layer1],
            [self.model.layer2],
            [self.model.layer3],
            [self.model.layer4],
        ]


    def forward(self, x):

        layer_features = [[torch.tensor(1)]] # add idenity layer for compatibility
        child_features = [[] for _ in range(len(self.layers))]

        for i, l in enumerate(self.layers):
            x, child_features[i] = self.layers[i](x)
                       
            # Resnet
            layer_features.append(x)

            # if len(child_features[i])>1: # ALREADY DID AT SUB-LAYER
            #     # child_features[i] = torch.nn.utils.rnn.pad_sequence(child_features[i], batch_first=False, padding_value=0).permute(1,2,0,3,4)
            #     child_features[i] = pad_tensor_list_equal(child_features[i], combine=2)

        child_features = [[torch.tensor(1)]] + child_features # add idenity layer for compatibility
                
        return [layer_features, child_features]
        


    def forward_backup(self, x):
        stages = self.get_stages()
        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)
        return features


    def forward_clstm(self, x):
        stages = self.get_stages()
        layer_features = [] # store the final features at each layer output
        
        child_features = [[] for _ in range(self._depth + 2)] # store all the features at each leavel

        for i in range(self._depth + 2):

            for j in range(len(stages[i])):

                if stages[i][j]._get_name() != 'Sequential':
                    x = stages[i][j](x)
                else:
                    for name, child in stages[i][j].named_children():
                        x = child(x)
                        child_features[i].append(Fn.relu(x))
                       
            # # Densenet
            # if isinstance(x, (list, tuple)):
            #     x, skip = x
            #     layer_features.append(skip)

            #     # child_features[i].append(skip.permute(1,0,2,3))
            #     child_features[i].append(skip)
            #     # child_features[i] = torch.cat([child_features[i], skip], dim=0)
                
            # else:
            #     layer_features.append(x)

            # Resnet
            layer_features.append(x)

            if len(child_features[i])>1:
                # child_features[i] = torch.nn.utils.rnn.pad_sequence(child_features[i], batch_first=False, padding_value=0).permute(1,2,0,3,4)
                child_features[i] = pad_tensor_list_equal(child_features[i], combine=2)

                
        return [layer_features, child_features]

    



    def load_state_dict(self, state_dict):
        pattern = re.compile(
            r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
        )
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        # remove linear
        # state_dict.pop("classifier.bias")
        # state_dict.pop("classifier.weight")

        super().load_state_dict(state_dict)


