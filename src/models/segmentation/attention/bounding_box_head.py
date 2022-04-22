from typing import Tuple, Type

import torch
import torch.nn as nn


class BoundingBoxHead(nn.Module):
    """
    This class implements the feed forward bounding box head as proposed in:
    https://arxiv.org/abs/2005.12872
    """

    def __init__(self, features: Tuple[Tuple[int, int]] = ((256, 64), (64, 16), (16, 4)),
                 activation: Type = nn.PReLU) -> None:
        """
        Constructor method
        :param features: (Tuple[Tuple[int, int]]) Number of input and output features in each layer
        :param activation: (Type) Activation function to be utilized
        """
        # Call super constructor
        super(BoundingBoxHead, self).__init__()
        # Init layers
        self.layers = []
        for index, feature in enumerate(features):
            if index < len(features) - 1:
                self.layers.extend([nn.Linear(in_features=feature[0], out_features=feature[1]), activation()])
            else:
                # self.layers.append(nn.Linear(in_features=feature[0], out_features=feature[1]))
                self.layers.extend([nn.Linear(in_features=feature[0], out_features=feature[1]), nn.Sigmoid()])
        self.layers = nn.Sequential(*self.layers)

        # self.layers
        # Sequential(
        # (0): Linear(in_features=192, out_features=96, bias=True)
        # (1): LeakyReLU(negative_slope=0.01)
        # (2): Linear(in_features=96, out_features=24, bias=True)
        # (3): LeakyReLU(negative_slope=0.01)
        # (4): Linear(in_features=24, out_features=6, bias=True)
        # (5): Sigmoid()
        # )

        self.linear_1 = nn.Linear(in_features=192, out_features=96, bias=True)
        self.act_1 = nn.LeakyReLU()
        self.linear_2 = nn.Linear(in_features=96, out_features=24, bias=True)
        self.act_2 = nn.LeakyReLU()
        self.linear_3 = nn.Linear(in_features=24, out_features=6, bias=True)
        self.act_3 = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of shape (batch size, instances, features)
        :return: (torch.Tensor) Output tensor of shape (batch size, instances, classes + 1 (no object))
        """
        # return self.layers(input)

        x = input
        x = self.linear_1(x)
        x = self.act_1(x)
        x = self.linear_2(x)
        x = self.act_2(x)
        x = self.linear_3(x)
        x = self.act_3(x)

        return x

