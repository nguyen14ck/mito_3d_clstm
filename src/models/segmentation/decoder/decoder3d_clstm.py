from monai.transforms.spatial.array import Affine
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import modules as md
# from .convlstm3d import *
from .convlstm3d import *

class DecoderBlock3D(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=True)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x



# =================================================================================================================================================



# =================================================================================================================================================

class CenterBlock3D(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv3dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder3D(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock3D(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock3D(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x



class UnetDecoder3DCLstm(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
            num_classes=2,
            num_clstm_layers = 1,
            device = torch.device('cpu'),
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        # if center:
        #     self.center = CenterBlock3D(
        #         head_channels, head_channels, use_batchnorm=use_batchnorm
        #     )
        # else:
        #     self.center = nn.Identity()

        # # combine decoder keyword arguments
        # kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        # blocks = [
        #     DecoderBlock3D(in_ch, skip_ch, out_ch, **kwargs)
        #     for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        # ]
        # self.blocks = nn.ModuleList(blocks)

        # ============================
        # Extra
        self.num_classes = num_classes
        self.clstm = [None for _ in  range(len(encoder_channels)-1)] # ignore two first stages
        # self.clstm = nn.ModuleList([None for _ in  range(len(encoder_channels))])
        self.num_layers = num_clstm_layers
        self.conv_trans = [None for _ in  range(len(encoder_channels)-1)] # ignore two first stages
        # self.conv_trans = nn.ModuleList([None for _ in  range(len(encoder_channels))])
        self.conv_trans_final = None
        # self.conv_final = [None for _ in range(3)]
        # self.conv_final = nn.ModuleList([None for _ in range(3)])
        self.final_stride = 1
        self.device = device

        self.clstm_convUp = []
        for i in range(len(encoder_channels)-1): # ignore two first stages
            self.clstm_convUp.append(self.clstm[i])
            self.clstm_convUp.append(self.conv_trans[i])
        self.clstm_convUp = nn.ModuleList(self.clstm_convUp)    



        # [ConvLSTM3D(
        #     (cell_list): ModuleList(
        #         (0): ConvLSTMCell3D(
        #         (conv): Conv3d(2048, 4096, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        #         )
        #     )
        #     ),
        #     ConvLSTM3D(
        #     (cell_list): ModuleList(
        #         (0): ConvLSTMCell3D(
        #         (conv): Conv3d(1536, 4096, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        #         )
        #     )
        #     ),
        #     ConvLSTM3D(
        #     (cell_list): ModuleList(
        #         (0): ConvLSTMCell3D(
        #         (conv): Conv3d(768, 2048, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        #         )
        #     )
        #     )]

        # [ConvTranspose3d(1024, 1024, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        # ConvTranspose3d(1024, 512, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        # ConvTranspose3d(512, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2))]
        
        



    # def forward(self, *features):

    #     features = features[1:]    # remove first skip with same spatial resolution
    #     features = features[::-1]  # reverse channels to start from head of encoder

    #     head = features[0]
    #     skips = features[1:]

    #     x = self.center(head)
    #     for i, decoder_block in enumerate(self.blocks):
    #         skip = skips[i] if i < len(skips) else None
    #         x = decoder_block(x, skip)
    #     return x

    def forward(self, *features):
        
        # lstm - decoding path
        output = features[0][2:] # first index: get the final features at each layer output; second index: ignore two first layers of encoded features (identity layer and intial layer)
        output = output[::-1] # reverse channels to start from head of encoder

        ft = features[1][2:] # first index: get all the features at each level; second index: ignore two first layers of encoded features (identity layer and intial layer)
        ft = ft[::-1]  # reverse channels to start from head of encoder
        # ft = ft.reverse()

        # head_ft = ft[0]
        skips = ft

        # x = self.center(head_output)

        # new_hidden = torch.zeros_like(features[0][-1])
        # new_cell = torch.zeros_like(features[0][-1])
        new_hidden = torch.zeros_like(output[0], device=output[0].device)
        new_cell = torch.zeros_like(output[0], device=output[0].device)
        # new_state = torch.cat([new_hidden, new_cell], dim=3) # inverse order (compare to Tensorflow CFCM)
        # new_state = [new_hidden, new_cell] # inverse order (compare to Tensorflow CFCM)
        new_state = []
        layer_output_list = [] # new_hidden

        for i in range(self.num_layers):
            new_state.append([new_hidden, new_cell])


        for i in range(len(skips)):
            if self.clstm[i] is None:
                # self.clstm[i] = ConvLSTM(input_dim=skips[i].shape[1], hidden_dim=skips[i].shape[1], kernel_size=(3,3), num_layers=self.num_layers, batch_first=False, bias=True, return_all_layers=False)
                self.clstm[i] = ConvLSTM3D(input_dim=output[i].shape[1], hidden_dim=new_state[self.num_layers-1][0].shape[1], kernel_size=(3,3,3), num_layers=self.num_layers, batch_first=False, bias=True, return_all_layers=False)
                self.clstm_convUp[i*2] = self.clstm[i]
                self.clstm[i].to(self.device)
            layer_output_list, new_state = self.clstm[i](input_tensor=skips[i], hidden_state=new_state)
            # new_hidden, new_cell = new_state[self.num_layers-1]

            if self.conv_trans[i] is None:
                self.conv_trans[i] = nn.ConvTranspose3d(in_channels=new_state[self.num_layers-1][1].shape[1], out_channels=output[i].shape[1],  kernel_size=2, stride=2)
                # self.conv_trans[i] = nn.Sequential(nn.ConvTranspose3d(in_channels=new_state[self.num_layers-1][1].shape[1], out_channels=output[i].shape[1],  kernel_size=2, stride=2),
                #                                     nn.InstanceNorm3d(output[i].shape[1], affine=True))
                self.clstm_convUp[i*2+1] = self.conv_trans[i]
                self.conv_trans[i].to(self.device)
            new_state[self.num_layers-1][0] = self.conv_trans[i](new_state[self.num_layers-1][0]) # new_hidden
            new_state[self.num_layers-1][1] = self.conv_trans[i](new_state[self.num_layers-1][1]) # new_cell


        final = new_state[self.num_layers-1][0]
        # if self.conv_trans_final is None:
            # self.conv_trans_final = nn.ConvTranspose2d(in_channels=layer_output_list[self.num_layers-1].shape[2], out_channels=output[-1].shape[1], kernel_size=2, stride=2).to(self.device)
            # self.conv_final_1 = nn.Conv2d(in_channels=output[-1].shape[1],
            #                     out_channels=output[-1].shape[1],
            #                     kernel_size=3,
            #                     stride=self.final_stride,
            #                     # padding=('SAME' if self.final_stride == 1 else 'VALID'),
            #                     # activation_fn=None
            #                     ).to(self.device)
            # self.conv_final_2 = nn.ReLU().to(self.device)



        # final = self.conv_trans_final(layer_output_list[self.num_layers-1])
        # final_1 = self.conv_final_1(final)
        # final_2 = self.conv_final_2(final_1)

        # if self.conv_final_3 is None:
        #     self.conv_final_3 = nn.Conv2d(in_channels=final_2.shape[1],
        #                         out_channels=self.num_classes,
        #                         kernel_size=3,
        #                         stride=self.final_stride,
        #                         # padding=('SAME' if self.final_stride == 1 else 'VALID'),
        #                         # activation_fn=None
        #                         ).to(self.device)
        # final_3 = self.conv_final_3(final_2)                        

        return final
