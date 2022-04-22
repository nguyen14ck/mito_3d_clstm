
# DERIVED MODEL: CONSTRUCTOR

# %%
import numpy as np
import pandas as pd
import torch

try:
    from torchinfo import summary
except: 
    from torchsummary import summary


# %%
from typing import Optional, Union, List
from src.models.segmentation.encoder.encoder_monai import EncoderMonaiCLstm
from src.models.segmentation.decoder.decoder3d_clstm import UnetDecoder3DCLstm
from src.models.segmentation.model import SegmentationModel
from src.models.segmentation.heads import SegmentationHead3D, ClassificationHead3D



# %%
class Unet3D_CLSTM(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        num_clstm_layers = 1,
        device = torch.device("cpu"), # all devices
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()
        
        self.spatial_dims = 3
        self.in_channel = 1
        self.encoder_block_config = []
        self.encoder_out_channels = (1, 32, 64, 128, 256)
        self.decoder_channels = (128, 64, 32)
        self. encoder_depth = encoder_depth

        

        self.decoder_use_batchnorm = True
        self.decoder_attention_type = None
        self.encoder_name = 'monai_resnet'
        self.segment_classes = classes
        self.num_clstm_layers = num_clstm_layers
        self.device = device


        self.encoder = EncoderMonaiCLstm('EncoderMonaiClstm',
                                    dimensions=3,
                                    in_channels=1,
                                    out_channels=3,
                                    channels=(32, 64, 128, 256),
                                    strides=(2, 2, 2),
                                    num_res_units=16,
                                    norm='INSTANCE',
                                    ratio=0.25,
                                )

        self.decoder = UnetDecoder3DCLstm(
            encoder_channels=self.encoder_out_channels,
            decoder_channels=self.decoder_channels,
            n_blocks=self.encoder_depth,
            use_batchnorm=self.decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=self.decoder_attention_type,
            num_classes=self.segment_classes,
            num_clstm_layers = self.num_clstm_layers,
            device = self.device,
            )

        self.segmentation_head = SegmentationHead3D(
            in_channels=64, # by checking decoder output
            out_channels=self.segment_classes,
            activation=activation,
            kernel_size=3,
            upsampling=2.0,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead3D(
            in_channels=self.encoder.out_channels[-1], **aux_params
        )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()














# # %%
# # -----------------------------------------------------------
# # SUMMARY


# # %%
# model_u3 = Unet3D_CLSTM(encoder_name='seresnet', encoder_depth=3, encoder_weights=None, 
#                 decoder_channels=(64, 32, 16), in_channels=1, classes=1, num_clstm_layers = 1, device = torch.device("cuda"),
#                 activation=None, decoder_attention_type=None)

# # %%
# print(model_u3)


# # %%
# inputs = torch.rand(1, 1, 64, 64, 64).cuda()
# model_u3 = model_u3.cuda()
# model_u3.eval()
# outputs = model_u3(inputs)


# # %%
# # model_u3 = model_u3.cuda()
# # summary(model_u3, input_size=(1, 512, 512))
# # summary(model_u3, input_size=(1, 192, 192, 192))
# summary(model_u3, (1, 64, 64, 64), depth=12, col_names=["kernel_size", "output_size", "num_params", "mult_adds"],)

# # Depth = 4
# # ===========================================================================
# # Layer (type:depth-idx)                             Param #
# # ===========================================================================
# # ├─CustomEncoder: 1-1                               --
# # |    └─SENet: 2-1                                  --
# # |    |    └─Sequential: 3-1                        22,080
# # |    |    └─Sequential: 3-2                        258,864
# # |    |    └─Sequential: 3-3                        1,477,760
# # |    |    └─Sequential: 3-4                        8,700,288
# # |    |    └─Sequential: 3-5                        17,893,760
# # |    |    └─AdaptiveAvgPool3d: 3-6                 --
# # |    |    └─Linear: 3-7                            4,098
# # ├─UnetDecoder3DCLstm: 1-2                          --
# # |    └─ModuleList: 2-2                             --
# # |    |    └─ConvLSTM3D: 3-8                        905,977,856
# # |    |    └─ConvTranspose3d: 3-9                   33,556,480
# # |    |    └─ConvLSTM3D: 3-10                       679,485,440
# # |    |    └─ConvTranspose3d: 3-11                  16,778,240
# # |    |    └─ConvLSTM3D: 3-12                       169,873,408
# # |    |    └─ConvTranspose3d: 3-13                  4,194,816
# # |    |    └─ConvLSTM3D: 3-14                       42,469,376
# # |    |    └─ConvTranspose3d: 3-15                  1,048,832
# # ├─SegmentationHead3D: 1-3                          --
# # |    └─Conv3d: 2-3                                 6,913
# # |    └─Upsample: 2-4                               --
# # |    └─Activation: 2-5                             --
# # |    |    └─Identity: 3-16                         --
# # ===========================================================================
# # Total params: 1,881,748,211
# # Trainable params: 1,881,748,211
# # Non-trainable params: 0
# # ===========================================================================


# # %%
# # Densenet
# # summary(model_u3, input_size=(1, 192, 192, 192))

# # Depth = 3
# # =================================================================
# # Layer (type:depth-idx)                   Param #
# # =================================================================
# # ├─CustomEncoder: 1-1                     --
# # |    └─DenseNet: 2-1                     --
# # |    |    └─Sequential: 3-1              11,242,624
# # ├─UnetDecoder3DCLstm: 1-2                --
# # |    └─ModuleList: 2-2                   --
# # |    |    └─ConvLSTM3D: 3-2              226,496,512
# # |    |    └─ConvTranspose3d: 3-3         8,389,632
# # |    |    └─ConvLSTM3D: 3-4              169,873,408
# # |    |    └─ConvTranspose3d: 3-5         4,194,816
# # |    |    └─ConvLSTM3D: 3-6              42,469,376
# # |    |    └─ConvTranspose3d: 3-7         1,048,832
# # ├─SegmentationHead3D: 1-3                --
# # |    └─Conv3d: 2-3                       6,913
# # |    └─Upsample: 2-4                     --
# # |    └─Activation: 2-5                   --
# # |    |    └─Identity: 3-8                --
# # =================================================================
# # Total params: 463,722,113
# # Trainable params: 463,722,113
# # Non-trainable params: 0
# # =================================================================



# # %%
# print(model_u3)


# # %%
# model_u3.decoder.clstm



# # %%
# model_u3.decoder.conv_trans


# # %%
# for sublayer in model_u3.decoder.conv_trans:
#     print(sublayer)