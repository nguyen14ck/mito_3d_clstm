# BASE MODEL: INITIALIZATION, FORWARD FUCNTION


import torch
import torch.nn as nn
import torch.nn.functional as Fn
from . import initialization as init


class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP,self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
 
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
 
    def forward_single(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = Fn.upsample(image_features, size=size, mode='bilinear')
 
        atrous_block1 = self.atrous_block1(x)
 
        atrous_block6 = self.atrous_block6(x)
 
        atrous_block12 = self.atrous_block12(x)
 
        atrous_block18 = self.atrous_block18(x)
 
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

    def forward(self, f):
        out = []
        for i, x in enumerate(f):
            size = x.shape[2:]
    
            image_features = self.mean(x)
            image_features = self.conv(image_features)
            image_features = Fn.upsample(image_features, size=size, mode='bilinear')
            # image_features = nn.Upsample(scale_factor=upsampling, mode='bilinear', align_corners=True) if upsampling > 1 else nn.Identity()
    
            atrous_block1 = self.atrous_block1(x)
    
            atrous_block6 = self.atrous_block6(x)
    
            atrous_block12 = self.atrous_block12(x)
    
            atrous_block18 = self.atrous_block18(x)
    
            net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                                atrous_block12, atrous_block18], dim=1))

            out.append(net)
        out = torch.stack(out)                       
        return out


class ASPP3D(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP3D,self).__init__()
        self.mean = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv = nn.Conv3d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv3d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv3d(in_channel, depth, 3, 1, padding=3, dilation=3)
        self.atrous_block12 = nn.Conv3d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block18 = nn.Conv3d(in_channel, depth, 3, 1, padding=9, dilation=9)
 
        self.conv_1x1_output = nn.Conv3d(depth * 5, depth, 1, 1)
 
    def forward_single(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = Fn.upsample(image_features, size=size, mode='trilinear')

        atrous_block1 = self.atrous_block1(x)
 
        atrous_block6 = self.atrous_block6(x)
 
        atrous_block12 = self.atrous_block12(x)
 
        atrous_block18 = self.atrous_block18(x)
 
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

    def forward(self, f):
        out = []
        for i, x in enumerate(f):
            size = x.shape[2:]
    
            image_features = self.mean(x)
            image_features = self.conv(image_features)
            image_features = Fn.upsample(image_features, size=size, mode='trilinear')
    
            atrous_block1 = self.atrous_block1(x)
    
            atrous_block6 = self.atrous_block6(x)
    
            atrous_block12 = self.atrous_block12(x)
    
            atrous_block18 = self.atrous_block18(x)
    
            net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                                atrous_block12, atrous_block18], dim=1))

            out.append(net)
        out = torch.stack(out)                       
        return out

class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x


class SegmentationModel_ASPP(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.aspp)
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)

        features_aspp = [features[0]]
        features_aspp.append([])
        features_aspp[1].extend(features[1][:-1])
        features_aspp[1].append(self.aspp(features[1][-1]))
        
        decoder_output = self.decoder(*features_aspp)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

class SegmentationModel_Double(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.aspp)
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

        init.initialize_decoder(self.aspp_2)
        init.initialize_decoder(self.decoder_2)
        init.initialize_head(self.segmentation_head_2)

        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        features_aspp = [features[0]]
        features_aspp.append([])
        features_aspp[1].extend(features[1][:-1])
        features_aspp[1].append(self.aspp(features[1][-1]))
        decoder_output = self.decoder(*features_aspp)
        masks = self.segmentation_head(decoder_output)

        x_2 = x*masks

        features_2 = self.encoder_2(x_2)
        features_aspp_2 = [features_2[0]]
        features_aspp_2.append([])
        features_aspp_2[1].extend(features_2[1][:-1])
        features_aspp_2[1].append(self.aspp_2(features_2[1][-1]))
        decoder_output_2 = self.decoder_2(features_aspp, *features_aspp_2)
        masks_2 = self.segmentation_head_2(decoder_output_2)

        if self.classification_head is not None:
            labels = self.classification_head(features_aspp_2[-1])
            return masks_2, labels

        return masks_2



class SegmentationModel_ASPP_AE(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder_ae)

        init.initialize_decoder(self.aspp)

        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)

        features_aspp = [features[0]]
        features_aspp.append([])
        features_aspp[1].extend(features[1][:-1])
        features_aspp[1].append(self.aspp(features[1][-1]))
        
        decoder_output = self.decoder(*features_aspp)
        masks = self.segmentation_head(decoder_output)

        ae_output = self.decoder_ae(*features)

        outputs = torch.cat([masks, ae_output], dim=1)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return outputs, labels

        return outputs


class SegmentationModel_ASPP_AE2(torch.nn.Module):

    def initialize(self):
        # init.initialize_decoder(self.decoder_ae)
        init.initialize_decoder(self.aspp)
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

        init.initialize_decoder(self.decoder_2)
        init.initialize_head(self.segmentation_head_2)

        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)

        features_aspp = [features[0]]
        features_aspp.append([])
        features_aspp[1].extend(features[1][:-1])
        features_aspp[1].append(self.aspp(features[1][-1]))
        
        decoder_output = self.decoder(*features_aspp)
        masks = self.segmentation_head(decoder_output)

        decoder_output_2 = self.decoder(*features_aspp)
        masks_2 = self.segmentation_head(decoder_output_2)

        # ae_output = self.decoder_ae(*features)

        outputs = torch.cat([masks, masks_2], dim=1)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return outputs, labels

        return outputs


class SegmentationModel_Attention(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels


        # Get features from backbone
        # last_features, layer_features = self.backbone(input)
        # Map features to the desired shape
        last_features = self.convolution_mapping(features[0][-1])
        # Get height and width of last_features
        height, width, depth = last_features.shape[2:]
        # Get batch size
        batch_size = last_features.shape[0]

        # Make positional embeddings
        positional_embeddings = torch.cat([self.column_embedding[:width].unsqueeze(dim=0).unsqueeze(dim=2).repeat(height, 1, depth, 1), # CHANGE
                                           self.row_embedding[:height].unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, width, depth, 1),
                                           self.depth_embedding[:depth].unsqueeze(dim=0).unsqueeze(dim=0).repeat(height, width, 1, 1)],
                                          dim=-1).permute(3, 0, 1, 2).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        latent_tensor, features_encoded = self.transformer(last_features, None, self.query_positions, positional_embeddings)
        latent_tensor = latent_tensor.permute(2, 0, 1)

        # Get class prediction
        # class_prediction = F.softmax(self.class_head(latent_tensor), dim=2).clone() # out_features = num_classes + 1
        # Get bounding boxes
        bounding_box_prediction = self.bounding_box_head(latent_tensor)

        return masks, bounding_box_prediction.clone()