import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import torch
import torch.nn as nn

from loss.InfoNCE import InfoNCELoss
from models.encoder.resnet import ResNetV2_Classifier

import torch
import torch.nn as nn

# from InfoNCE import InfoNCELoss
# from resnet_nh import ResNetV2_Classifier


class CPCV1(nn.Module):
    def __init__(self, encoder_net, autoreg_net, num_pred_steps, num_neg_samples, device, learning_mode='self-supervised', num_classes=None):
        super().__init__()

        assert learning_mode in ['self-supervised', 'linear-classification', 'efficient-classification']

        self.learning_mode = learning_mode
        self.device = device
        self.encoder_net = encoder_net.to(device)
        self.autoreg_net = autoreg_net
        if self.autoreg_net is not None:
            self.autoreg_net = self.autoreg_net.to(device)

        opt = {
            'num_negative_samples': num_neg_samples,
            'num_pred_steps': num_pred_steps,
            'device': device
        }

        # From CPCV2 appendix:
        # > For linear classification we encode each image in the same way as during self-supervised
        # > pre-training (section A), yielding a 6x6 grid of 4096-dimensional features vectors. We then
        # > use Batch-Normalization [...] to normalize the features (omitting the scale parameter) followed
        # > by a 1x1 convolution mapping each feature in the grid to the 1000 logits for ImageNet
        #
        # From CPCv1 paper:
        # > Note that when training the linear classifier we first spatially mean-pool the 7x7x1024
        # > representation to a single 1024 dimensional vector
        if self.learning_mode == 'self-supervised':
            # we only need InfoNCELoss during self-supervised pre-training
            in_channels = encoder_net.encoding_num_features
            self.info_nce_loss = InfoNCELoss(opt, in_channels=in_channels, out_channels=in_channels).to(device)
        elif self.learning_mode == 'linear-classification':
            # if we spatially meanpool first, then the pooled input to the linear layer
            # is of shape (N, encoding_size)
            # also, CPCV1 uses a 1x1 conv for its linear layer, but CPCv1 just says "linear"
            self.linear_classifier = nn.Linear(encoder_net.encoding_num_features, num_classes).to(device)
        elif self.learning_mode == 'efficient-classification':
            layer_num_blocks = [2, 2, 2]
            layer_num_features = [512, 512, 512]

            self.efficient_classifier = ResNetV2_Classifier(encoder_net.encoding_num_features, layer_num_features=layer_num_features, layer_num_blocks=layer_num_blocks, num_classes=num_classes).to(device)
        else:
            raise Exception("Unrecognized learning_mode")

    def forward(self, x):
        # encoder outputs (N, H_grid, W_grid, encoding_size), where encoding_size is the
        # number of feature maps output from the encoder network.
        encoding = self.encoder_net(x)

        if self.learning_mode == 'linear-classification':
            # spatial mean-pool, (N, H_grid, W_grid, enc_size) -> (N, enc_size)
            enc_mean_pool = torch.mean(encoding, dim=(1,2))
            logits = self.linear_classifier(enc_mean_pool)
            # enc_mean_pool.shape = (batch_size, encoding_num_features)
            # logits.shape = (batch_size, num_classes)
            # Note: we don't return a loss, because during classification we need labels to do that
            return logits

        # here, we switcheroo to make the standard (N, C, H, W) shape, and then save in a
        # new contiguous chunk of memory
        encoding = encoding.permute(0,3,1,2).contiguous()

        if self.learning_mode == 'efficient-classification':
            logits = self.efficient_classifier(encoding)
            return logits

        # here we only do one direction, for CPCV1
        context = self.autoreg_net(encoding)
        loss = self.info_nce_loss(encoding, context)
        return loss


# TODO. from CPCv2 paper:
#
# def CPC(latents , target_dim=64 , emb_scale= 0.1 , steps_to_ignore=2 , steps_to_predict= 3 ):
# # latents: [B, H, W, D]
#     loss = 0.0
#     context = pixelCNN(latents)
#     targets = Conv2D(output_channels=target_dim , kernel_shape= ( 1 , 1 ))(latents)
#     batch_dim , col_dim , rows = targets . shape [ : -1 ]
#     targets = reshape(targets , [ - 1 , target_dim ] )
#     for i in range(steps_to_ignore , steps_to_predict) :
#         col_dim_i = col_dim - i - 1
#         total_elements = batch_dim * col_dim_i * rows
#         preds_i = Conv2D(output_channels=target_dim ,
#         kernel_shape= ( 1 , 1 ) ) (context)
#         preds_i = preds_i [ : , : - (i+ 1 ) , : , : ] * emb_scale
#         preds_i = reshape(preds_i , [ - 1 , target_dim ] )
#         logits = matmul(preds_i , targets , transp_b=True)
#         b = range(total_elements) / (col_dim_i * rows)
#         col = range(total_elements) % (col_dim_i * rows)
#         labels = b * col_dim * rows + (i+ 1 ) * rows + col
#         loss += cross_entropy_with_logits(logits , labels)
#     return loss
#
class CPCV2(nn.Module):
    def __init__(self):
        super().__init__()


class CPCVIT(nn.Module):
    def __init__(self, encoder_net, autoreg_net, num_pred_steps, num_neg_samples, device, learning_mode='self-supervised', num_classes=None):
        super().__init__()

        assert learning_mode in ['self-supervised', 'linear-classification', 'efficient-classification']

        self.learning_mode = learning_mode
        self.device = device
        self.encoder_net = encoder_net.to(device)
        self.autoreg_net = autoreg_net
        if self.autoreg_net is not None:
            self.autoreg_net = self.autoreg_net.to(device)

        opt = {
            'num_negative_samples': num_neg_samples,
            'num_pred_steps': num_pred_steps,
            'device': device
        }

        if self.learning_mode == 'self-supervised':
            in_channels = encoder_net.encoding_num_features
            self.info_nce_loss = InfoNCELoss(opt, in_channels=in_channels, out_channels=in_channels).to(device)

        elif self.learning_mode == 'linear-classification':
            # if we spatially meanpool first, then the pooled input to the linear layer
            # is of shape (N, encoding_size)
            # also, CPCV1 uses a 1x1 conv for its linear layer, but CPCv1 just says "linear"
            self.linear_classifier = nn.Linear(encoder_net.encoding_num_features, num_classes).to(device)

        # elif self.learning_mode == 'efficient-classification':

        #     self.efficient_classifier = ResNetV2_Classifier(encoder_net.encoding_num_features, layer_num_features=layer_num_features, layer_num_blocks=layer_num_blocks).to(device)
        else:
            raise Exception("Unrecognized learning_mode")
        

    def forward(self, x):
        # encoder outputs (N, H_grid, W_grid, encoding_size), where
        # encoding_size is the number of feature maps output from the
        # encoder network. here, we switcheroo to make the standard
        # (N, C, H, W) shape, and then save in a new contiguous chunk
        # of memory
    
        # x.shape = torch.Size([2, 7, 7, 1, 8, 8]) vit ([49, 1, 8, 8])
        batch_size, grid_size, channels, patch_size = x.shape[0], x.shape[1], x.shape[3], x.shape[4]
        x = x.view(-1, channels, patch_size, patch_size)

        encoding = self.encoder_net(x)
        encoding = encoding.view(batch_size, grid_size, grid_size, self.encoder_net.encoding_num_features)

        if self.learning_mode == 'linear-classification':
            # spatial mean-pool, (N, H_grid, W_grid, enc_size) -> (N, enc_size)
            enc_mean_pool = torch.mean(encoding, dim=(1,2))
            logits = self.linear_classifier(enc_mean_pool)
            # enc_mean_pool.shape = (batch_size, encoding_num_features)
            # logits.shape = (batch_size, num_classes)
            # Note: we don't return a loss, because during classification we need labels to do that
            return logits
            
        encoding = encoding.permute(0,3,1,2).contiguous()  # encoding torch.Size([2, 32, 7, 7])

        # here we only do one direction, for CPCV1
        context = self.autoreg_net(encoding)      #torch.Size([2, 32, 7, 7])
        loss = self.info_nce_loss(encoding, context)
        return loss
