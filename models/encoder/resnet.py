import torch.nn as nn
import torch.nn.functional as F

# in ResNet-v2 paper they show:
#   Sublayer(x) = Conv(ReLU(BN(Conv(ReLU(BN(x))))))
#   ResidualUnit(x) = x + Sublayer(x)
#
# getting rid of BNs, we end up with a
#   ResidualUnitNoBN(x) = x + Conv(ReLU(Conv(ReLU(x))))
class ResNetV2Block(nn.Module):
    def __init__(self, in_maps, out_maps):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(out_maps, out_maps, kernel_size=3, stride=1, padding=1, bias=True)
        )

        # use a 1x1 convolution if needed to make the dimensions work on the skip connection
        self.skip = (nn.Identity() if in_maps == out_maps
            else nn.Conv2d(in_maps, out_maps, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        output = self.skip(x) + self.layers(x)
        return output

# with layer norm before each ReLU. GroupNorm(1, C) is LayerNorm
class ResNetV2BlockLN(nn.Module):
    def __init__(self, in_maps, out_maps):
        super().__init__()

        self.layers = nn.Sequential(
            nn.GroupNorm(1, in_maps),
            nn.ReLU(),
            nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, out_maps),
            nn.ReLU(),
            nn.Conv2d(out_maps, out_maps, kernel_size=3, stride=1, padding=1, bias=False)
        )

        # use a 1x1 convolution if needed to make the dimensions work on the skip connection
        self.skip = (nn.Identity() if in_maps == out_maps
            else nn.Conv2d(in_maps, out_maps, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        output = self.skip(x) + self.layers(x)
        return output

class ResNetV2BlockBN(nn.Module):
    def __init__(self, in_maps, out_maps):
        super().__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_maps),
            nn.ReLU(),
            nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_maps),
            nn.ReLU(),
            nn.Conv2d(out_maps, out_maps, kernel_size=3, stride=1, padding=1, bias=True)
        )

        # use a 1x1 convolution if needed to make the dimensions work on the skip connection
        self.skip = (nn.Identity() if in_maps == out_maps
            else nn.Conv2d(in_maps, out_maps, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        output = self.skip(x) + self.layers(x)
        return output


class ResNetV2_Encoder(nn.Module):
    def __init__(self, input_channels, layer_num_features, layer_num_blocks):
        super().__init__()

        if (len(layer_num_blocks) != len(layer_num_features)):
            raise Exception('layer_num_blocks and layer_num_features must have the same length')

        init_num_features = layer_num_features[0]
        self.init_conv = nn.Conv2d(input_channels, init_num_features, kernel_size=3, stride=1, padding=1, bias=True)
        self.encoding_num_features = layer_num_features[-1]

        curr_num_features = init_num_features
        the_layers = []
        for i in range(len(layer_num_blocks)):
            the_blocks = []
            for _ in range(layer_num_blocks[i]):
                the_blocks.append(ResNetV2Block(curr_num_features, layer_num_features[i]))
                curr_num_features = layer_num_features[i]
            the_layers.append(nn.Sequential(*the_blocks))
        self.layers = nn.Sequential(*the_layers)

        # from paper: "terminates with a mean-pooling operation"
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """ forward(x)
        Takes a tensor of shape (N, H_grid, W_grid, C, H, W)
        Returns a tensor of shape (N, H_grid, W_grid, encoding_size)"""
        # print(x.shape)
        N, H_grid, W_grid, C, H, W = x.shape

        # reshape for input into a conv layer

        x_reshaped = x.view(-1, C, H, W)
        z = self.init_conv(x_reshaped)
        print(z.shape)
        z = self.layers(z)
        print(z.shape)
        z = self.avg_pool(z)
        print(z.shape)
        # here, the output of avg_pool should have shape (N*H_grid*W_grid, encoding_size, 1, 1)
        z = z.view(N, H_grid, W_grid, z.shape[1])
        print(z.sahpe)
        return z


# operates on images instead of overlapping patches of images
class ResNetV2_Classifier(nn.Module):

    def __init__(self, input_channels, layer_num_features, layer_num_blocks, num_classes):
        super().__init__()

        if (len(layer_num_blocks) != len(layer_num_features)):
            raise Exception('layer_num_blocks and layer_num_features must have the same length')

        init_num_features = layer_num_features[0]
        self.init_conv = nn.Conv2d(input_channels, init_num_features, kernel_size=3, stride=1, padding=1, bias=True)
        self.encoding_num_features = layer_num_features[-1]

        curr_num_features = init_num_features
        the_layers = []
        for i in range(len(layer_num_blocks)):
            the_blocks = []
            for _ in range(layer_num_blocks[i]):
                the_blocks.append(ResNetV2BlockBN(curr_num_features, layer_num_features[i]))
                curr_num_features = layer_num_features[i]
            the_layers.append(nn.Sequential(*the_blocks))
        self.layers = nn.Sequential(*the_layers)

        # from paper: "terminates with a mean-pooling operation"
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_layer = nn.Linear(input_channels, num_classes)

    def forward(self, x):
        """ forward(x)
        Takes a tensor of shape (N, H_grid, W_grid, C, H, W)
        Returns a tensor of shape (N, num_classes)"""
        z = self.init_conv(x)
        z = self.layers(z)
        z = self.avg_pool(z)
        # here, the output of avg_pool should have shape (N, encoding_size, 1, 1)
        z = z.view(x.shape[0], z.shape[1])
        z = self.fc_layer(z)
        return z
