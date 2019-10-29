import torch
import torch.nn as nn


class DownConvolutionalBlock(nn.Module):
    def __init__(self, input_dim, output_dim, initializers, padding=True, pool=True):
        super(DownConvolutionalBlock, self).__init__()

        layers = []
        if pool:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

        layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        #self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)


class UpConvolutionalBlock(nn.Module):
    """
        A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
        If bilinear is set to false, we do a transposed convolution instead of upsampling
        """

    def __init__(self, input_dim, output_dim, initializers, padding, bilinear=True):
        super(UpConvolutionalBlock, self).__init__()
        self.bilinear = bilinear

        if not self.bilinear:
            self.upconv_layer = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1),
                nn.ReLU(),
            )
            #self.upconv_layer.apply(init_weights)

        self.conv_block = DownConvolutionalBlock(input_dim, output_dim, initializers, padding, pool=False)

    def forward(self, x, bridge):
        if self.bilinear:
            up = nn.functional.interpolate(x, mode='bilinear', scale_factor=2, align_corners=True)
            up = self.upconv_layer(x)
        else:
            up = self.upconv_layer(x)

        assert up.shape[3] == bridge.shape[3]
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out


class LikelihoodUpConvolutionBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LikelihoodUpConvolutionBlock, self).__init__()
        self.upconv = nn.Sequential(
            nn.functional.interpolate(mode='bilinear', scale_factor=2, align_corners=True),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1),
            nn.ReLU()
        )


class PHISeg(nn.Module):
    """
    A PHISeg (https://arxiv.org/abs/1906.04045) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in PHISeg)
    padding: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels, num_classes, num_filters, initializers, apply_last_layer=True, padding=True):
        super(PHISeg, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer

        # POSTERIOR
        self.contracting_path = nn.ModuleList()

        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.contracting_path.append(DownConvolutionalBlock(input, output, initializers, padding, pool=pool))

        self.upsampling_path = nn.ModuleList()

        n = len(self.num_filters) - 2
        for i in range(n, -1, -1):
            input = output + self.num_filters[i]
            output = self.num_filters[i]
            self.upsampling_path.append(UpConvolutionalBlock(input, output, initializers, padding))

        # LIKELIHOOD

    def posterior(self, x, val):
        blocks = []
        z = [None] * (len(self.num_filters) - 1) # contains all hidden z
        sigma = [None] * (len(self.num_filters) - 1)
        mu = [None] * (len(self.num_filters) - 1)

        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path) - 1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):

            if i != 0:
                mu[-i] = nn.Conv2d(x.shape[1], x.shape[1], kernel_size=(3, 3), padding=1)
                sigma[-i] = nn.Conv2d(x.shape[1], x.shape[1], kernel_size=(1, 1), padding=1)
                sigma[-i] = nn.ReLU(sigma[-i])
                z_hidden = mu[-i] + sigma[-i] * torch.randn_like(sigma[-i], dtype=torch.float32)
                z[-i] = z_hidden # downmost layer

            x = up(x, blocks[-i - 1])

        del blocks

        # Used for saving the activations and plotting
        if val:
            self.activation_maps.append(x)

        if self.apply_last_layer:
            x = self.last_layer(x)

        return x

    def likelihood(self, z):
        s = [None] * (len(self.num_filters) - 2)

        return s
