import torch
import torch.nn as nn


class Conv2DSequence(nn.Module):
    """Block with 2D convolutions after each other with ReLU activation"""
    def __init__(self, input_dim, output_dim, kernel=3, depth=2):
        super(Conv2DSequence, self).__init__()

        assert depth >= 1
        if kernel == 3:
            padding = 1
        else:
            padding = 0

        layers = []
        layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=kernel, padding=padding))
        layers.append(nn.ReLU())

        for i in range(depth-1):
            layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=kernel, padding=padding))
            layers.append(nn.ReLU())

        self.convolution = nn.Sequential(*layers)

    def forward(self, x):
        return self.convolution(x)


class DownConvolutionalBlock(nn.Module):
    def __init__(self, input_dim, output_dim, initializers, depth=3, padding=True, pool=True):
        super(DownConvolutionalBlock, self).__init__()

        if depth < 1:
            raise ValueError

        layers = []
        if pool:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

        layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))

        if depth > 1:
            for i in range(depth-1):
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

        if self.bilinear:
            self.upconv_layer = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError

    def forward(self, x, bridge):
        if self.bilinear:
            x = nn.functional.interpolate(x, mode='bilinear', scale_factor=2, align_corners=True)
            x = self.upconv_layer(x)

        assert x.shape[3] == bridge.shape[3]
        assert x.shape[2] == bridge.shape[2]
        out = torch.cat([x, bridge], 1)

        return out


class SampleZBlock(nn.Module):
    """
    Performs 2 3X3 convolutions and a 1x1 convolution to mu and sigma which are used as parameters for a Gaussian
    for generating z
    """
    def __init__(self, input_dim, z_dim0=2, depth=2):
        super(SampleZBlock, self).__init__()
        self.input_dim = input_dim

        layers = []
        for i in range(depth):
            layers.append(nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU())

        self.conv = nn.Sequential(*layers)

        self.mu_conv = nn.Sequential(nn.Conv2d(input_dim, z_dim0, kernel_size=1),
                                     nn.ReLU())
        self.sigma_conv = nn.Sequential(nn.Conv2d(input_dim, z_dim0, kernel_size=1),
                                        nn.ReLU())

    def forward(self, pre_z):
        pre_z = self.conv(pre_z)
        mu = self.mu_conv(pre_z)
        sigma = self.sigma_conv(pre_z)

        z = mu + sigma * torch.randn_like(sigma, dtype=torch.float32)

        return mu, sigma, z


class Likelihood(nn.Module):
    def __init__(self, input_channels, num_classes, num_filters, initializers=None, apply_last_layer=True, padding=True):
        super(Likelihood, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters

        self.latent_levels = (len(self.num_filters) - 1)

        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        # LIKELIHOOD
        self.likelihood_ups_path = nn.ModuleList()
        self.likelihood_post_ups_path = nn.ModuleList()

        # path for upsampling
        for i in range(self.latent_levels, 0, -1):
            input = self.num_filters[i]
            output = self.num_filters[i - 1]
            self.likelihood_ups_path.append(Conv2DSequence(input_dim=2, output_dim=input, depth=2))
            self.likelihood_post_ups_path.append(Conv2DSequence(input_dim=input, output_dim=output, depth=1))

        # path after concatenation
        self.likelihood_post_c_path = nn.ModuleList()
        for i in range(self.latent_levels-1, 0, -1):
            input = self.num_filters[i] + self.num_filters[i-1]
            output = self.num_filters[i-1]
            self.likelihood_post_c_path.append(Conv2DSequence(input_dim=input, output_dim=output, depth=2))

        self.s_layer = nn.ModuleList()
        for i in range(self.latent_levels, 0, -1):
            input = self.num_filters[i-1]
            output = self.num_filters[i-1]
            self.s_layer.append(Conv2DSequence(input_dim=input, output_dim=output, depth=2, kernel=1))

    def forward(self, z):
        """Likelihood network which takes list of latent variables z with dimension latent_levels"""
        s = [None] * self.latent_levels
        post_z = [None] * self.latent_levels
        post_c = [None] * self.latent_levels

        # start from the downmost layer and the last filter
        for i in range(self.latent_levels):
            post_z[-i - 1] = self.likelihood_ups_path[i](z[-i - 1])
            post_z[-i - 1] = nn.functional.interpolate(
                post_z[-i - 1],
                mode='bilinear',
                scale_factor=2,
                align_corners=True)
            post_z[-i - 1] = self.likelihood_post_ups_path[i](post_z[-i - 1])

        post_c[self.latent_levels - 1] = post_z[self.latent_levels - 1]

        for i in reversed(range(self.latent_levels - 1)):
            ups_below = nn.functional.interpolate(
                post_c[i+1],
                mode='bilinear',
                scale_factor=2,
                align_corners=True)

            assert post_z[i].shape[3] == ups_below.shape[3]
            assert post_z[i].shape[2] == ups_below.shape[2]
            concat = torch.cat([post_z[i], ups_below], 1)

            post_c[i] = self.likelihood_post_c_path[-i-1](concat)

        for i, block in enumerate(self.s_layer):
            s_in = block(post_c[-i-1])
            s[-i-1] = s_in

        return s


class Posterior(nn.Module):
    """
    Posterior network of the PHiSeg Module
    For each latent level a sample of the distribution of the latent level is returned

    Parameters
    ----------
    input_channels : Number of input channels, 1 for greyscale,
    is_posterior: if True, the mask is concatenated to the input of the encoder, causing it to be a ConditionalVAE
    """
    def __init__(self, input_channels, num_classes, num_filters, initializers=None, padding=True, is_posterior=True):
        super(Posterior, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters

        self.latent_levels = (len(self.num_filters) - 1)

        self.padding = padding
        self.activation_maps = []

        if is_posterior:
            # increase input channel by one to accomodate place for mask
            self.input_channels += 1

        self.contracting_path = nn.ModuleList()

        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            pool = False if i == 0 else True

            self.contracting_path.append(DownConvolutionalBlock(input,
                                                                output,
                                                                initializers,
                                                                depth=3,
                                                                padding=padding,
                                                                pool=pool)
                                         )

        self.upsampling_path = nn.ModuleList()

        for i in range(self.latent_levels - 1, 0, -1):
            input = 2
            output = self.num_filters[0]*2
            self.upsampling_path.append(UpConvolutionalBlock(input, output, initializers, padding))

        self.sample_z_path = nn.ModuleList()
        for i in range(self.latent_levels, 0, -1):
            input = 2*self.num_filters[0] + self.num_filters[i]
            if i == self.latent_levels:
                input = self.num_filters[i]
                self.sample_z_path.append(SampleZBlock(input, depth=0))
            else:
                self.sample_z_path.append(SampleZBlock(input, depth=2))

    def forward(self, patch, segm=None):
        if segm is not None:
            patch = torch.cat((patch, segm), dim=1)

        blocks = []
        z = [None] * (len(self.num_filters) - 1)  # contains all hidden z
        sigma = [None] * (len(self.num_filters) - 1)
        mu = [None] * (len(self.num_filters) - 1)

        x = patch
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path) - 1:
                blocks.append(x)

        pre_conv = x
        for i, sample_z in enumerate(self.sample_z_path):
            if i != 0:
                pre_conv = self.upsampling_path[i-1](z[-i], blocks[-i])
            mu[-i-1], sigma[-i-1], z[-i-1] = self.sample_z_path[i](pre_conv)

        del blocks

        return z


class PHISeg(nn.Module):
    """
    A PHISeg (https://arxiv.org/abs/1906.04045) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in PHISeg)
    padding: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self,
                 input_channels,
                 num_classes,
                 num_filters,
                 latent_dim=2,
                 initializers=None,
                 no_convs_fcomb=4,
                 beta=10.0,
                 reversible=False,
                 apply_last_layer=True,
                 padding=True):
        super(PHISeg, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters

        self.latent_levels = (len(self.num_filters) - 1)

        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer

        self.posterior = Posterior(input_channels, num_classes, num_filters, initializers=None, padding=True)
        self.likelihood = Likelihood(input_channels, num_classes, num_filters, initializers=None, apply_last_layer=True, padding=True)
        self.prior = Posterior(input_channels, num_classes, num_filters, initializers=None, padding=True, is_posterior=False)

    def sample(self):
        pass

    def reconstruct(self):
        pass

    def forward(self, patch, mask, training=True):
        if training:
            self.posterior_latent_space = self.posterior(patch, mask)
            self.segm_vector = self.likelihood(self.posterior_latent_space)
        else:
            self.prior_latent_space = self.prior(patch)
            self.segm_vector = self.likelihood(self.prior_latent_space)

        return self.segm_vector

    def kl_divergence(self):
    """Kullback-Leibler Divergence for n latent levels"""
        pass
    def elbo(self):
        pass
