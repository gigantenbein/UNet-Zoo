import torch
import torch.nn as nn
import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# TODO: only debugging
from utils import show_tensor


class Conv2D(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, kernel_size=3, padding=1, activation=torch.nn.ReLU, norm=torch.nn.BatchNorm2d,
                 norm_before_activation=True):
        super(Conv2D, self).__init__()

        if kernel_size == 3:
            padding = 1
        else:
            padding = 0

        layers = []
        layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, padding=padding, stride=stride))
        if norm_before_activation:
            layers.append(norm(num_features=output_dim))
            layers.append(activation())
        else:
            layers.append(activation())
            layers.append(norm(num_features=output_dim))

        self.convolution = nn.Sequential(*layers)

    def forward(self, x):
        return self.convolution(x)


class Conv2DSequence(nn.Module):
    """Block with 2D convolutions after each other with ReLU activation"""
    def __init__(self, input_dim, output_dim, kernel=3, depth=2, activation=torch.nn.ReLU, norm=torch.nn.BatchNorm2d, norm_before_activation=True):
        super(Conv2DSequence, self).__init__()

        assert depth >= 1
        if kernel == 3:
            padding = 1
        else:
            padding = 0

        layers = []
        layers.append(Conv2D(input_dim, output_dim, kernel_size=kernel, padding=padding, activation=activation, norm=norm))

        for i in range(depth-1):
            layers.append(Conv2D(output_dim, output_dim, kernel_size=kernel, padding=padding, activation=activation, norm=norm))

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

        layers.append(Conv2D(input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))

        if depth > 1:
            for i in range(depth-1):
                layers.append(Conv2D(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))

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
                Conv2D(input_dim, output_dim, kernel_size=3, stride=1, padding=1),
                Conv2D(output_dim, output_dim, kernel_size=3, stride=1, padding=1),
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
            layers.append(Conv2D(input_dim, input_dim, kernel_size=3, padding=1))

        self.conv = nn.Sequential(*layers)

        self.mu_conv = nn.Sequential(nn.Conv2d(input_dim, z_dim0, kernel_size=1))
        self.sigma_conv = nn.Sequential(nn.Conv2d(input_dim, z_dim0, kernel_size=1),
                                        nn.Softplus())

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
        output = self.num_classes
        for i in range(self.latent_levels, 0, -1):
            input = self.num_filters[i-1]
            self.s_layer.append(Conv2DSequence(
                input_dim=input, output_dim=output, depth=1, kernel=1, activation=torch.nn.Identity, norm=torch.nn.Identity))

    def forward(self, z):
        """Likelihood network which takes list of latent variables z with dimension latent_levels"""
        s = [None] * self.latent_levels
        post_z = [None] * self.latent_levels
        post_c = [None] * self.latent_levels

        # start from the downmost layer and the last filter
        for i in range(self.latent_levels):
            assert z[-i-1].shape[1] == 2
            assert z[-i-1].shape[2] == 128 * 2**(-self.latent_levels + i)
            post_z[-i - 1] = self.likelihood_ups_path[i](z[-i - 1])
            post_z[-i - 1] = nn.functional.interpolate(
                post_z[-i - 1],
                mode='bilinear',
                scale_factor=2,
                align_corners=True)
            assert post_z[-i - 1].shape[2] == 128 * 2 ** (-self.latent_levels + i + 1)
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

            # Reminder: Pytorch standard is NCHW, TF NHWC
            concat = torch.cat([post_z[i], ups_below], dim=1)

            post_c[i] = self.likelihood_post_c_path[-i-1](concat)

        for i, block in enumerate(self.s_layer):
            s_in = block(post_c[-i-1]) # no activation in the last layer
            s[-i-1] = torch.nn.functional.interpolate(s_in, size=[128, 128], mode='nearest')

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

        return z, mu, sigma


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
                 exponential_weighting=True,
                 padding=True):
        super(PHISeg, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters

        self.latent_levels = (len(self.num_filters) - 1)

        self.loss_dict={}
        self.kl_divergence_loss_weight = 1.0

        self.beta = 1.0

        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.exponential_weighting = exponential_weighting
        self.exponential_weight = 2 # default was 4
        self.residual_multinoulli_loss_weight = 1.0

        self.kl_divergence_loss = 0
        self.reconstruction_loss = 0

        self.posterior = Posterior(input_channels, num_classes, num_filters, initializers=None, padding=True)
        self.likelihood = Likelihood(input_channels, num_classes, num_filters, initializers=None, apply_last_layer=True, padding=True)
        self.prior = Posterior(input_channels, num_classes, num_filters, initializers=None, padding=True, is_posterior=False)

        self.s_out_list = [None] * self.latent_levels
        self.s_out_list_with_softmax = [None] * self.latent_levels

    def sample_posterior(self):
        z_sample = [None] * self.latent_levels
        mu = self.posterior_mu
        sigma = self.posterior_sigma
        for i, _ in enumerate(z_sample):
            z_sample[i] = mu[i] + sigma[i] * torch.randn_like(sigma[i])

        return z_sample

    def sample_prior(self):
        z_sample = [None] * self.latent_levels
        mu = self.prior_mu
        sigma = self.prior_sigma
        for i, _ in enumerate(z_sample):
            z_sample[i] = mu[i] + sigma[i] * torch.randn_like(sigma[i])
        return z_sample

    def sample(self, testing=True):
        if testing:
            sample, _ = self.reconstruct(self.sample_prior(), use_softmax=False)
            return sample
        else:
            raise NotImplementedError

    def reconstruct(self, z_posterior, use_softmax=True):
        layer_recon = self.likelihood(z_posterior)
        return self.accumulate_output(layer_recon, use_softmax=use_softmax), layer_recon

    def forward(self, patch, mask, training=True):
        if training:
            self.posterior_latent_space, self.posterior_mu, self.posterior_sigma = self.posterior(patch, mask)
            self.s_out_list = self.likelihood(self.posterior_latent_space)

        self.prior_latent_space, self.prior_mu, self.prior_sigma = self.prior(patch)
        if not training:
            self.s_out_list = self.likelihood(self.prior_latent_space)

        return self.s_out_list

    def accumulate_output(self, output_list, use_softmax=False):
        s_accum = output_list[-1]
        for i in range(len(output_list) - 1):
            s_accum += output_list[i]
        if use_softmax:
            return torch.nn.functional.softmax(s_accum)
        return s_accum

    def KL_two_gauss_with_diag_cov(self, mu0, sigma0, mu1, sigma1):

        sigma0_fs = torch.mul(torch.flatten(sigma0, start_dim=1), torch.flatten(sigma0, start_dim=1))
        sigma1_fs = torch.mul(torch.flatten(sigma1, start_dim=1), torch.flatten(sigma0, start_dim=1))

        logsigma0_fs = torch.log(sigma0_fs + 1e-10)
        logsigma1_fs = torch.log(sigma1_fs + 1e-10)

        mu0_f = torch.flatten(mu0, start_dim=1)
        mu1_f = torch.flatten(mu1, start_dim=1)

        return torch.mean(
            0.5*torch.sum(
                torch.div(
                    sigma0_fs + torch.mul((mu1_f - mu0_f), (mu1_f - mu0_f)),
                    sigma1_fs + 1e-10)
                + logsigma1_fs - logsigma0_fs - 1, dim=1)
        )

    def calculate_hierarchical_KL_div_loss(self):

        prior_sigma_list = self.prior_sigma
        prior_mu_list = self.prior_mu
        posterior_sigma_list = self.posterior_sigma
        posterior_mu_list = self.posterior_mu

        loss_tot = 0

        if self.exponential_weighting:
            level_weights = [self.exponential_weight ** i for i in list(range(self.latent_levels))]
        else:
            level_weights = [1] * self.exp_config.latent_levels

        for ii, mu_i, sigma_i in zip(reversed(range(self.latent_levels)),
                                     reversed(posterior_mu_list),
                                     reversed(posterior_sigma_list)):

            self.loss_dict['KL_divergence_loss_lvl%d' % ii] = level_weights[ii]*self.KL_two_gauss_with_diag_cov(
                mu_i,
                sigma_i,
                prior_mu_list[ii],
                prior_sigma_list[ii])

            loss_tot += self.kl_divergence_loss_weight * self.loss_dict['KL_divergence_loss_lvl%d' % ii]

        return loss_tot

    def residual_multinoulli_loss(self, reconstruction, target):

        self.s_accumulated = [None] * self.latent_levels
        loss_tot = 0
        #target = target.view(-1, 128, 128).long()
        #criterion = torch.nn.BCEWithLogitsLoss(size_average=False, reduce=False, reduction='sum')

        # TODO: equivalent to tf.reduce_mean(tf.reduce_sum(CE_with_logits(), axis=1)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        for ii, s_ii in zip(reversed(range(self.latent_levels)),
                            reversed(reconstruction)):

            if ii == self.latent_levels-1:

                self.s_accumulated[ii] = s_ii
                self.loss_dict['residual_multinoulli_loss_lvl%d' % ii] = criterion(self.s_accumulated[ii], target)

            else:

                self.s_accumulated[ii] = self.s_accumulated[ii+1] + s_ii
                self.loss_dict['residual_multinoulli_loss_lvl%d' % ii] = criterion(self.s_accumulated[ii], target)

            loss_tot += self.residual_multinoulli_loss_weight * self.loss_dict['residual_multinoulli_loss_lvl%d' % ii]
        return loss_tot

    def kl_divergence(self):
        loss = self.calculate_hierarchical_KL_div_loss()
        return loss

    def elbo(self, segm, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """

        z_posterior = self.posterior_latent_space

        self.kl_divergence_loss = self.kl_divergence()

        # Here we use the posterior sample sampled above
        self.reconstruction, layer_reconstruction = self.reconstruct(z_posterior=z_posterior, use_softmax=False)

        self.reconstruction_loss = self.residual_multinoulli_loss(reconstruction=layer_reconstruction, target=segm)

        return self.reconstruction_loss + self.kl_divergence_loss_weight * self.kl_divergence_loss

    def loss(self, segm):
        return self.elbo(segm)
