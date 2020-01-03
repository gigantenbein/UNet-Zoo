"""Custom layers with activation and norm for code readability"""
import torch
import torch.nn as nn
import revtorch as rv


class Conv2D(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, activation=torch.nn.ReLU, norm=torch.nn.BatchNorm2d,
                 norm_before_activation=True):
        super(Conv2D, self).__init__()

        if kernel_size == 3:
            padding = 1
        else:
            padding = 0

        layers = []
        layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding))
        if norm_before_activation:
            layers.append(norm(num_features=output_dim, eps=1e-3, momentum=0.01))
            layers.append(activation())
        else:
            layers.append(activation())
            layers.append(norm(num_features=output_dim, eps=1e-3, momentum=0.01))

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


class ReversibleSequence(nn.Module):
    """This class implements a a reversible sequence made out of n convolutions with ReLU activation and BN
        There is an initial 1x1 convolution to get to the desired number of channels.
    """
    def __init__(self, input_dim, output_dim, reversible_depth=3, kernel=3):
        super(ReversibleSequence, self).__init__()

        if input_dim  != output_dim:
            self.inital_conv = Conv2D(input_dim, output_dim, kernel_size=1)
        else:
            self.inital_conv = nn.Identity()

        blocks = []
        for i in range(reversible_depth):

            #f and g must both be a nn.Module whos output has the same shape as its input
            f_func = nn.Sequential(Conv2D(output_dim//2, output_dim//2, kernel_size=kernel, padding=1))
            g_func = nn.Sequential(Conv2D(output_dim//2, output_dim//2, kernel_size=kernel, padding=1))

            #we construct a reversible block with our F and G functions
            blocks.append(rv.ReversibleBlock(f_func, g_func))

        #pack all reversible blocks into a reversible sequence
        self.sequence = rv.ReversibleSequence(nn.ModuleList(blocks))

    def forward(self, x):
        x = self.inital_conv(x)
        return self.sequence(x)
