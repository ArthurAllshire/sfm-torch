import torch

import torch.nn as nn
import torch.nn.functional as F

image_dim = (384, 128)
base_channels = 32


class SFMConvNet(nn.Module):
    """ConvNet from the paper.

    Used for both the Structure and Motion networks - both have the same
    underlying encoder/decoder structure.
    """

    n_conv_layers = 11
    n_deconv_layers = 5
    kernel_size = 3

    inner_channels = 1024

    def __init__(self, input_channels, use_skips=True, ret_embedding=False):
        super(SFMConvNet, self).__init__()

        self.use_skips = use_skips
        self.ret_embedding = ret_embedding

        # number of channels goes from input_channles in onput to
        channels_conv_layers = [input_channels] + [
            base_channels * 2 ** (i // 2) for i in range(1, self.n_conv_layers + 1)
        ]
        strides_per_layer = [1 if i % 2 == 0 else 2 for i in range(self.n_conv_layers)]

        # TODO: Add batch norm

        self.conv = [
            nn.Conv2d(
                channels_conv_layers[i],
                channels_conv_layers[i + 1],
                self.kernel_size,
                padding=1,
                stride=strides_per_layer[i],
            )
            for i in range(self.n_conv_layers)
        ]

        # number of channels goes from 1024 on input to 32 (base_channels) on output
        channels_deconv_layers = [
            base_channels * 2 ** i for i in reversed(range(self.n_deconv_layers + 1))
        ]
        self.deconv = [
            nn.ConvTranspose2d(
                channels_deconv_layers[i],
                channels_deconv_layers[i + 1],
                self.kernel_size,
                padding=1,
                output_padding=1,
                stride=2,
            )
            for i in range(self.n_deconv_layers)
        ]

    def forward(self, x):
        """Perform a forward pass on the ConvNet.

        Args:
            x: Pytorch Tensor with shape (batch_size, n_channels, width, height)

        Returns:
            Pytorch Tensor representing the output of the ConvNet.
            This tensor has shape (batch_size, base_channels, width, height).
            If ret_embedding was set, returns the middle layer of the network
            with shape (batch_size, max_channels, width, height), where max_channels
            is the number of channels in the middle of the network (1024).
        """
        skips = []
        for i, layer in enumerate(self.conv):
            x = F.relu(layer(x))
            # store intermediate representations to use in skip connections later
            # only store every second layer (before dimenson reduction from stride 2)
            # and don't store the last layer
            if self.use_skips and i % 2 == 0 and not i == len(self.conv) - 1:
                skips.append(x)
        skips.reverse()

        if self.ret_embedding:
            conv_out = x
        for i, layer in enumerate(self.deconv):
            # TODO: come back and check this implementation of skip connections
            x = layer(x)
            x += skips[i]

        if self.ret_embedding:
            return x, conv_out
        return x


class StructureNet(nn.Module):
    """Structure Net from the paper."""

    def __init__(self, input_channels):
        """
        Args:
            input_channels: number of channels in the input image
        """
        super(StructureNet, self).__init__()

        self.conv_net = SFMConvNet(input_channels, use_skips=True, ret_embedding=False)
        # 1x1 convolution (equivalent to a FC neural network applied pixelwise
        # with channels as inputs)
        self.conv_output = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x):
        """
        Perform a forward pass on the Structure Network.

        Args:
            x: Pytorch Tensor with shape (batch_size, n_channels, width, height)
        Returns:
            Pytorch Tensor representing the output of the structure network,
            the depth mask. This result must be passed through the pinhole
            camera model to create the point cloud.
            This tensor has shape (batch_size, 1, width, height)

        Note that the result from this is simply the depth mask - must pass
        through the pinhole camera model to convert to point cloud.
        """
        x = self.conv_net(x)
        x = self.conv_output(x)
        return x


class MotionNet(nn.Module):
    """Motion Net from the paper."""

    n_fc_layers = 2
    fc_dim = 512
    motion_representation_parameters = 9

    def __init__(self, input_channels, n_segmentations):
        """
        Initialize the MotionNet.

        Args:
            input_channels: combined number of channels in the pair of input images.
            n_segmentaitons: number of segmentation masks to predict (K in the paper).
        """
        super(MotionNet, self).__init__()

        self.conv_net = SFMConvNet(input_channels, use_skips=True, ret_embedding=True)

        # 1x1 convolution to produce the segmentations
        self.conv_output = nn.Conv2d(base_channels, n_segmentations, 1)

        # TODO: finish implementation of motion and object prediction
        inner_c = self.conv_net.inner_channels
        linear_input_dim = int(inner_c*image_dim[0]*image_dim[1]/(inner_c/base_channels)**2)

        self.fc = torch.nn.Sequential(
            nn.Linear(
                linear_input_dim, self.fc_dim
            ),
            nn.ReLU(),
            nn.Linear(
                self.fc_dim,
                self.motion_representation_parameters * (n_segmentations + 1),
            ),
        )

    def forward(self, x):
        """Perform a forward past on the Motion Network.

        Args:
            x: Pytorch Tensor with shape (batch_size, n_channels, width, height)
        Returns:
            Pytorch Tensor representing the output of the motion network.
            This tensor has shape (batch_size, n_segmentations, width, height).
        """
        x, embedding = self.conv_net(x)
        x = self.conv_output(x)
