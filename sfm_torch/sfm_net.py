import torch
import torch.nn as nn

base_channels = 32


class SfM(nn.Module):
    """SfM-Net from the paper"""

    def __init__(self, image_dim, n_segmentations, intrinsics):
        """
        TODO docstring!
        """
        super(SfM, self).__init__()

        self.image_dim = image_dim
        self.intrinsics = intrinsics
        self.structure = Structure(n_segmentations, 3)
        self.motion = Motion(image_dim, 6, n_segmentations)
        self.n_segmentations = n_segmentations

        # sets up the parameters from equation (1) of the paper
        x = (
            torch.arange(self.image_dim[0], dtype=torch.float)
            .repeat(self.image_dim[1], 1)
            .t()
        )
        x = self.pinhole_model(x, image_dim[0], self.intrinsics[0])
        y = torch.arange(self.image_dim[1], dtype=torch.float).repeat(
            self.image_dim[0], 1
        )
        y = self.pinhole_model(y, image_dim[1], self.intrinsics[1])
        z = torch.ones(self.image_dim[0], self.image_dim[1]) * self.intrinsics[2]
        self.X = torch.stack([x, y, z]) / self.intrinsics[2]

    def pinhole_model(self, px_idx, px_size, physical_size):
        """Computes the actual position on the dimension with a pinhole camera model.

        Args:
            px_idx: position in the dimension in pixels.
            px_size: size of the dimension in pixels (image width or height).
            physical_size: physical size of the dimension of the camera chip in {m, cm, ft, etc}.
        Returns:
            The position along the dimension (from equation 1 in the SfM paper).
        """
        return px_idx / px_size - physical_size

    def rotation_tensor(self, sin_alpha, sin_beta, sin_gamma):
        """Computes the rotation tensor representing the rotation matrices
        for each object for a batch of images.

        Args:
            sin_alpha: sin of the alpha euler angle as predicted by the network.
            sin_beta: sin of the alpha euler angle as predicted by the network.
            sin_gamma: sin of the alpha euler angle as predicted by the network.
        Returns:
            tensor packing together the rotation matrices for each mask in each
            batch, shape (batch_size, n_masks, 3, 3)
        """
        in_sz = sin_alpha.size()
        [cos_alpha, cos_beta, cos_gamma] = [
            torch.cos(torch.asin(s)) for s in [sin_alpha, sin_beta, sin_gamma]
        ]

        def zero():
            return torch.zeros(in_sz)

        def one():
            return torch.zeros(in_sz)

        # Yuck! Is there a cleaner way of doing this?
        R_alpha = torch.empty((3, 3) + sin_alpha.size())
        R_alpha[0] = torch.stack([cos_alpha, -sin_alpha, zero()])
        R_alpha[1] = torch.stack([sin_alpha, cos_alpha, zero()])
        R_alpha[2] = torch.stack([zero(), zero(), one()])
        R_beta = torch.empty((3, 3) + sin_alpha.size())
        R_beta[0] = torch.stack([cos_beta, zero(), sin_beta])
        R_beta[1] = torch.stack([zero(), one(), zero()])
        R_beta[2] = torch.stack([-sin_beta, zero(), cos_beta])
        R_gamma = torch.empty((3, 3) + sin_alpha.size())
        R_gamma[0] = torch.stack([one(), zero(), zero()])
        R_gamma[1] = torch.stack([zero(), cos_gamma, -sin_gamma])
        R_gamma[2] = torch.stack([zero(), sin_gamma, cos_gamma])
        [R_alpha, R_beta, R_gamma] = [
            m.view(in_sz[0], -1, 3, 3) for m in [R_alpha, R_beta, R_gamma]
        ]
        R = torch.matmul(torch.matmul(R_alpha, R_beta), R_gamma)
        return R

    def forward(self, image_1, image_2):
        """Perform a forward pass on the SfM network.

        Args:
            image_1: image for the current timestep (I_t in the paper).
            image_2: image for the next timestep (I_t+1 in the paper).

        Returns:
            Pytorch Tensor representing the flow between the two frames.
        """
        batch_size = image_1.size()[0]
        d_t = self.structure.forward(image_1)
        # Apply equation 1 from the paper
        X = self.X.repeat(batch_size, 1, 1, 1) * d_t

        masks, rot_k, t_k, p_k, rot_c, t_c, p_c = self.motion.forward(
            torch.cat([image_1, image_2], dim=1)
        )
        R_k = self.rotation_tensor(
            *[
                rot_k[:, i * self.n_segmentations : (i + 1) * self.n_segmentations]
                for i in range(3)
            ]
        )
        R_c = self.rotation_tensor(*[rot_c[:, i : i + 1] for i in range(3)])

        # TODO: finish implementation of this function
        return None


class SfMConvNet(nn.Module):
    """ConvNet from the paper.

    Used for both the Structure and Motion networks - both have the same
    underlying encoder/decoder structure.
    """

    n_conv_layers = 11
    n_deconv_layers = 5
    kernel_size = 3

    inner_channels = 1024

    def __init__(self, input_channels, use_skips=True, ret_embedding=False):
        super(SfMConvNet, self).__init__()

        self.use_skips = use_skips
        self.ret_embedding = ret_embedding

        # number of channels goes from input_channles in onput to
        channels_conv_layers = [input_channels] + [
            base_channels * 2 ** (i // 2) for i in range(1, self.n_conv_layers + 1)
        ]
        strides_per_layer = [1 if i % 2 == 0 else 2 for i in range(self.n_conv_layers)]

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
        self.batch_norm = [
            nn.BatchNorm2d(channels_conv_layers[i + 1])
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
            x = torch.relu(layer(x))
            x = self.batch_norm[i](x)
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


class Structure(nn.Module):
    """Structure Net from the paper."""

    min_depth = 1.0
    max_depth = 100.0

    def __init__(self, image_dim, input_channels):
        """
        Args:
            input_channels: number of channels in the input image
        """
        super(Structure, self).__init__()

        self.conv_net = SfMConvNet(input_channels, use_skips=True, ret_embedding=False)
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
        # relu + depth bias of 1 as noted in the paper to prevent small/negative depth values
        x = torch.relu(self.conv_output(x)) + self.min_depth
        # clamp x to be less than 100
        x = torch.clamp(x, max=self.max_depth)
        return x


class Motion(nn.Module):
    """Motion Net from the paper."""

    # size of the fully connected layer on the ouput of the emdedding
    fc_dim = 512
    # dimensions
    R_dim = 3
    t_dim = 3
    p_dim = 3

    def __init__(self, image_dim, input_channels, n_segmentations):
        """
        Initialize the Motion.

        Args:
            input_channels: combined number of channels in the pair of input images.
            n_segmentaitons: number of segmentation masks to predict (K in the paper).
        """
        super(Motion, self).__init__()

        self.n_segmentations = n_segmentations

        self.conv_net = SfMConvNet(input_channels, use_skips=True, ret_embedding=True)

        # 1x1 convolution to produce the segmentations
        self.conv_output = nn.Conv2d(base_channels, n_segmentations, 1)

        # TODO: finish implementation of motion and object prediction
        inner_c = self.conv_net.inner_channels
        self.linear_input_dim = int(
            inner_c * image_dim[0] * image_dim[1] / (inner_c / base_channels) ** 2
        )

        self.R_sz = self.R_dim * (self.n_segmentations + 1)
        self.t_sz = self.t_dim * (self.n_segmentations + 1)
        self.p_sz = self.p_dim * (self.n_segmentations + 1)
        # one extra for the camera parameters
        self.linear_output_dim = self.R_sz + self.t_sz + self.p_sz

        self.fc = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.linear_input_dim, self.fc_dim),
            nn.ReLU(),
            nn.Linear(self.fc_dim, self.linear_output_dim,),
        )

    def forward(self, x):
        """Perform a forward past on the Motion Network.

        Args:
            x: Pytorch Tensor with shape (batch_size, n_channels, width, height),
            representing the concatenated input images.
        Returns:
            List of tensors containing:
            The segmentation masks (batch_sz, n_masks, width, height)
            Rotation of each segmentation mask, sin(alpha), sin(beta), sin(gamma),
            packed togother in that order, shape (batch_sz, n_masks*3)
            Translation of each segmentation mask, t_c, shape (batch_sz, n_masks*3)
            Rotation of the camera, R_c, shape (batch_sz, 3)
            Rotation of the camera, sin(alpha), sin(beta), sin(gamma), in that order,
            shape (batch_sz, 3)
            Translation of the camera, t_c, shape (batch_sz, 3)
            The pivot points of the masks' rotation, t_m, shape (batch_sz, n_masks*3)
            The pivot points of the camera rotation, t_c, shape (batch_sz, 3)
        """

        masks, embedding = self.conv_net(x)
        masks = self.conv_output(masks)
        masks = torch.sigmoid(masks)

        batch_size = masks.size()[0]
        motion_output = self.fc(embedding).view(self.linear_output_dim, -1)
        # R and t pack together the R, t, and p information for both the n_segmentations masks
        # as well as for the camera
        R = torch.tanh(motion_output[0 : self.R_sz])
        t = torch.sigmoid(motion_output[self.R_sz : self.R_sz + self.t_sz])
        p = torch.sigmoid(
            motion_output[self.R_sz + self.t_sz : self.R_sz + self.t_sz + self.p_sz]
        )
        motion = [
            R[: -self.R_dim],
            t[: -self.t_dim],
            p[: -self.p_dim],
            R[-self.R_dim :],
            t[-self.t_dim :],
            p[-self.p_dim :],
        ]
        motion = [r.view(batch_size, -1) for r in motion]
        return [masks] + motion
