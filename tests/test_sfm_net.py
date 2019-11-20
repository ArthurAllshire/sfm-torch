from sfm_torch import sfm_net
import numpy as np
import torch
import itertools


def test_conv_net_shapes():
    net = sfm_net.SfMConvNet(6, ret_embedding=True)
    # batch, channels, h, w
    image = torch.randn(4, 6, 384, 128)
    out, conv = net(image)
    assert out.size() == (
        4,
        32,
        384,
        128,
    ), "SfM Conv net primary output shape does not match"
    assert conv.size() == (
        4,
        1024,
        12,
        4,
    ), "SfM Conv net conv output shape does not match"


def test_structure_net_shapes():
    image_dim = (384 * 2, 128 * 2)
    net = sfm_net.Structure(image_dim, 3)

    image = torch.randn(2, 3, image_dim[0], image_dim[1])
    out = net(image)
    assert out.size() == (2, 1, image_dim[0], image_dim[1])


def test_motion_net_shapes():
    image_dim = (384, 128)
    net = sfm_net.Motion(image_dim, 6, 10)
    image = torch.randn(2, 6, image_dim[0], image_dim[1])

    masks, rot_k, t_k, p_k, rot_c, t_c, p_c = net.forward(image)

    assert masks.size() == (2, 10, image_dim[0], image_dim[1])
    assert rot_k.size() == (2, 10 * 3)
    assert t_k.size() == (2, 10 * 3)
    assert p_k.size() == (2, 10 * 3)
    assert rot_c.size() == (2, 1 * 3)
    assert t_c.size() == (2, 1 * 3)
    assert p_c.size() == (2, 1 * 3)


def test_sfm_shapes():
    image_dim = (384, 128)
    intrinsics = (0.5, 0.5, 1.0)

    def pinhole_x(x):
        return (x / image_dim[0] - intrinsics[0]) / intrinsics[2]

    def pinhole_y(y):
        return (y / image_dim[1] - intrinsics[1]) / intrinsics[2]

    n_segmentations = 3
    net = sfm_net.SfM(image_dim, n_segmentations, intrinsics)
    assert net.X.shape == (3, image_dim[0], image_dim[1])

    # test pinhole camera model
    for x, y in itertools.product([0, image_dim[0] - 1], [0, image_dim[1] - 1]):
        assert np.isclose(net.X[0][x][y], pinhole_x(x))
        assert np.isclose(net.X[1][x][y], pinhole_y(y))

    batch = 2
    image_1 = torch.randn(batch, 3, image_dim[0], image_dim[1])
    image_2 = torch.randn(batch, 3, image_dim[0], image_dim[1])
    flow, depth, masks = net.forward(image_1, image_2)

    assert flow.shape == (batch, 2, *image_dim)
    assert depth.shape == (batch, 1, *image_dim)
    assert masks.shape == (batch, n_segmentations, *image_dim)
