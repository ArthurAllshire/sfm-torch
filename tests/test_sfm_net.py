from sfm_torch import sfm_net
import torch


def test_conv_net_shapes():
    net = sfm_net.SFMConvNet(6, ret_embedding=True)
    # batch, channels, h, w
    image = torch.randn(4, 6, 384, 128)
    out, conv = net(image)
    assert out.size() == (
        4,
        32,
        384,
        128,
    ), "SFM Conv net primary output shape does not match"
    assert conv.size() == (
        4,
        1024,
        12,
        4,
    ), "SFM Conv net conv output shape does not match"


def test_structure_net_shapes():
    net = sfm_net.StructureNet(3)

    image = torch.randn(2, 3, 384 * 2, 128 * 2)
    out = net(image)
    assert out.size() == (2, 1, 384 * 2, 128 * 2)


def test_motion_net_shapes():
    net = sfm_net.MotionNet(6, 10)
    image = torch.randn(2, 6, 384, 128)

    masks, R_k, R_c, t_k, t_c, p = net.forward(image)
    masks, R_k, t_k, R_c, t_c, p = net.forward(image)

    assert masks.size() == (2, 10, 384, 128)
    assert R_k.size() == (2, 10 * 3)
    assert t_k.size() == (2, 10 * 3)
    assert R_c.size() == (2, 1 * 3)
    assert t_c.size() == (2, 1 * 3)
    assert p.size() == (2, 3 * 1)
