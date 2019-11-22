import torch
import pykitti
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


def transform_stereo_lidar(samples):
    for k in samples:
        samples[k] = TF.to_tensor(samples[k])
    return samples


class KittiDenseDrive(Dataset):
    """Dataset for prediction of dense (images) from the Kitti Dataset."""

    def __init__(self, basedir, date, drive, transform=None):
        """
        """
        self.basedir = basedir
        self.date = date
        self.drive = drive
        self.kitti = pykitti.raw(basedir, date, drive)
        self.transform = transform

    def __len__(self):
        return len(self.kitti)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # rgb - left, right
        cam2, cam3 = self.kitti.get_rgb(idx)

        # velodyne scan
        velo = self.kitti.get_velo(idx)

        samples = {"left_rgb": cam2, "right_rgb": cam3, "velo": velo}
        samples = self.transform(samples)

        return samples
