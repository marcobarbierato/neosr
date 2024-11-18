from os import path as osp
import numpy as np
import torch

from torch.utils import data
from torchvision.transforms.functional import normalize

from neosr.data.single_dataset import single
from neosr.data.data_util import paths_from_lmdb
from neosr.utils import FileClient, imfrombytes, img2tensor, scandir
from neosr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class pancamsingle(single):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super().__init__(opt)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load lq image
        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')

        try:
            img_lq = imfrombytes(img_bytes, float32=True, flag='unchanged')
        except AttributeError:
            raise AttributeError(lq_path)
        
    
        img_lq = img_lq[:, :, :-1]
        dist = img_lq[:, :, -1]
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True, color=self.color)
        distortion = torch.from_numpy(dist.astype('float32')).unsqueeze(0)
        img_lq = torch.cat((img_lq, distortion), dim=0)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
        return {'lq': img_lq, 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths)
