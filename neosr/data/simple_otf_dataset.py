import os
import os.path as osp
import random
import time
import math
import cv2
import numpy as np
import torch
from torch.utils import data

from neosr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from neosr.data.transforms import basic_augment
from neosr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, scandir
from neosr.utils.registry import DATASET_REGISTRY
from neosr.utils.rng import rng

rng = rng()

@DATASET_REGISTRY.register()
class simple_otf(data.Dataset):
    """OTF degradation dataset. Originally from Real-ESRGAN

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.color = False if 'color' in self.opt and self.opt['color'] == 'y' else True
        self.gt_folder = opt['dataroot_gt']

        if opt.get('dataroot_lq', None) is not None:
            raise ValueError("'dataroot_lq' not supported by otf, please switch to paired")

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(
                    f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        elif 'meta_info' in self.opt and self.opt['meta_info'] is not None:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip().split(' ')[0] for line in fin]
                self.paths = [os.path.join(self.gt_folder, v) for v in paths]
        else:
            # disk backend
            # it will scan the whole folder to get meta info
            # it will be time-consuming for folders with too many files. It is recommended using an extra meta txt file
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
                if img_bytes is None:
                    raise ValueError(f'No data returned from path: {gt_path}')
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(
                    f'File client error: {e} in path {gt_path}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1

        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except AttributeError:
            raise AttributeError(gt_path)


        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = basic_augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # crop or pad to 512
        # TODO: 512 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        crop_pad_size = self.opt.get('crop_pad_size', 512)
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(
                img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size,
                            left:left + crop_pad_size, ...]


        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True, color=self.color)[0]
        # NOTE: using torch.tensor(device='cuda') won't work.
        # Keeping old constructor for now.

        return_d = {'gt': img_gt, 'gt_path': gt_path}
        return return_d

    def __len__(self):
        return len(self.paths)
