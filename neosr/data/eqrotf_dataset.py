import os
import os.path as osp
import random
import time
import math
import cv2
import numpy as np
import torch
from torch.utils import data

from neosr.data.degradations import circular_lowpass_kernel, random_mixed_kernels, random_bivariate_Gaussian
from neosr.data.transforms import basic_augment
from neosr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, scandir
from neosr.utils.registry import DATASET_REGISTRY
from neosr.utils.rng import rng

rng = rng()

@DATASET_REGISTRY.register()
class eqrotf(data.Dataset):
    """OTF degradation dataset. Originally from Real-ESRGAN, modified for equirectangular images

    """

    def __init__(self, opt):
        super(eqrotf, self).__init__()
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

        # blur settings for the first degradation
        self.kernel_list = opt['kernel_list']
        # a list for each kernel probability
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        # betag used in generalized Gaussian blur kernels
        self.betag_range = opt['betag_range']
        # betap used in plateau blur kernels
        self.betap_range = opt['betap_range']
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        # kernel size ranges from 7 to 21
        self.kernel_range = opt.get('kernel_range', [2 * v + 1 for v in range(3, 11)])  # kernel size ranges from 7 to 21
        # convolving with pulse tensor brings no blurry effect

        # Note: this operation must run on cpu, otherwise CUDAPrefetcher will fail
        with torch.device('cpu'):
            self.pulse_tensor = torch.zeros(21, 21, dtype=torch.float32)
        self.pulse_tensor[10, 10] = 1

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
            img_gt = imfrombytes(img_bytes, float32=True, flag='unchanged')
        except AttributeError:
            raise AttributeError(gt_path)


        # ----------------- Check for 4th channel --------------- #
        if img_gt.shape[2]==3:
            h, w = img_gt.shape[0:2]
            dist = np.ones((h, w, 1))
            img_gt = np.concatenate((img_gt, dist), 2)


        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = basic_augment(img_gt, self.opt['use_hflip'], self.opt['use_rot']) # use_rot = False!

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
        # ------------------------ Choose kernel variance based on eqr angle ------------------------ #
        
        angle0 = np.arccos( img_gt[0, 0, 3] ) / (np.pi/2) # in [0,1]
        angle1 = np.arccos( img_gt[-1, 0, 3] ) / (np.pi/2)
        angle_mid = np.abs(angle0-angle1)/2
        
        sigma = 2*(1-angle_mid)+10*angle_mid # this constants can be changed
        sigma_var = 1+ 2 * np.sqrt(angle_mid)
        sigma_rangey= [sigma-sigma_var, sigma+sigma_var] # this interval is always positive
        rotation_range = [-1e-3, 1e-3]
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)

        #kernel = random_bivariate_Gaussian(kernel_size, self.blur_sigma, sigma_rangey, rotation_range=rotation_range, noise_range=None, isotropic=False)
        
        kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                sigma_rangey, rotation_range,
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        #kernel2 = random_bivariate_Gaussian(kernel_size, self.blur_sigmax, sigma_rangey, rotation_range=rotation_range, noise_range=None, isotropic=False)

        kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma2,
                sigma_rangey, rotation_range,
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if rng.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = rng.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(
                omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True, color=self.color)[0]
        # NOTE: using torch.tensor(device='cuda') won't work.
        # Keeping old constructor for now.
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2,
                    'sinc_kernel': sinc_kernel, 'gt_path': gt_path}
        return return_d

    def __len__(self):
        return len(self.paths)
