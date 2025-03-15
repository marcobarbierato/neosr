import numpy as np
import random
import torch
from torch.nn import functional as F

from neosr.models.default import default
from neosr.models.otf import otf
from neosr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from neosr.data.transforms import paired_random_crop
from neosr.utils import DiffJPEG
from neosr.utils.img_process_util import filter2D
from neosr.data.augmentations import apply_augment
from neosr.utils.rng import rng
from neosr.utils.registry import MODEL_REGISTRY

rng = rng()


@MODEL_REGISTRY.register()
class pancamotf(otf):
    """On The Fly degradations, based on RealESRGAN pipeline."""

    def __init__(self, opt):
        self.gt_size = opt["gt_size"]
        super().__init__(opt)

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if self.is_train:
            # training data synthesis
            self.gt = data['gt'].to(device=self.device, non_blocking=True)

            self.kernel1 = data['kernel1'].to(device=self.device, non_blocking=True)
            if data.get('kernel2', None) is not None:
                self.kernel2 = data['kernel2'].to(device=self.device, non_blocking=True)
            self.sinc_kernel = data['sinc_kernel'].to(device=self.device, non_blocking=True)
            
            if 'kernel1final' in data:
                linear_blur=True
                self.kernel1final = data['kernel1final'].to(device=self.device, non_blocking=True)
            distortion = self.gt[:, -1, :, :][:, None, :, :] #.clone()
            self.gt = self.gt[:, :-1, :, :] #.clone()
            ori_h, ori_w = self.gt.size()[2:4]
            


            # ----------------------- The first degradation process ----------------------- #
            

            # blur
            
            out = filter2D(self.gt, self.kernel1)
            
            # combines blurs
            if linear_blur:
                out2 = filter2D(self.gt, self.kernel1final)
                lin = torch.arange(0, 1, 1/self.gt.shape[2]).view(self.gt.shape[2], -1).expand(self.gt.shape[2], self.gt.shape[3])
                out = (1-lin)*out + lin*out2


            # random resize
            updown_type = random.choices(
                ['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = rng.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = rng.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            # out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if rng.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            
            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt.get('second_degrad_prob', 1):
                if rng.uniform() < self.opt['second_blur_prob']:
                    out = filter2D(out, self.kernel2)
                # random resize
                updown_type = random.choices(
                    ['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
                if updown_type == 'up':
                    scale = rng.uniform(1, self.opt['resize_range2'][1])
                elif updown_type == 'down':
                    scale = rng.uniform(self.opt['resize_range2'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                # out = F.interpolate(
                #     out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
                # add noise
                gray_noise_prob = self.opt['gray_noise_prob2']
                if rng.uniform() < self.opt['gaussian_noise_prob2']:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)

            
            
            # resize back + the final sinc filter
            #mode = random.choice(['area', 'bilinear', 'bicubic'])
            #out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
            # out = filter2D(out, self.sinc_kernel)
            
            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            #combine distortions for randomcrop+aug
            #distortion_lq = F.interpolate(distortion, scale_factor=1/self.opt['scale'], mode='bilinear')
            distortion_lq = distortion
            self.lq = torch.cat((self.lq, distortion_lq), dim=1)
            self.gt = torch.cat((self.gt, torch.zeros_like(distortion)), dim=1)
            
            # random crop
            gt_size = self.opt['gt_size']
            #(self.gt), self.lq = paired_random_crop([self.gt], self.lq, gt_size, self.opt['scale'])
            (self.gt), self.lq = paired_random_crop([self.gt], self.lq, gt_size, 1)

            # training pair pool
            self._dequeue_and_enqueue()
            # for the warning: grad and param do not obey the gradient layout contract
            self.lq = self.lq.contiguous()

            # augmentation error handling
            if self.aug is not None and self.gt_size % 4 != 0:
                msg = "The gt_size value must be a multiple of 4. Please change it."
                raise ValueError(msg)
            # apply augmentation
            if self.aug is not None:
                self.gt, self.lq = apply_augment(self.gt, self.lq, scale=self.scale, augs=self.aug, prob=self.aug_prob)

            self.lq = F.interpolate(self.lq, scale_factor=1/self.opt['scale'], mode=mode)
            self.gt = self.gt[:, :-1, :, :] # drop distortion from ground truth
                
                
        else:
            # for paired training or validation
            self.lq = data['lq'].to(device=self.device, memory_format=torch.channels_last, non_blocking=True)
            if 'gt' in data:
                self.gt = data['gt'].to(device=self.device, memory_format=torch.channels_last, non_blocking=True)
            
