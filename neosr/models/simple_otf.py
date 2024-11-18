import numpy as np
import random
import torch
from torch.nn import functional as F

from neosr.models.default import default
from neosr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from neosr.data.transforms import paired_random_crop
from neosr.utils import DiffJPEG
from neosr.utils.img_process_util import filter2D
from neosr.data.augmentations import apply_augment
from neosr.utils.rng import rng
from neosr.utils.registry import MODEL_REGISTRY

rng = rng()


@MODEL_REGISTRY.register()
class simple_otf(default):
    """On The Fly degradations, based on RealESRGAN pipeline."""

    def __init__(self, opt):
        super().__init__(opt)
        # simulate JPEG compression artifacts
        self.queue_size = opt.get('queue_size', 180)
        self.gt_size = opt['gt_size']
        self.device = torch.device('cuda')

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w, device=self.device).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w, device=self.device).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size, device=self.device)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr +
                          b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr +
                          b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if self.is_train:
            # training data synthesis
            self.gt = data['gt'].to(device=self.device, non_blocking=True)

               
            ori_h, ori_w = self.gt.size()[2:4]
        
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out=self.gt
            out = F.interpolate(out, size=(
                ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
            
            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['gt_size']
            (self.gt), self.lq = paired_random_crop([self.gt], self.lq, gt_size,
                                                    self.opt['scale'])

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
        else:
            # for paired training or validation
            self.lq = data['lq'].to(device=self.device, memory_format=torch.channels_last, non_blocking=True)
            if 'gt' in data:
                self.gt = data['gt'].to(device=self.device, memory_format=torch.channels_last, non_blocking=True)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super().nondist_validation(
            dataloader, current_iter, tb_logger, save_img)
        self.is_train = True
