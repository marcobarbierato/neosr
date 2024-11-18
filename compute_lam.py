import datetime
import logging
import math
import sys
import time
import os
from os import path as osp

import torch

from neosr.models import build_model
from neosr.utils import (
    get_root_logger,
    get_time_str,
)
from neosr.utils.options import parse_options
from torchprofile import profile_macs


def compute_lam(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt["root_path"] = root_path

    # default device
    torch.set_default_device("cuda")

    # enable tensorfloat32 and possibly bfloat16 matmul
    fast_matmul = opt.get("fast_matmul", False)
    if fast_matmul:
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # create model
    model = build_model(opt)
    
    log_file = osp.join(opt["path"]["log"], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name="neosr", log_level=logging.INFO, log_file=log_file
    )
    logger.info(
        f"\n------------------------ neosr ------------------------\nPytorch Version: {torch.__version__}"
    )

    input_size= opt.get('input_size')
    if input_size is not None:
        inputs = torch.randn(1, input_size['channels'], input_size['size'], input_size['size'])
        macs = profile_macs(model.net_g, inputs)
        logger.info(f'Network MACs: {macs:,d}')
    

    window_size = 16  # Define windoes_size of D
    img_lr, img_hr = prepare_images('LAM_Demo/test_images/7.png') 
    tensor_lr = PIL2Tensor(img_lr)[:3] ; tensor_hr = PIL2Tensor(img_hr)[:3]
    cv2_lr = np.moveaxis(tensor_lr.numpy(), 0, 2) ; cv2_hr = np.moveaxis(tensor_hr.numpy(), 0, 2)

    w = 110  # The x coordinate of your select patch, 125 as an example
    h = 150  # The y coordinate of your select patch, 160 as an example

    draw_img = pil_to_cv2(img_hr)
    cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
    position_pil = cv2_to_pil(draw_img)


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    torch.multiprocessing.set_start_method('spawn')
    compute_lam(root_path)
