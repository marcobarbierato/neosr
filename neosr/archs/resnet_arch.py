import os
from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import resnet34
from torchvision.models import ResNet34_Weights

from neosr.utils.registry import ARCH_REGISTRY

VGG_PRETRAIN_PATH = 'experiments/pretrained_models/vgg19-dcbb9e9d.pth'
NAMES = {
    'vgg19': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    ]
}


@ARCH_REGISTRY.register()
class ResNetFeatureExtractor(nn.Module):
    """VGG network for feature extraction.

    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.

    Args:
        layer_name_list (list[str]): Forward function returns the corresponding
            features according to the layer_name_list.
            Example: {'relu1_1', 'relu2_1', 'relu3_1'}.
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image. Importantly,
            the input feature must in the range [0, 1]. Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        requires_grad (bool): If true, the parameters of VGG network will be
            optimized. Default: False.
        remove_pooling (bool): If true, the max pooling operations in VGG net
            will be removed. Default: False.
        pooling_stride (int): The stride of max pooling operation. Default: 2.
    """

    def __init__(self,
                 requires_grad=False):
        super(ResNetFeatureExtractor, self).__init__()

        res_net = resnet34(weights=ResNet34_Weights.DEFAULT)

        self.prenet = nn.Sequential(res_net.conv1, res_net.bn1, res_net.relu, res_net.maxpool)
        self.layer1= res_net.layer1
        self.layer2= res_net.layer2
        self.layer3= res_net.layer3
        self.layer4= res_net.layer4

        self.register_buffer('mean', torch.tensor(
            [0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1))
        # the std is for image with range [0, 1]
        self.register_buffer('std', torch.tensor(
            [0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        x = (x - self.mean) / self.std

        output = {}
        pre = self.prenet(x)
        o1 = self.layer1(pre)
        o2 = self.layer2(o1)
        o3 = self.layer3(o2)
        o4 = self.layer4(o3)

        output = {
            'feat1': o1.clone(),
            'feat2': o2.clone(),
            'feat3': o3.clone(),
            'feat4': o4.clone(),
        }

        return output