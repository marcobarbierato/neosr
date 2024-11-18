import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from neosr.utils.registry import ARCH_REGISTRY

class simam_module(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        _, _, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

@ARCH_REGISTRY.register()
class simple_unet(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch=3, num_feat=64, skip_connection=True, upscale=True, attention=False):
        super().__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat,
                               kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(
            nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(
            nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        # upsample
        self.upscale = upscale
        self.conv4 = norm(
            nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv5 = norm(
            nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv6 = norm(nn.Conv2d(num_feat, 1, 3, 1, 1))
        
        if attention:
            self.attention = simam_module()
        else:
            self.attention = nn.Identity()

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x2 = self.attention(x2)
        # upsample
        x2 = F.interpolate(x2, scale_factor=2,
                           mode='bilinear', align_corners=False)
        x3 = F.leaky_relu(self.conv4(x2), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x3 = x3 + x1

        x3 = F.interpolate(x3, scale_factor=2,
                           mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv5(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x0

        out = self.attention(self.conv6(x4))

        return out

