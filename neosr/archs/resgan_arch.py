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
class resgan(nn.Module):


    def __init__(self, num_in_ch=3, num_feat=64, skip_connection=True, upscale=True, attention=False):
        super().__init__()
        self.skip_connection = skip_connection

        norm = spectral_norm

        self.conv0 = nn.Conv2d(num_in_ch, num_feat,
                               kernel_size=3, stride=1, padding=1)

        self.conv1 = norm(
            nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False))
        self.conv2 = norm(
            nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False))
        self.conv3 = norm(
            nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False))

        self.final_conv = norm(nn.Conv2d(num_feat, 16, 4, 2, 1))        

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.linear = nn.Linear(16, 1)
        self.act = nn.LeakyReLU(negative_slope=0.2)

        if attention:
            self.attention = simam_module()
        else:
            self.attention = nn.Identity()

    def forward(self, x):
        # downsample
        # x is NxNxC
        x0 = self.act(self.conv0(x))
        # x0 is NxNxF
        x1 = self.act(self.conv1(x0))
        x0residual = F.interpolate(x0, scale_factor=0.5, mode='bilinear')
        x1 = x1+x0residual
        # x1 is N/2xN/2xF
        x2 = self.act(self.conv2(x1))
        x1residual = F.interpolate(x1, scale_factor=0.5, mode='bilinear')
        x2 = x2+x1residual
        # x2 is N/4xN/4xF
        x3 = self.act(self.conv2(x2))
        x2residual = F.interpolate(x2, scale_factor=0.5, mode='bilinear')
        x3 = x3+x2residual
        # x3 is N/8xN/8xF

        x4 = self.act(self.final_conv(x3))
        # x4 is N/16xN/16x16

        # xavg is 1x1x16
        xavg = self.avg_pool(x4)

        out = self.linear(xavg.squeeze())

        return out
    
        
