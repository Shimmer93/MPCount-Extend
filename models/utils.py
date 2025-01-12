import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange
from math import sqrt
import functools

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, bn=False, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        if self.relu is not None:
            y = self.relu(y)
        return y
    
# fix the non-deterministic behavior of F.interpolate
# From: https://github.com/open-mmlab/mmsegmentation/issues/255
class FixedUpsample(nn.Module):

    def __init__(self, channel: int, scale_factor: int):
        super().__init__()
        # assert 'mode' not in kwargs and 'align_corners' not in kwargs and 'size' not in kwargs
        assert isinstance(scale_factor, int) and scale_factor > 1 and scale_factor % 2 == 0
        self.scale_factor = scale_factor
        kernel_size = scale_factor + 1  # keep kernel size being odd
        self.weight = nn.Parameter(
            torch.empty((1, 1, kernel_size, kernel_size), dtype=torch.float32).expand(channel, -1, -1, -1).clone()
        )
        self.conv = functools.partial(
            F.conv2d, weight=self.weight, bias=None, padding=scale_factor // 2, groups=channel
        )
        with torch.no_grad():
            self.weight.fill_(1 / (kernel_size * kernel_size))

    def forward(self, t):
        if t is None:
            return t
        return self.conv(F.interpolate(t, scale_factor=self.scale_factor, mode='nearest'))
    
class Upsample(nn.Module):
    def __init__(self, channel: int, scale_factor: int, deterministic=True):
        super().__init__()
        self.deterministic = deterministic
        if deterministic:
            self.upsample = FixedUpsample(channel, scale_factor)
        else:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

    def forward(self, t):
        return self.upsample(t)

def upsample_nearest(x, scale_factor=2):
    return F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    
class VGGMSEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None)
        self.enc1 = nn.Sequential(*list(vgg.features.children())[:23])
        self.enc2 = nn.Sequential(*list(vgg.features.children())[23:33])
        self.enc3 = nn.Sequential(*list(vgg.features.children())[33:43])

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        return x1, x2, x3
    
class MSDecoder(nn.Module):
    def __init__(self, deterministic=True):
        super().__init__()
        self.dec3 = nn.Sequential(
            ConvBlock(512, 1024, bn=True),
            ConvBlock(1024, 512, bn=True)
        )
        self.dec2 = nn.Sequential(
            ConvBlock(1024, 512, bn=True),
            ConvBlock(512, 256, bn=True)
        )
        self.dec1 = nn.Sequential(
            ConvBlock(512, 256, bn=True),
            ConvBlock(256, 128, bn=True)
        )

        self.up1 = Upsample(512, 2, deterministic)
        self.up2 = Upsample(256, 2, deterministic)

    def forward(self, x1, x2, x3):
        x = self.dec3(x3)
        y3 = x
        x = self.up1(x)
        x = torch.cat([x, x2], dim=1)

        x = self.dec2(x)
        y2 = x
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)

        x = self.dec1(x)
        y1 = x

        return y1, y2, y3
    
class MSAggregator(nn.Module):
    def __init__(self, dim_out, deterministic=True):
        super().__init__()
        self.agg = ConvBlock(512+256+128, dim_out, kernel_size=1, padding=0, bn=True)

        self.up1 = Upsample(256, 2, deterministic)
        self.up2 = Upsample(512, 4, deterministic)

    def forward(self, y1, y2, y3):
        y2 = self.up1(y2)
        y3 = self.up2(y3)
        y_cat = torch.cat([y1, y2, y3], dim=1)
        y = self.agg(y_cat)
        return y
    
class AttentionMemoryBank(nn.Module):
    def __init__(self, mem_size, mem_dim):
        super().__init__()
        self.mem = nn.Parameter(torch.FloatTensor(1, mem_dim, mem_size).normal_(0, 1.0))

    def forward(self, x, mask=None):
        if mask is not None:
            x = x.detach() * (~mask) + x * mask
        B, K, H, W = x.shape
        m = self.mem.repeat(B, 1, 1)
        m_key = m.transpose(1, 2)
        x = x.view(B, K, -1)
        logits = torch.bmm(m_key, x) / sqrt(K)
        x = torch.bmm(m_key.transpose(1, 2), F.softmax(logits, dim=1))
        x = x.view(B, K, H, W)
        return x, logits
    
def attn_consist_loss(logits1, logits2, type='mse'):
    if type == 'mse':
        return F.mse_loss(logits1, logits2)
    elif type == 'mse_softmax':
        return F.mse_loss(F.softmax(logits1, dim=1), F.softmax(logits2, dim=1))
    elif type == 'mse_log_softmax':
        return F.mse_loss(F.log_softmax(logits1, dim=1), F.log_softmax(logits2, dim=1))
    elif type == 'kld':
        return F.kl_div(F.log_softmax(logits1, dim=1), F.softmax(logits2, dim=1))
    elif type == 'jsd':
        p = F.softmax(logits1, dim=1)
        q = F.softmax(logits2, dim=1)
        m = 0.5 * (p + q)
        return 0.5 * F.kl_div(F.log_softmax(p, dim=1), m) + 0.5 * F.kl_div(F.log_softmax(q, dim=1), m)
    elif type == 'cos':
        return 1 - F.cosine_similarity(logits1, logits2, dim=1)
    else:
        raise ValueError('Invalid type')
    
def transform_cls_map(cls_map, scale_factor=4, thrs=0.5):
    cls_map = cls_map.clone().detach()
    cls_map = F.sigmoid(cls_map)
    cls_map[cls_map >= thrs] = 1
    cls_map[cls_map < thrs] = 0
    cls_map = upsample_nearest(cls_map, scale_factor=scale_factor)
    return cls_map
