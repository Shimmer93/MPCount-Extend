import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange
from math import sqrt
import functools

from models.utils import ConvBlock, Upsample, upsample_nearest, VGGMSEncoder, \
                         MSDecoder, MSAggregator, AttentionMemoryBank, attn_consist_loss, \
                         transform_cls_map

class MPCountExtend(nn.Module):
    def __init__(self, pretrained=True, mem_size=1024, mem_dim=256, cls_thrs=0.5, \
                 err_thrs=0.5, den_drop=0.5, cls_drop=0.5, deterministic=True, acl_type='mse'):
        super().__init__()

        self.cls_thrs = cls_thrs
        self.err_thrs = err_thrs
        self.den_drop = den_drop
        self.acl_type = acl_type

        self.enc = VGGMSEncoder(pretrained)
        self.dec = MSDecoder(deterministic)
        self.agg = MSAggregator(mem_dim, deterministic)
        self.amb = AttentionMemoryBank(mem_size, mem_dim)

        self.cls_head = nn.Sequential(
            ConvBlock(512, 256, bn=True),
            nn.Dropout2d(p=cls_drop),
            ConvBlock(256, 1, kernel_size=1, padding=0, relu=False)
        )
        self.err_head = ConvBlock(mem_dim, mem_dim, kernel_size=1, padding=0, bn=False, relu=False)
        self.den_head = ConvBlock(mem_dim, 1, kernel_size=1, padding=0)

        self.up = Upsample(1, 4, deterministic)

    def forward(self, x, return_dict=False):
        x1, x2, x3 = self.enc(x)
        y1, y2, y3 = self.dec(x1, x2, x3)
        y = self.agg(y1, y2, y3)
        e_mask = self.err_head(y) > 0
        y, _ = self.amb(y * e_mask, e_mask)

        c_ = self.cls_head(x3)
        c = transform_cls_map(c_, 4, self.cls_thrs)

        d_ = self.den_head(y)
        d = self.up(d_ * c)

        if return_dict:
            return {'dmap': d, 'dmap_raw': d_, 'pcm': c_}
        else:
            return d
        
    def forward_train(self, x1, x2):
        x1_1, x1_2, x1_3 = self.enc(x1)
        x2_1, x2_2, x2_3 = self.enc(x2)
        y1_1, y1_2, y1_3 = self.dec(x1_1, x1_2, x1_3)
        y2_1, y2_2, y2_3 = self.dec(x2_1, x2_2, x2_3)
        y1 = self.agg(y1_1, y1_2, y1_3)
        y2 = self.agg(y2_1, y2_2, y2_3)

        y1_in = F.instance_norm(y1, eps=1e-5)
        y2_in = F.instance_norm(y2, eps=1e-5)
        e = torch.abs(y1_in - y2_in)
        e_mask = (e < self.err_thrs).clone().detach()
        y1 = F.dropout2d(y1 * e_mask, self.den_drop)
        y2 = F.dropout2d(y2 * e_mask, self.den_drop)

        e_mask_pred1 = self.err_head(y1)
        e_mask_pred2 = self.err_head(y2)
        l_err = F.binary_cross_entropy_with_logits(e_mask_pred1, e_mask.to(e_mask_pred1.dtype)) + \
                F.binary_cross_entropy_with_logits(e_mask_pred2, e_mask.to(e_mask_pred2.dtype))

        y1, logits1 = self.amb(y1, e_mask)
        y2, logits2 = self.amb(y2, e_mask)
        l_acl = attn_consist_loss(logits1, logits2, self.acl_type)

        c1_ = self.cls_head(x1_3)
        c2_ = self.cls_head(x2_3)
        c1 = transform_cls_map(c1_, 4, self.cls_thrs)
        c2 = transform_cls_map(c2_, 4, self.cls_thrs)

        d1_ = self.den_head(y1)
        d2_ = self.den_head(y2)
        d1 = self.up(d1_ * c1)
        d2 = self.up(d2_ * c2)

        return {'dmap': d1, 'dmap_raw': d1_, 'pcm': c1_}, {'dmap': d2, 'dmap_raw': d2_, 'pcm': c2_}, l_err, l_acl
    