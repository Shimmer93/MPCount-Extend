import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from PIL import Image

from utils.misc import random_crop, get_padding, cal_inner_area, divide_img_into_patches

class Transform():
    def __init__(self, img_keys=[], map_keys=[], pt_keys=[]):
        self.locked = False
        self.set_keys(img_keys, map_keys, pt_keys)

    def set_keys(self, img_keys=[], map_keys=[], pt_keys=[]):
        if not self.locked:
            self.img_keys = img_keys
            self.map_keys = map_keys
            self.pt_keys = pt_keys

    def lock_keys(self):
        self.locked = True

    def unlock_keys(self):
        self.locked = False

    def __call__(self, sample):
        pass
    
class PadToFitCropSize(Transform):
    def __init__(self, crop_size, **kwargs):
        super().__init__(**kwargs)
        self.crop_size = crop_size

    def __call__(self, sample):
        w, h = sample['img_size']
        pad, h, w = get_padding(h, w, self.crop_size[0], self.crop_size[1])
        l, t = pad[0], pad[1]

        for key in self.img_keys + self.map_keys:
            sample[key] = TF.pad(sample[key], pad)

        for key in self.pt_keys:
            if len(sample[key]) > 0:
                sample[key] = sample[key] + [l, t]

        sample['img_size'] = [w, h]

        return sample
    
class PadToFitUnitSize(Transform):
    def __init__(self, unit_size, **kwargs):
        super().__init__(**kwargs)
        self.unit_size = unit_size

    def __call__(self, sample):
        if self.unit_size == 1:
            return sample
        w, h = sample['img_size']
        new_w = (w // self.unit_size + 1) * self.unit_size if w % self.unit_size != 0 else w
        new_h = (h // self.unit_size + 1) * self.unit_size if h % self.unit_size != 0 else h
        pad, new_h, new_w = get_padding(h, w, new_h, new_w)
        l, t = pad[0], pad[1]

        for key in self.img_keys + self.map_keys:
            sample[key] = TF.pad(sample[key], pad)

        for key in self.pt_keys:
            if len(sample[key]) > 0:
                sample[key] = sample[key] + [l, t]

        sample['img_size'] = [new_w, new_h]
        sample['pad'] = pad

        return sample
    
class RandomCrop(Transform):
    def __init__(self, crop_size, **kwargs):
        super().__init__(**kwargs)
        self.crop_size = crop_size

    def __call__(self, sample):
        w, h = sample['img_size']
        i, j = random_crop(h, w, self.crop_size[0], self.crop_size[1])

        for key in self.img_keys + self.map_keys:
            sample[key] = TF.crop(sample[key], i, j, self.crop_size[0], self.crop_size[1])

        for key in self.pt_keys:
            if len(sample[key]) > 0:
                sample[key] = sample[key] - [j, i]
                idx_mask = (sample[key][:, 0] >= 0) * (sample[key][:, 0] < self.crop_size[1]) * \
                        (sample[key][:, 1] >= 0) * (sample[key][:, 1] < self.crop_size[0])
                sample[key] = sample[key][idx_mask]
            else:
                sample[key] = np.empty((0, 2))

        sample['img_size'] = [self.crop_size[1], self.crop_size[0]]

        return sample
    
class DownsampleDensityMap(Transform):
    def __init__(self, factor, downsample_pt=True, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.downsample_pt = downsample_pt

    def __call__(self, sample):
        w, h = sample['img_size']

        for key in self.map_keys:
            down_w = w // self.factor
            down_h = h // self.factor
            sample[key] = sample[key].reshape([1, down_h, self.factor, down_w, self.factor]).sum([2, 4])

        if self.downsample_pt:
            for key in self.pt_keys:
                if len(sample[key]) > 0:
                    sample[key] = sample[key] // self.factor

        return sample
    
class HorizontalFlip(Transform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, sample):
        w, h = sample['img_size']

        for key in self.img_keys + self.map_keys:
            sample[key] = TF.hflip(sample[key])

        for key in self.pt_keys:
            if len(sample[key]) > 0:
                sample[key][:, 0] = w - sample[key][:, 0]

        return sample
    
class RandomApply(Transform):
    def __init__(self, transform, p, **kwargs):
        self.transform = transform
        self.transform.set_keys(**kwargs)
        self.p = p

    def set_keys(self, img_keys=[], map_keys=[], pt_keys=[]):
        self.transform.set_keys(img_keys, map_keys, pt_keys)
    
    def lock_keys(self):
        self.transform.lock_keys()

    def unlock_keys(self):
        self.transform.unlock_keys()

    def __call__(self, sample):
        if random.random() < self.p:
            sample = self.transform(sample)
        return sample

class ToTensor(Transform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, sample):
        for key in self.img_keys + self.map_keys:
            if isinstance(sample[key], Image.Image):
                sample[key] = TF.to_tensor(sample[key])
                
        for key in self.pt_keys:
            sample[key] = torch.from_numpy(sample[key]).float()

        return sample

class ImageTransformWrapper(Transform):
    def __init__(self, transform, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform

    def __call__(self, sample):
        for key in self.img_keys:
            sample[key] = self.transform(sample[key])
        return sample
        
class Compose(Transform):
    def __init__(self, transforms, **kwargs):
        self.transforms = transforms
        for t in self.transforms:
            t.set_keys(**kwargs)

    def set_keys(self, img_keys=[], map_keys=[], pt_keys=[]):
        for t in self.transforms:
            t.set_keys(img_keys, map_keys, pt_keys)

    def lock_keys(self):
        for t in self.transforms:
            t.lock_keys()

    def unlock_keys(self):
        for t in self.transforms:
            t.unlock_keys()

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample