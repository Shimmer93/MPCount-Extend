import torch
import numpy as np
from PIL import Image

from datasets.den_dataset import DensityMapDataset
from datasets.transforms import *

class MPCountDataset(DensityMapDataset):
    def __init__(self, data_dir, split, transforms):
        super().__init__(data_dir, split, transforms)

    def __len__(self):
        return len(self.img_fns)

    def __getitem__(self, idx):
        img_fn = self.img_fns[idx]
        name = img_fn.split('/')[-1].split('.')[0]
        img = Image.open(img_fn).convert('RGB')

        img_aug = img.copy()

        pt_fn = img_fn.replace('.jpg', '.npy')
        pt = np.load(pt_fn)

        dmap_fn = img_fn.replace('.jpg', '_dmap.npy')
        dmap = np.load(dmap_fn)
        dmap = dmap[np.newaxis, :, :]
        dmap = torch.from_numpy(dmap).float()

        sample = {'name': name, 'img': img, 'img_aug': img_aug, 'pt': pt, 'dmap': dmap, 'img_size': img.size}
        sample = self.transforms(sample)
        pcm = sample['dmap'].clone().reshape(1, sample['dmap'].shape[1]//16, 16, sample['dmap'].shape[2]//16, 16).sum(dim=(2, 4))
        pcm = (pcm > 0).float()
        sample['pcm'] = pcm
        sample['count'] = torch.tensor(len(sample['pt']))

        return sample
    
    @staticmethod
    def collate_fn(batch):
        batch_data = {}
        for key in ['img', 'img_aug', 'dmap', 'pcm', 'count']:
            batch_data[key] = torch.stack([sample[key] for sample in batch], dim=0)
        for key in ['name', 'pt']:
            batch_data[key] = [sample[key] for sample in batch]
        return batch_data

    @staticmethod
    def get_train_transforms(hparams):
        img_transforms_wrapper1 = ImageTransformWrapper(
            T.Compose([
            T.RandomApply([T.Grayscale(num_output_channels=3)], p=0.1),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
            img_keys = ['img']
        )
        img_transforms_wrapper1.lock_keys()
        img_transforms_wrapper2 = ImageTransformWrapper(
            T.Compose([
            T.RandomApply([T.Grayscale(num_output_channels=3)], p=0.1),
            T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=1)], p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=5, p=0.5),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
            img_keys = ['img_aug']
        )
        img_transforms_wrapper2.lock_keys()
        transforms = Compose(
            [
                PadToFitCropSize(hparams.crop_size),
                RandomCrop(hparams.crop_size),
                DownsampleDensityMap(hparams.downsample_factor),
                RandomApply(transform=HorizontalFlip(), p=0.5),
                ToTensor(),
                img_transforms_wrapper1,
                img_transforms_wrapper2
            ],
            img_keys = ['img', 'img_aug'], 
            pt_keys = ['pt'],
            map_keys = ['dmap']
        )

        return transforms
    
    @staticmethod
    def get_val_transforms(hparams):
        img_transforms_wrapper1 = ImageTransformWrapper(
            T.Compose([
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
            img_keys = ['img']
        )
        img_transforms_wrapper1.lock_keys()
        img_transforms_wrapper2 = ImageTransformWrapper(
            T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=1)], p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=5, p=0.5),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
            img_keys = ['img_aug']
        )
        img_transforms_wrapper2.lock_keys()
        transforms = Compose(
            [
                PadToFitUnitSize(hparams.unit_size),
                ToTensor(),
                img_transforms_wrapper1,
                img_transforms_wrapper2
            ],
            img_keys = ['img', 'img_aug'], 
            pt_keys = ['pt'],
            map_keys = ['dmap']
        )

        return transforms