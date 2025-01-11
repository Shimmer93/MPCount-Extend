import torch
import numpy as np
from PIL import Image

from datasets.base_dataset import BaseDataset
from datasets.transforms import *

class DensityMapDataset(BaseDataset):
    def __init__(self, data_dir, split, transforms):
        super().__init__(data_dir, split, transforms)

    def __len__(self):
        return len(self.img_fns)

    def __getitem__(self, idx):
        img_fn = self.img_fns[idx]
        name = img_fn.split('/')[-1].split('.')[0]
        img = Image.open(img_fn).convert('RGB')

        pt_fn = img_fn.replace('.jpg', '.npy')
        pt = np.load(pt_fn)

        dmap_fn = img_fn.replace('.jpg', '_dmap.npy')
        dmap = np.load(dmap_fn)
        dmap = dmap[np.newaxis, :, :]
        dmap = torch.from_numpy(dmap).float()

        sample = {'name': name, 'img': img, 'pt': pt, 'dmap': dmap, 'img_size': img.size}
        sample = self.transforms(sample)
        sample['count'] = torch.tensor(len(sample['pt']))

        return sample
    
    @staticmethod
    def collate_fn(batch):
        batch_data = {}
        for key in ['img', 'dmap', 'count']:
            batch_data[key] = torch.stack([sample[key] for sample in batch], dim=0)
        for key in ['name', 'pt']:
            batch_data[key] = [sample[key] for sample in batch]
        return batch_data
    
    @staticmethod
    def get_train_transforms(hparams):
        img_transforms = T.Compose([
            T.RandomApply([T.Grayscale(num_output_channels=3)], p=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        transforms = Compose(
            [
                PadToFitCropSize(hparams.crop_size),
                RandomCrop(hparams.crop_size),
                DownsampleDensityMap(hparams.downsample_factor),
                RandomApply(HorizontalFlip(), p=0.5),
                ToTensor(),
                ImageTransformWrapper(img_transforms)
            ],
            img_keys = ['img'], 
            pt_keys = ['pt'],
            map_keys = ['dmap']
        )
        return transforms
    
    @staticmethod
    def get_val_transforms(hparams):
        img_transforms = T.Compose([
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        transforms = Compose(
            [
                PadToFitUnitSize(hparams.unit_size),
                ToTensor(),
                ImageTransformWrapper(img_transforms)
            ],
            img_keys = ['img'], 
            pt_keys = ['pt'],
            map_keys = ['dmap']
        )
        return transforms