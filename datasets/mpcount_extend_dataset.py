import torch
import numpy as np
from PIL import Image
import random
from glob import glob
import os

from datasets.den_dataset import DensityMapDataset
from datasets.transforms import *

class MPCountExtendDataset(DensityMapDataset):
    def __init__(self, data_dir, split, transforms):
        super().__init__(data_dir, split, transforms)
        self.aug_dir = os.path.join(data_dir, 'aug')
        self.num_aug = len(glob(self.img_fns[0].replace(data_dir, self.aug_dir).replace('.jpg', '_*.jpg')))

        dmap_fns = [img_fn.replace('.jpg', '_dmap.npy') for img_fn in self.img_fns]
        densities = []
        for dmap_fn in dmap_fns:
            dmap = np.load(dmap_fn)
            d_nonzeros = dmap[dmap > 0]
            densities.append(d_nonzeros)
        densities = np.concatenate(densities)
        self.log_sqrt_ds = np.log(np.sqrt(densities) + 1)
        self.bins = np.linspace(self.log_sqrt_ds.min(), self.log_sqrt_ds.max(), 10)
        self.bin_counts, self.bin_edges = np.histogram(self.log_sqrt_ds, bins=self.bins)

    def __len__(self):
        return len(self.img_fns)

    def __getitem__(self, idx):
        img_fn = self.img_fns[idx]
        name = img_fn.split('/')[-1].split('.')[0]
        img = Image.open(img_fn).convert('RGB')

        if random.random() > 0.1 and self.split == 'train':
            id = random.randint(0, self.num_aug - 1)
            aug_fn = img_fn.replace(self.data_dir, self.aug_dir).replace('.jpg', f'_aug_{id}.jpg')
            img = Image.open(aug_fn).convert('RGB')
            img_aug = Image.open(aug_fn).convert('RGB')
        else:
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

        weight = torch.ones_like(sample['dmap'])
        # for i in range(len(self.bin_counts)):
        #     mask = (self.bins[i] <= torch.log(torch.sqrt(sample['dmap'])+1)) * \
        #            (torch.log(torch.sqrt(sample['dmap'])+1) < self.bins[i+1])
        #     weight[mask] = torch.sqrt(torch.tensor(len(self.log_sqrt_ds) / (self.bin_counts[i] * len(self.bin_counts))))
        sample['weight'] = weight

        return sample
    
    @staticmethod
    def collate_fn(batch):
        batch_data = {}
        for key in ['img', 'img_aug', 'dmap', 'pcm', 'count', 'weight']:
            batch_data[key] = torch.stack([sample[key] for sample in batch], dim=0)
        for key in ['name', 'pt']:
            batch_data[key] = [sample[key] for sample in batch]
        return batch_data
    
    @staticmethod
    def get_train_transforms(hparams):
        return MPCountExtendDataset.get_train_transforms(hparams)
    
    @staticmethod
    def get_val_transforms(hparams):
        return MPCountExtendDataset.get_val_transforms(hparams)