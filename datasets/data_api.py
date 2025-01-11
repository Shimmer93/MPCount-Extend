import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import DataLoader
import os

from datasets.base_dataset import BaseDataset
from datasets.den_dataset import DensityMapDataset
from datasets.mpcount_dataset import MPCountDataset
from datasets.mpcount_extend_dataset import MPCountExtendDataset

def create_dataset(hparams, split=None):
    assert split in ['train', 'val', 'test']
    if split == 'train':
        data_dir = hparams.train_data_dir if hasattr(hparams, 'train_data_dir') else hparams.data_dir
    elif split == 'val':
        data_dir = hparams.val_data_dir if hasattr(hparams, 'val_data_dir') else hparams.data_dir
    elif split == 'test':
        data_dir = hparams.test_data_dir if hasattr(hparams, 'test_data_dir') else hparams.data_dir
    
    if hparams.dataset_name == 'base':
        dataset_class = BaseDataset
    elif hparams.dataset_name == 'den':
        dataset_class = DensityMapDataset
    elif hparams.dataset_name == 'mpcount':
        dataset_class = MPCountDataset
    elif hparams.dataset_name == 'mpcount_extend':
        dataset_class = MPCountExtendDataset
    else:
        raise ValueError(f'Unknown dataset name: {hparams.dataset_name}')

    transforms = dataset_class.get_train_transforms(hparams) if split == 'train' else dataset_class.get_val_transforms(hparams)
    dataset = dataset_class(data_dir, split, transforms)
    collate_fn = dataset_class.collate_fn
    
    return dataset, collate_fn

class LitDataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.train_dataset, self.train_collate_fn = create_dataset(self.hparams, self.hparams.train_split)
            self.val_dataset, self.val_collate_fn = create_dataset(self.hparams, self.hparams.val_split)
        elif stage == 'test':
            self.test_dataset, self.test_collate_fn = create_dataset(self.hparams, self.hparams.test_split)
        else:
            raise ValueError(f'Unknown stage: {stage}')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.train_collate_fn,
            pin_memory=self.hparams.pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.val_collate_fn,
            pin_memory=self.hparams.pin_memory,
            drop_last=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.test_collate_fn,
            pin_memory=self.hparams.pin_memory,
            drop_last=False
        )