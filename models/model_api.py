from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import matplotlib.pyplot as plt
import pickle

import os
from collections import OrderedDict
# import wandb
# import tensorboard

from models.mpcount import MPCount
from models.mpcount_extend import MPCountExtend
from utils.misc import divide_img_into_patches, denormalize

def create_model(hparams):
    if hparams.model_name.lower() == 'mpcount':
        model = MPCount(pretrained=hparams.pretrained, mem_size=hparams.mem_size, mem_dim=hparams.mem_dim, cls_thrs=hparams.cls_thrs,
                    err_thrs=hparams.err_thrs, den_drop=hparams.den_drop, cls_drop=hparams.cls_drop, deterministic=hparams.deterministic, 
                    acl_type=hparams.acl_type)
    elif hparams.model_name.lower() == 'mpcount_extend':
        model = MPCountExtend(pretrained=hparams.pretrained, mem_size=hparams.mem_size, mem_dim=hparams.mem_dim, cls_thrs=hparams.cls_thrs,
                    err_thrs=hparams.err_thrs, den_drop=hparams.den_drop, cls_drop=hparams.cls_drop, deterministic=hparams.deterministic, 
                    acl_type=hparams.acl_type)
    else:
        raise ValueError(f'Unknown model name: {hparams.model_name}')
    
    return model

def create_optimizer(hparams, mparams):
    if hparams.optim_name == 'adam':
        return optim.Adam(mparams, lr=hparams.lr, weight_decay=hparams.weight_decay)
    elif hparams.optim_name == 'adamw':
        return optim.AdamW(mparams, lr=hparams.lr, weight_decay=hparams.weight_decay)
    elif hparams.optim_name == 'sgd':
        return optim.SGD(mparams, lr=hparams.lr, momentum=hparams.momentum)
    else:
        raise NotImplementedError
    
def create_scheduler(hparams, optimizer):
    if hparams.sched_name == 'cosine':
        return LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=hparams.warmup_epochs, 
                max_epochs=hparams.epochs, warmup_start_lr=hparams.warmup_lr, eta_min=hparams.min_lr)
    elif hparams.sched_name == 'step':
        return sched.MultiStepLR(optimizer, milestones=hparams.milestones, gamma=hparams.gamma)
    elif hparams.sched_name == 'plateau':
        return sched.ReduceLROnPlateau(optimizer, patience=hparams.patience, factor=hparams.factor, 
                min_lr=hparams.min_lr)
    elif hparams.sched_name == 'onecycle':
        return sched.OneCycleLR(optimizer, max_lr=hparams.lr, epochs=hparams.epochs, steps_per_epoch=hparams.steps_per_epoch,
                pct_start=hparams.pct_start, final_div_factor=hparams.final_div_factor)
    else:
        raise NotImplementedError
    
def compute_metrics(pred, gt):
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    mae = np.mean(np.abs(pred - gt), axis=0)
    mse = np.mean(np.square(pred - gt), axis=0)
    return mae, mse

class LitModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = create_model(hparams)
        if hparams.checkpoint_path is not None:
            self.load_state_dict(torch.load(hparams.checkpoint_path, map_location=self.device)['state_dict'], strict=False)


    def _calculate_loss(self, batch):
        if self.hparams.model_name.lower() == 'mpcount':
            img = batch['img']
            img_aug = batch['img_aug']
            dmap_gt = batch['dmap'] * self.hparams.log_para
            pcm_gt = batch['pcm']
            res1, res2, l_acl = self.model.forward_train(img, img_aug)
            l_den = F.mse_loss(res1['dmap'], dmap_gt) + F.mse_loss(res2['dmap'], dmap_gt)
            l_cls = F.binary_cross_entropy_with_logits(res1['pcm'], pcm_gt)
            loss = l_den + self.hparams.w_acl * l_acl + self.hparams.w_cls * l_cls
            losses = {'loss': loss, 'den': l_den, 'cls': l_cls, 'acl': l_acl}
            count = res1['dmap'].sum(dim=[1,2,3]) / self.hparams.log_para
        elif self.hparams.model_name.lower() == 'mpcount_extend':
            img = batch['img']
            img_aug = batch['img_aug']
            dmap_gt = batch['dmap'] * self.hparams.log_para
            pcm_gt = batch['pcm']
            w = batch['weight']
            res1, res2, l_err, l_acl = self.model.forward_train(img, img_aug)
            l_den = F.mse_loss(res1['dmap'] * w, dmap_gt * w) + F.mse_loss(res2['dmap'] * w, dmap_gt * w)
            l_cls = F.binary_cross_entropy_with_logits(res1['pcm'], pcm_gt)
            loss = l_den + self.hparams.w_acl * l_acl + self.hparams.w_cls * l_cls + self.hparams.w_err * l_err
            losses = {'loss': loss, 'den': l_den, 'cls': l_cls, 'acl': l_acl, 'err': l_err}
            count = res1['dmap'].sum(dim=[1,2,3]) / self.hparams.log_para
        else:
            raise NotImplementedError
        
        return losses, count
    
    def _evaluate(self, batch):
        img = batch['img']
        h, w = img.shape[2:]
        ps = self.hparams.patch_size
        if h >= ps or w >= ps:
            count = 0
            img_patches, _, _ = divide_img_into_patches(img, ps)
            for patch in img_patches:
                dmap = self.model(patch)
                count += dmap.sum(dim=[1,2,3]) / self.hparams.log_para
        else:
            dmap = self.model(img)
            count = dmap.sum(dim=[1,2,3]) / self.hparams.log_para
        return count
    
    def _evaluate_and_generate_maps(self, batch):
        img = batch['img']
        h, w = img.shape[2:]
        ps = self.hparams.patch_size

        if self.hparams.model_name.lower() in ['mpcount_extend', 'mpcount']:
            res = {}
            for ii, key in enumerate(['img', 'img_aug']):
                img = batch[key]
                if h >= ps or w >= ps:
                    dmap = torch.zeros(1, 1, h, w)
                    dmap_raw = torch.zeros(1, 1, h//4, w//4)
                    pcm = torch.zeros(1, 1, h//16, w//16)
                    img_patches, nh, nw = divide_img_into_patches(img, ps)
                    for i in range(nh):
                        for j in range(nw):
                            patch = img_patches[i*nw+j]
                            res_i = self.model(patch, return_dict=True)
                            dmap[:, :, i*ps:(i+1)*ps, j*ps:(j+1)*ps] = res_i['dmap']
                            dmap_raw[:, :, i*ps//4:(i+1)*ps//4, j*ps//4:(j+1)*ps//4] = res_i['dmap_raw']
                            pcm[:, :, i*ps//16:(i+1)*ps//16, j*ps//16:(j+1)*ps//16] = res_i['pcm']
                    res.extend({f'dmap_{ii}': dmap, f'dmap_raw_{ii}': dmap_raw, f'pcm_{ii}': pcm})
                else:
                    res_i = self.model(img, return_dict=True)
                    res.update({f'dmap_{ii}': res_i['dmap'], f'dmap_raw_{ii}': res_i['dmap_raw'], f'pcm_{ii}': F.sigmoid(res_i['pcm'])})
            count = res['dmap_0'].sum(dim=[1,2,3]) / self.hparams.log_para
        else:
            if h >= ps or w >= ps:
                dmap = torch.zeros(1, 1, h, w)
                img_patches, nh, nw = divide_img_into_patches(img, ps)
                for i in range(nh):
                    for j in range(nw):
                        patch = img_patches[i*nw+j]
                        res = self.model(patch)
                        dmap[:, :, i*ps:(i+1)*ps, j*ps:(j+1)*ps] = res
                res = {'dmap': dmap}
            else:
                res = self.model(img)
            count = res['dmap'].sum(dim=[1,2,3]) / self.hparams.log_para

        return count, res
    
    def _visualize(self, batch, res):
        if self.hparams.model_name.lower() in ['mpcount_extend', 'mpcount']:
            img = denormalize(batch['img'])[0].detach().cpu().permute(1, 2, 0).numpy()
            img_aug = denormalize(batch['img_aug'])[0].detach().cpu().permute(1, 2, 0).numpy()
            dmap_gt = batch['dmap'][0, 0].detach().cpu().numpy()
            pcm_gt = batch['pcm'][0, 0].detach().cpu().numpy().astype(np.float32)
            dmap_pred = res['dmap_0'][0, 0].detach().cpu().numpy()
            dmap_raw = res['dmap_raw_0'][0, 0].detach().cpu().numpy()
            pcm_pred = res['pcm_0'][0, 0].detach().cpu().numpy()
            dmap_pred_aug = res['dmap_1'][0, 0].detach().cpu().numpy()
            dmap_raw_aug = res['dmap_raw_1'][0, 0].detach().cpu().numpy()
            pcm_pred_aug = res['pcm_1'][0, 0].detach().cpu().numpy()
            pcm_bin = (pcm_pred > 0.5).astype(np.float32)
            pcm_bin_aug = (pcm_pred_aug > 0.5).astype(np.float32)

            count_gt = batch['count'][0].item()
            count_pred = res['dmap_0'].sum().item() / self.hparams.log_para
            count_pred_aug = res['dmap_1'].sum().item() / self.hparams.log_para

            name = batch['name'][0]

            data = [img, img_aug, dmap_gt, pcm_gt, dmap_raw, pcm_pred, dmap_pred, 
                    pcm_bin, dmap_raw_aug, pcm_pred_aug, dmap_pred_aug, pcm_bin_aug]
            labels = [f'img: {name}', 'img_aug', f'dmap_gt: {count_gt:.2f}', 'pcm_gt', 
                      'dmap_raw', 'pcm_pred', f'dmap_pred: {count_pred:.2f}', 'pcm_bin', 
                      'dmap_raw_aug', 'pcm_pred_aug', f'dmap_pred_aug: {count_pred_aug:.2f}', 'pcm_bin_aug']
            
            fig, axes = plt.subplots(3, 4, figsize=(16, 9))
            for i, ax in enumerate(axes.flat):
                ax.imshow(data[i])
                ax.set_title(labels[i])
                ax.axis('off')
            plt.tight_layout()

            tensorboard = self.logger.experiment
            tensorboard.add_figure('vis', fig, global_step=self.global_step)
            plt.close(fig)
            plt.clf()
        else:
            raise NotImplementedError

    def training_step(self, batch, batch_idx):
        count_gt = batch['count']
        losses, count = self._calculate_loss(batch)

        mae, mse = compute_metrics(count, count_gt)
        log_dict = {'train_mae': mae, 'train_mse': mse}

        for loss_name, loss in losses.items():
            log_dict[f'train_{loss_name}'] = loss

        self.log_dict(log_dict, sync_dist=True, on_step=True, on_epoch=True)

        return losses['loss']
    
    def validation_step(self, batch, batch_idx):
        count_gt = batch['count']

        if batch_idx == 0:
            count, res = self._evaluate_and_generate_maps(batch)
            self._visualize(batch, res)
        else:
            count = self._evaluate(batch)

        mae, mse = compute_metrics(count, count_gt)
        self.log_dict({'val_mae': mae, 'val_mse': mse}, sync_dist=True, on_epoch=True)

    def on_validation_epoch_end(self):
        val_mse = torch.sqrt(self.trainer.callback_metrics['val_mse'])
        self.log('val_rmse', val_mse, sync_dist=True)
        return super().on_validation_epoch_end()
    
    def test_step(self, batch, batch_idx):
        count_gt = batch['count']

        if batch_idx == 0:
            count, res = self._evaluate_and_generate_maps(batch)
            self._visualize(batch, res)
        else:
            count = self._evaluate(batch)

        mae, mse = compute_metrics(count, count_gt)
        self.log_dict({'test_mae': mae, 'test_mse': mse}, sync_dist=True, on_epoch=True)

    def on_test_epoch_end(self):
        test_mse = torch.sqrt(self.trainer.callback_metrics['test_mse'])
        self.log('test_rmse', test_mse, sync_dist=True)
        return super().on_test_epoch_end()

    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams, self.model.parameters())
        scheduler = create_scheduler(self.hparams, optimizer)
        return [optimizer], [scheduler]