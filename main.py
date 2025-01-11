import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from lightning_fabric.utilities.seed import seed_everything

from datasets.data_api import LitDataModule
from models.model_api import LitModel
from utils.misc import load_cfg, merge_args_cfg
from utils.callbacks import OverrideEpochStepCallback

def main(args):
    torch.set_float32_matmul_precision('medium')
    seed_everything(args.seed)

    dm = LitDataModule(hparams=args)
    model = LitModel(hparams=args)

    callbacks = [
        ModelCheckpoint(
            monitor='val_mae',
            dirpath=os.path.join('logs', args.exp_name, args.version),
            filename=args.model_name+'-{epoch}-{val_mae:.2f}',
            save_top_k=1,
            save_last=True,
            mode='min'),
        RichProgressBar(refresh_rate=5),
        OverrideEpochStepCallback()
    ]

    logger = TensorBoardLogger(save_dir='logs', 
                               name=args.exp_name,
                               version=args.version)
    logger.log_hyperparams(args)

    trainer = pl.Trainer(
        fast_dev_run=args.dev,
        logger=logger,
        max_epochs=args.epochs,
        devices=args.gpus,
        accelerator="gpu",
        sync_batchnorm=args.sync_batchnorm,
        num_nodes=args.num_nodes,
        gradient_clip_val=args.clip_grad,
        strategy=DDPStrategy(find_unused_parameters=False) if args.strategy == 'ddp' else args.strategy,
        callbacks=callbacks,
        precision=args.precision,
        benchmark=args.benchmark
    )

    if bool(args.test):
        trainer.test(model, datamodule=dm, ckpt_path=args.checkpoint_path)
    else:
        trainer.fit(model, datamodule=dm, ckpt_path=args.checkpoint_path)
        if args.dev==0:
            trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='cfg/test.yaml')
    parser.add_argument('-g', "--gpus", type=str, default=None,
                        help="Number of GPUs to train on (int) or which GPUs to train on (list or str) applied per node.")
    parser.add_argument('-d', "--dev", type=int, default=0, help='fast_dev_run for debug')
    parser.add_argument('-n', "--num_nodes", type=int, default=1)
    parser.add_argument('-w', "--num_workers", type=int, default=4)
    parser.add_argument('-b', "--batch_size", type=int, default=2048)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument("--model_ckpt_dir", type=str, default="./model_ckpt/")
    parser.add_argument("--data_dir", type=str, default="../../data/imagenet")
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--exp_name', type=str, default='fasternet')
    parser.add_argument("--version", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    cfg = load_cfg(args.cfg)
    args = merge_args_cfg(args, cfg)

    main(args)