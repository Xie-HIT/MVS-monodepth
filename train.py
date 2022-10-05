#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn
import torch.optim
from torch.optim import lr_scheduler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from datasets.Replica_TANDEM import Replica_TANDEM
from models import monoMVSNet
from utils.utils import *


class LitNetwork(pl.LightningModule):
    def __init__(self, opt):
        super(LitNetwork, self).__init__()
        self.model = monoMVSNet(opt)
        self.option = opt

        # loss
        self.loss_type = self.option.loss['norm']
        if self.loss_type == 'L1':
            self.loss = nn.SmoothL1Loss(reduction='mean')
        elif self.loss_type == 'L2':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError('loss un-recognized')

        self.weight_scale = self.option.loss['weight']['scale']
        self.weight_depth_stage1 = self.option.loss['weight']['depth_stage1']
        self.weight_depth_stage2 = self.option.loss['weight']['depth_stage2']
        self.weight_depth_stage3 = self.option.loss['weight']['depth_stage3']

        def compute_loss(output, scale_gt, depth_gt, depth_mask):
            scale = output['scale']
            depth_stage1 = output['depth_stage1']
            depth_stage2 = output['depth_stage2']
            depth_stage3 = output['depth_stage3']

            depth_gt_stage1 = depth_gt['stage1']
            depth_gt_stage2 = depth_gt['stage2']
            depth_gt_stage3 = depth_gt['stage3']

            depth_mask_stage1 = depth_mask['stage1']
            depth_mask_stage2 = depth_mask['stage2']
            depth_mask_stage3 = depth_mask['stage3']

            loss = self.weight_scale * self.loss(scale, scale_gt) \
                   + self.weight_depth_stage1 * self.loss(depth_stage1[depth_mask_stage1], depth_gt_stage1[depth_mask_stage1]) \
                   + self.weight_depth_stage2 * self.loss(depth_stage2[depth_mask_stage2], depth_gt_stage2[depth_mask_stage2]) \
                   + self.weight_depth_stage3 * self.loss(depth_stage3[depth_mask_stage3], depth_gt_stage3[depth_mask_stage3])

            return loss

        self.compute_loss = compute_loss

        # training
        self.epochs = self.option.training['Replica']['epochs']
        self.batch_size = self.option.training['Replica']['batch_size']
        self.num_workers = self.option.training['Replica']['num_workers']
        self.learning_rate = self.option.training['Replica']['learing_rate']['lr']
        self.milestones = self.option.training['Replica']['learing_rate']['milestones']
        self.gamma = self.option.training['Replica']['learing_rate']['gamma']

        # save to wandb
        config = {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'learning_rate': self.learning_rate,
            'lr_milestones': self.milestones,
            'lr_gamma': self.gamma,
            'loss/type': self.loss_type,
            'loss/weight': [self.weight_scale, self.weight_depth_stage1,
                            self.weight_depth_stage2, self.weight_depth_stage3]
        }
        self.save_hyperparameters(config)

    def forward(self, sample):
        return self.model(sample)

    def prepare_data(self):
        self.train_dataset = Replica_TANDEM(split='train', option=self.option)
        self.val_dataset = Replica_TANDEM(split='val', option=self.option)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def common_step(self, batch, batch_idx, stage):
        images = batch['images']
        intrinsics = batch['intrinsics']
        cam_to_world = batch['cam_to_world']
        depth_gt = {
            'stage1': batch['depth']['stage1'],
            'stage2': batch['depth']['stage2'],
            'stage3': batch['depth']['stage3']
        }
        depth_mask = {
            'stage1': batch['depth_mask']['stage1'],
            'stage2': batch['depth_mask']['stage2'],
            'stage3': batch['depth_mask']['stage3']
        }

        batch = depth_gt['stage1'].shape[0]
        scale_gt = depth_gt['stage1'].view(batch, -1).median(dim=1).values

        # forward
        output = self.model(images, intrinsics, cam_to_world)

        # loss
        loss = self.compute_loss(output, scale_gt, depth_gt, depth_mask)
        if torch.isnan(loss):
            print(f"Loss is nan, loss: {loss}, batch_idx={batch_idx}")
        else:
            self.log('{}_loss'.format(stage), loss)

        # log
        with torch.no_grad():
            scale = output['scale']
            depth_pred = output['depth_stage3']
            depth_gt = depth_gt['stage3']
            mask = depth_mask['stage3']

            scale_abs_err = abs_error(scale, scale_gt).mean()
            depth_abs_err = abs_error(depth_pred, depth_gt, mask).mean()

            self.log('scale_err', round(scale_abs_err.item(), 2), prog_bar=True)
            self.log('depth_err', round(depth_abs_err.item(), 2), prog_bar=True)
            self.log('train/depth_acc_1mm', round(acc_threshold(depth_pred, depth_gt, mask, 1).mean().item(), 2))
            self.log('train/depth_acc_2mm', round(acc_threshold(depth_pred, depth_gt, mask, 2).mean().item(), 2))
            self.log('train/depth_acc_4mm', round(acc_threshold(depth_pred, depth_gt, mask, 4).mean().item(), 2))

        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, stage='val')


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", UserWarning)

    # read config file
    opt = Option()
    opt.read('./config/default.yaml')

    # for reproducibility and distribute training
    pl.seed_everything(42, workers=True)

    # model
    network = LitNetwork(opt)

    # wandb instance
    wandb_logger = WandbLogger(project="monoMVSNet")
    wandb_logger.watch(network, log="all", log_freq=1)

    # training
    trainer = pl.Trainer(min_epochs=10, max_epochs=network.epochs,
                         auto_scale_batch_size=False, accumulate_grad_batches=1,
                         auto_lr_find=True,
                         track_grad_norm=-1, gradient_clip_val=10.0,
                         gpus=-1, num_nodes=1, sync_batchnorm=True,  # [strategy='ddp'] to choose distributed backend
                         logger=WandbLogger(project="monoMVSNet"),
                         fast_dev_run=False,  # debug mode will disable logger
                         deterministic=True,
                         callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
    torch.use_deterministic_algorithms(False)  # median() will use non-determinstic method
    trainer.tune(network)  # find the best learning rate
    trainer.fit(network)  # train
