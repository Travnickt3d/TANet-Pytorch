import sys, os
sys.path.append('E:\\BSP\\TANNet\\TANet-Pytorch')
#sys.path.append(os.getcwd())

# set os environment variable
os.environ["FORCE_CUDA"] = "1"

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from models import LitModule
from data import LitDataModule
from openpoints.utils import EasyConfig

import warnings
#warnings.filterwarnings("ignore", message="No audio backend is available.")


def train(cfg_path: str):
    cfg = EasyConfig()
    cfg.load(cfg_path, recursive=True)
    
    pl.seed_everything(cfg.seed)

    datamodule = LitDataModule(cfg.LitDataModule)

    datamodule.setup()

    module = LitModule(cfg)

    cfg = cfg.Trainer
    model_checkpoint = ModelCheckpoint(cfg.checkpoint_dir,
                                        monitor="val_loss",
                                        filename=f"{module.model.__class__.__name__}_{module.model.global_encoder.__class__.__name__}",
                                        )

    trainer = pl.Trainer(accumulate_grad_batches=cfg.accumulate_grad_batches,
                        auto_lr_find=cfg.auto_lr_find,
                        auto_scale_batch_size=cfg.auto_scale_batch_size,
                        benchmark=True,
                        callbacks=[model_checkpoint],
                        deterministic=True,
                        fast_dev_run=cfg.fast_dev_run,
                        gpus=cfg.gpus,
                        max_epochs=cfg.epochs,
                        precision=cfg.precision,
                        log_every_n_steps=cfg.logger.log_every_n_steps,
                        logger=WandbLogger(project=cfg.logger.wandb.project) if cfg.logger.wandb.use == True else False,
                        )

    trainer.tune(module, datamodule=datamodule)

    trainer.fit(module, datamodule=datamodule)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("-cfg", "--config_file", type=str, metavar="", help="configuration file", default="config/pointnet++.yaml")

    args = parser.parse_args()

    # import os
    # import sys
    # sys.stderr = open(os.devnull, "w")  # silence stderr

    train(args.config_file)
