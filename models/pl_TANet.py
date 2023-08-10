from typing import Dict, List, Optional, Tuple, Callable
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pl_bolts.optimizers import lr_scheduler

from models.submodules import Tooth_Assembler, Tooth_Centering
from losses import ConditionalWeightingLoss
from openpoints.utils import EasyConfig
from openpoints.models.build import build_model_from_cfg


# t3d------------------------------
# debug
from vedo import Points, show
import trimesh
import time

from clearml import Task
import datetime
def get_timestamp():
    """ Get now as timestamp in format 'YYYY_MMDD_HHMM'

    :return: timestamp string.
    :rtype: str
    """

    now = datetime.datetime.now()
    year = '{:02d}'.format(now.year)
    month = '{:02d}'.format(now.month)
    day = '{:02d}'.format(now.day)
    hour = '{:02d}'.format(now.hour)
    minute = '{:02d}'.format(now.minute)
    return '{}_{}{}_{}{}'.format(year, month, day, hour, minute)
# t3d end ------------------------------


class LitModule(pl.LightningModule):
    def __init__(
        self,
        cfg,
    ):
        super(LitModule, self).__init__()

        self.cfg = cfg
        self.cfg_optimizer = self.cfg.LitModule.optimizer
        self.cfg_scheduler = self.cfg.LitModule.scheduler
        self.cfg_scheduler.epochs = self.cfg.Trainer.epochs
        self.batch_size = self.cfg.LitDataModule.dataloader.batch_size
        self.learning_rate = self.cfg.LitModule.optimizer.learning_rate

        self.tooth_centering = Tooth_Centering()
        self.model = build_model_from_cfg(self.cfg.model)
        self.tooth_assembler = Tooth_Assembler()

        self.loss_fn = ConditionalWeightingLoss(sigma=5, criterion_mode=cfg.criterion.mode)

        # t3d------------------------------
        self.use_clearml = self.cfg.Clearml.use_clearml
        self.clearml_task_id = self.cfg.Clearml.clearml_task_id
        self.project_name = self.cfg.Clearml.project_name
        run_name = str(self.cfg.Clearml.run_name) + "_" + get_timestamp()

        if self.use_clearml:
            if self.clearml_task_id == '':
                self.clearml_task = Task.init(project_name=self.project_name, task_name=run_name)
                #self.clearml_task.connect(params_dictionary)
            else:
                self.clearml_task = Task.get_task(task_id=self.clearml_task_id)
                self.clearml_task.mark_started()
        # t3d end ------------------------------

    def forward(self, X: Dict[str, torch.Tensor]) -> torch.Tensor:
        dof = self.model(X)
        return dof

    def configure_optimizers(self):
        # Setup the optimizer
        optimizer = create_optimizer_v2(self.parameters(),
                                        opt=self.cfg_optimizer.NAME,
                                        lr=self.cfg_optimizer.learning_rate,
                                        weight_decay=self.cfg_optimizer.weight_decay,
                                        )

        # Setup the schedulerwarmup_epochs: int,
        scheduler = lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer,
                                                            warmup_epochs=self.cfg_scheduler.warmup_epochs,
                                                            max_epochs=self.cfg_scheduler.epochs,
                                                            warmup_start_lr=self.cfg_scheduler.warmup_start_lr,
                                                            eta_min=self.cfg_scheduler.eta_min,
                                                            last_epoch=-1,
                                                            )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def _step(self, batch: Dict[str, torch.Tensor], step: str) -> torch.Tensor:
        center_batch = self.tooth_centering(batch, self.device)
        dofs, Xi = self(center_batch)
        assembled, pred2gt_matrices = self.tooth_assembler(batch, dofs, self.device)
        loss = self.loss_fn(assembled, batch["target_X_v"], pred2gt_matrices, batch["C"].shape[1], Xi, self.device)

        self.log(f"{step}_loss", loss)

        # t3d------------------------------
        # print sum of the values in dof
        print(f"\n{step}: Loss: ", loss.item())
        print(f"\n{step}: Pose regresor output sum: ", dofs.sum().item())
        print(f"\n{step}: GT 6dof input to target sum: ", batch["6dof"].sum().item())

        self.clearml_task.get_logger().report_scalar(title='loss', series=step + '_loss',
                                                     iteration=int(self.global_step), value=loss)
        self.clearml_task.get_logger().report_scalar(title='dofs', series=step + '_dofs_sum',
                                                     iteration=int(self.global_step), value=dofs.sum().item())
        self.clearml_task.get_logger().report_scalar(title='dofs', series=step + '_GT_dofs_sum',
                                                     iteration=int(self.global_step), value=batch["6dof"].sum().item())

        if step == "train":
            points_color = "winter"
        elif step == "val":
            points_color = "gist_heat"
        else:
            points_color = "winter"

        pointcloud_X = trimesh.PointCloud(batch["X"].detach().cpu().numpy()[0, :, :])
        pointcloud_target_X = trimesh.PointCloud(batch["target_X"].detach().cpu().numpy()[0, :, :])
        numpy_assembled = assembled.detach().cpu().clone().reshape(shape=batch["X"].shape).numpy()[0, :, :]
        pointcloud_pred_X = trimesh.PointCloud(numpy_assembled)
        points_vedo_X = Points(pointcloud_X.vertices, r=5)
        points_vedo_target_X = Points(pointcloud_target_X.vertices, r=5)
        points_vedo_pred_X = Points(pointcloud_pred_X.vertices, r=5)
        points_vedo_pred_X.cmap(points_color, pointcloud_pred_X.vertices[:,1])
        points_vedo_X.cmap(points_color, pointcloud_X.vertices[:,1])
        points_vedo_target_X.cmap("summer", pointcloud_target_X.vertices[:,1])
        scene_thing = show([points_vedo_X, points_vedo_pred_X, points_vedo_target_X], N=3, axes=1)
        # sleep for 5 seconds
        time.sleep(2)
        scene_thing.close()

        return loss
