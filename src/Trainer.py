import sys
from typing import Optional
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import mlflow
from mlflow.models import infer_signature
from tqdm import tqdm
import yaml
from loguru import logger


from src.utils.metrics import LossMonitor, IOUScore, NumSteps
from src.utils.visualization import make_images_with_masks, visualize_segmentation_masks


class Trainer:
    def __init__(
        self,
        num_epochs: int,
        loss_fn: nn.Module,
        validate_every: int = 1,
        log_every_n_steps: int = 450,
        device: str = "cpu",
        labels_path: str | Path = "../configs/label_ids.yaml",
    ):
        with open(labels_path, "r") as label_file:
            self.label_to_id = yaml.safe_load(label_file)
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        self.num_epochs = num_epochs
        self.device = device
        self.validate_every = validate_every
        self.log_every_n_steps: int = log_every_n_steps
        self.loss_fn: nn.Module = loss_fn
        self.loss_monitor = LossMonitor().to(self.device)

        self.iou_score = IOUScore(
            num_classes=len(self.id_to_label.items()), include_background_as_class=False
        ).to(self.device)
        self.best_iou = torch.tensor(0.0, device=self.device)
        self.train_step_counter = NumSteps()
        self.eval_step_counter = NumSteps()
        self.logger = logger
        logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None

    def _train_epoch(
        self,
        epoch: int,
    ):

        self.loss_monitor.reset()
        self.model.train()

        for i, batch in enumerate(tqdm(self.train_dataloader)):
            self.train_step_counter.update()

            self.optimizer.zero_grad()

            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            y_pred: torch.Tensor = self.model(x)
            loss: torch.Tensor = self.loss_fn(y_pred, y)
            loss = loss.mean(
                dim=(
                    -1,
                    -2,
                )
            )
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

            self.loss_monitor.update(loss.to(self.device))
            self.iou_score.update(y_pred, y)

            if self.train_step_counter.compute() % self.log_every_n_steps == 0:
                mlflow.log_metric("epoch", epoch)
                mlflow.log_metric("train_loss", self.loss_monitor.compute())
                self.loss_monitor.reset()
                class_iou = self.iou_score.compute()
                self.iou_score.reset()
                mlflow.log_metrics(
                    {
                        "train_" + label + "_iou_score": class_iou[label_id]
                        for label, label_id in self.label_to_id.items()
                    }
                )
                self.visualization_callback(
                    self.model, self.train_dataloader, epoch=epoch, epoch_type="train"
                )

        self.scheduler.step()

    def _validation_epoch(self):
        self.loss_monitor.reset()
        self.model.eval()
        self.iou_score.compute()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_dataloader)):
                self.eval_step_counter.update()
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                y_pred: torch.Tensor = self.model(x)
                loss: torch.Tensor = self.loss_fn(y_pred, y)
                loss = loss.mean(
                    dim=(
                        -1,
                        -2,
                    )
                )
                self.iou_score.update(y_pred, y)
                self.loss_monitor.update(loss)

                if self.eval_step_counter.compute() % self.log_every_n_steps == 0:
                    mlflow.log_metric("val_loss", self.loss_monitor.compute())
                    self.loss_monitor.reset()
                    class_iou = self.iou_score.compute()
                    self.iou_score.reset()
                    mlflow.log_metrics(
                        {
                            "val_" + label + "_iou_score": class_iou[label_id]
                            for label, label_id in self.label_to_id.items()
                        }
                    )
                    self.log_best_model(
                        self.model, class_iou.mean(dim=0), example_data=x
                    )

    def log_best_model(
        self,
        model: nn.Module,
        new_score: torch.Tensor,
        example_data: torch.Tensor | None = None,
    ) -> None:
        if new_score > self.best_iou:
            signature = (
                infer_signature(
                    example_data.cpu().numpy(),
                    model(example_data).detach().cpu().numpy(),
                )
                if example_data is not None
                else None
            )
            self.model_info = mlflow.pytorch.log_model(
                model, "model", signature=signature
            )
            self.best_model = model
            self.best_model_signature = signature
            self.logger.info(f"Logged Model Info: {self.model_info}")
            self.best_iou = new_score

    def __del__(self):
        if hasattr(self, "best_model"):
            logger.info("Saving best model...")
            mlflow.pytorch.save_model(
                self.best_model.to("cpu"), "best_model", self.best_model_signature
            )

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_loader
        self.val_dataloader = val_loader

        for epoch_idx in range(1, self.num_epochs + 1):
            self._train_epoch(
                epoch=epoch_idx,
            )

            if epoch_idx % self.validate_every == 0:
                self._validation_epoch()

    def visualization_callback(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        epoch: int,
        epoch_type: str,
        num_to_visualize: int = 4,
    ) -> None:
        idx = torch.randint(
            low=0, high=len(dataloader.dataset), size=(num_to_visualize,), dtype=int
        )
        all_masks = []
        all_target_masks = []
        all_images = []
        with torch.no_grad():
            for i in idx:
                img, label = dataloader.dataset[i]
                if len(img.shape) < 4:
                    img = img[None, ...]
                if len(label.shape) < 4:
                    label = label[None, ...]
                masks = model(img.to(self.device))
                all_images.append(img)
                all_masks.append(masks.to("cpu")[:, :3, ...])
                all_target_masks.append(label)

        all_images = torch.cat(all_images, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        all_target_masks = torch.cat(all_target_masks, dim=0)
        figure = make_images_with_masks(
            all_images,
            all_masks,
            num_classes=len(self.id_to_label) - 1,
        )

        mlflow.log_figure(
            figure, f"{epoch_type}/epoch_{epoch}/model_mask_prediction.png"
        )

        figure = visualize_segmentation_masks(
            target=all_target_masks, prediction=all_masks, background_idx=3
        )

        mlflow.log_figure(
            figure,
            f"{epoch_type}/epoch_{epoch}/mask_comparison/comparison_{i+1}.png",
        )
