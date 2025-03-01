from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import mlflow
from mlflow.models import infer_signature
from tqdm import tqdm
import yaml

from src.utils.metrics import LossMonitor, IOUScore, NumSteps


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
        self.num_epochs = num_epochs
        self.device = device
        self.validate_every = validate_every
        self.log_every_n_steps: int = log_every_n_steps
        self.loss_fn: nn.Module = loss_fn
        self.loss_monitor = LossMonitor().to(self.device)

        self.iou_score = IOUScore(num_classes=3, include_background_as_class=True).to(
            self.device
        )
        self.best_iou = torch.tensor(0.0, device=self.device)
        self.train_step_counter = NumSteps()
        self.eval_step_counter = NumSteps()

        with open(labels_path, "r") as label_file:
            self.label_to_id = yaml.safe_load(label_file)

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def _train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: Optimizer,
        epoch: int,
    ):
        self.loss_monitor.reset()
        model.train()

        for i, batch in enumerate(tqdm(dataloader)):
            self.train_step_counter.update()

            optimizer.zero_grad()

            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            y_pred: torch.Tensor = model(x)
            loss: torch.Tensor = self.loss_fn(y, y_pred)
            loss = loss.mean(dim=(-1, -2))

            loss.mean().backward()
            optimizer.step()
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

    def _validation_epoch(self, model: nn.Module, dataloader: DataLoader):
        self.loss_monitor.reset()
        model.eval()
        self.iou_score.compute()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader)):
                self.eval_step_counter.update()
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                y_pred: torch.Tensor = model(x)
                loss: torch.Tensor = self.loss_fn(y, y_pred)
                loss = loss.mean(dim=(-1, -2))
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
                    self.log_best_model(model, class_iou.mean(dim=0), example_data=x)

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
            model_info = mlflow.pytorch.log_model(model, "model", signature=signature)
            self.best_iou = new_score

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
    ):

        for epoch_idx in range(1, self.num_epochs + 1):
            self._train_epoch(model, train_loader, optimizer=optimizer, epoch=epoch_idx)
            if epoch_idx % self.validate_every == 0:
                self._validation_epoch(model, val_loader)
