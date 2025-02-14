from typing import Callable
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import mlflow
from tqdm import tqdm

from src.utils.metrics import iou_score


class Trainer:
    def __init__(
        self,
        num_epochs: int,
        loss_fn: nn.Module,
        validate_every: int = 1,
        log_every_n_steps: int = 100,
    ):
        self.num_epochs = num_epochs
        self.validate_every = validate_every
        self.log_every_n_steps: int = log_every_n_steps
        self.loss_fn: nn.Module = loss_fn
        self.label_to_id = {"blood_vessel": 0, "glomerulus": 1, "unsure": 2}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def class_iou_score(self, y_pred, y_true):
        iou = iou_score(y_pred, y_true)

    def _train_epoch(
        self, model: nn.Module, dataloader: DataLoader, optimizer: Optimizer
    ):
        model.train()
        running_loss = torch.tensor(0.0)
        num_examples = 0
        for i, batch in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            x, y = batch
            y_pred: torch.Tensor = model(x)
            loss: torch.Tensor = self.loss_fn(y, y_pred)
            running_loss += loss.mean(dim=(-1, -2, -3)).sum()
            num_examples += y.shape[0]
            loss.mean().backward()
            optimizer.step()
            if self.metric_calc:
                self.metric_calc.calc_metrics(y, y_pred)

            if self.log_every_n_steps % i == 0:
                mlflow.log_metric("train_loss", running_loss / num_examples)
                num_examples = 0
                running_loss = torch.tensor(0.0)

    def _validation_epoch(
        self, model: nn.Module, dataloader: DataLoader, optimizer: Optimizer
    ):
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                x, y = batch
                y_pred: torch.Tensor = model(x)
                loss: torch.Tensor = self.loss_fn(y, y_pred)

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
    ):

        for epoch_idx in range(1, self.num_epochs + 1):

            self._train_epoch(model, train_loader, optimizer=optimizer)
            if epoch_idx % self.validate_every == 0:
                self._validation_epoch(model, val_loader)

    def log_metrics(self):
        pass
