from typing import Callable
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tqdm import tqdm


class Trainer:
    def __init__(self, num_epochs: int, loss_fn: nn.Module, validate_every: int = 1):
        self.num_epochs = num_epochs
        self.validate_every = validate_every
        self.loss_fn: nn.Module = loss_fn

    def _train_epoch(
        self, model: nn.Module, dataloader: DataLoader, optimizer: Optimizer
    ):
        model.train()
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            x, y = batch
            ypred: torch.Tensor = model(x)
            loss: torch.Tensor = self.loss_fn(y, ypred)
            loss.backward()
            optimizer.step()

    def _validation_epoch(
        self, model: nn.Module, dataloader: DataLoader, optimizer: Optimizer
    ):
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                x, y = batch
                ypred: torch.Tensor = model(x)
                loss: torch.Tensor = self.loss_fn(y, ypred)

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
