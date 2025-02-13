from pathlib import Path
import torch
import mlflow
from torch.utils.data import DataLoader, random_split

from src.dataset.SegmentationDataset import SegemetationDataset
from src.Trainer import Trainer
from src.model.UNet import UNet

data_path = "./data/"

batch_size = 4
lr = 1e-4
random_seed = 100
experiment = "Example-Run"


mlflow.set_tracking_uri(uri="http://127.0.0.1:44000")


def main():
    mlflow.set_experiment(experiment)
    torch.random.seed(random_seed)

    train_set, validation_set = random_split(
        SegemetationDataset(
            images_path=Path(data_path, "train"),
            labels_path=Path(data_path, "polygons.jsonl"),
        ),
        lengths=[0.8, 0.2],
    )
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    model: torch.nn.Module = UNet(hidden_channels=[32, 64, 128, 256])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    trainer = Trainer(loss_fn=loss)

    with mlflow.start_run() as run:

        trainer.train(
            model,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            optimizer=optimizer,
        )
