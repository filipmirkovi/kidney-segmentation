from pathlib import Path
import torch
import mlflow
from loguru import logger

from torch.utils.data import DataLoader, random_split

from src.dataset.SegmentationDataset import SegemetationDataset
from src.Trainer import Trainer
from src.model.UNet import UNet


data_path = "./data/"

batch_size = 2
lr = 1e-4
num_epochs = 2
random_seed = 100
experiment = "Example-Run"
device = "cuda:0"
save_dir = "./app_data/model/"
mlflow_tracking_uri = "http://10.100.111.210:44555"  # "http://127.0.0.1:44555"


mlflow.set_tracking_uri(uri=mlflow_tracking_uri)

# Configure loguru at the start of the script
logger.add(
    "training.log",  # Log file path
    rotation="100 MB",  # Rotate file when it reaches 100MB
    level="INFO",  # Log level
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",  # Log format
)
import sys

logger.add(
    sys.stdout, format="{time} {level} {message}", filter="my_module", level="INFO"
)


def main():
    mlflow.set_experiment(experiment)
    # torch.random.seed(random_seed)
    logger.info("Setting up datasets...")
    train_set, validation_set = random_split(
        SegemetationDataset(
            images_path=Path(data_path, "train"),
            labels_path=Path(data_path, "polygons.jsonl"),
        ),
        lengths=[0.8, 0.2],
    )
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    logger.info("Datasets created!...")

    logger.info("Initializing model...")
    model: torch.nn.Module = UNet(
        in_channels=3, num_classes=3, hidden_channels=[32, 64, 128, 256]
    )
    logger.info(f"Model built!\nSending model to device {device}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    logger.info("Initializing trainer...")
    trainer = Trainer(num_epochs=num_epochs, loss_fn=loss, device=device)
    logger.info("Trainer initialzed!")

    with mlflow.start_run() as run:
        logger.info(f"Starting run {run.info.run_name}\nBeginning training...")
        trainer.train(
            model,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            optimizer=optimizer,
        )
    logger.info("Training done!")


if __name__ == "__main__":
    main()
