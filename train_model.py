from pathlib import Path
import torch
import mlflow
from argparse import ArgumentParser
from loguru import logger
import yaml
from torch.utils.data import DataLoader, random_split

from src.dataset.SegmentationDataset import SegemetationDataset
from src.Trainer import Trainer
from src.model.UNet import UNet


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


def main(config_path: str | Path):
    logger.info("Loading configs...")
    with open(config_path, "r") as config_file:
        configs = yaml.safe_load(config_file)
    logger.info(
        "Starting experiment with configs: "
        + "\n".join([f"{cfg}:{cfg_val}" for cfg, cfg_val in configs.items()])
    )
    mlflow.set_tracking_uri(uri=configs["mlflow_tracking_uri"])

    mlflow.set_experiment(configs["experiment"])
    # torch.random.seed(random_seed)
    logger.info("Setting up datasets...")
    train_set, validation_set = random_split(
        SegemetationDataset(
            images_path=Path(configs["data_path"], "train"),
            labels_path=Path(configs["data_path"], "polygons.jsonl"),
            labels_yaml="configs/label_ids.yaml",
        ),
        lengths=[0.8, 0.2],
    )
    train_dataloader = DataLoader(
        train_set, batch_size=configs["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(
        validation_set, batch_size=configs["batch_size"], shuffle=False
    )
    logger.info("Datasets created!...")

    logger.info("Initializing model...")
    model: torch.nn.Module = UNet(
        in_channels=3,
        num_classes=int(configs["num_segmentation_regions"])
        + 1,  # The last +1 indicates the model should handle background as well.
        hidden_channels=[32, 64, 128, 256],
    )
    logger.info("Model built!\nSending model to device {}".format(configs["device"]))
    model = model.to(configs["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(configs["lr"]))
    loss = torch.nn.CrossEntropyLoss(reduction="none")
    logger.info("Initializing trainer...")
    trainer = Trainer(
        num_epochs=int(configs["num_epochs"]),
        loss_fn=loss,
        device=configs["device"],
        labels_path="configs/label_ids.yaml",
    )
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


parser = ArgumentParser()
parser.add_argument("--config_path", type=str, default="configs/train_configs.yaml")

if __name__ == "__main__":
    args = parser.parse_args()
    main(config_path=args.config_path)
