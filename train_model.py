import PIL
import sys
from pathlib import Path
import torch
import mlflow
from argparse import ArgumentParser
from loguru import logger
import yaml
from torch.utils.data import DataLoader, random_split

from src.dataset.SegmentationDataset import (
    SegmentationDataset,
    ImageSplittingDatasetWrapper,
)
from src.Trainer import Trainer
from src.model.loss import RecallWeightedCrossEntropy
from src.model.Perciever.Perciever import Perciever
from src.dataset.utils import custom_collate_fn
from src.model.UNet.UNet import UNet
from src.model.utils import num_params


# Configure loguru at the start of the script
logger.add(
    "training.log",  # Log file path
    rotation="100 MB",  # Rotate file when it reaches 100MB
    level="INFO",  # Log level
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",  # Log format
)
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
    logger.info("Setting up datasets...")
    full_dataset = SegmentationDataset(
        images_path=Path(configs["data_path"], "train"),
        labels_path=Path(configs["data_path"], "polygons.jsonl"),
        labels_yaml="configs/label_ids.yaml",
    )

    train_set, validation_set = random_split(
        full_dataset,
        lengths=[0.8, 0.2],
    )
    patch_size = 64
    train_set = ImageSplittingDatasetWrapper(
        core_dataset=train_set,
        patch_size=patch_size,
        num_mask_regions=4,
        background_idx=3,
        image_channels=3,
    )
    validation_set = ImageSplittingDatasetWrapper(
        core_dataset=validation_set,
        patch_size=patch_size,
        num_mask_regions=4,
        background_idx=3,
        image_channels=3,
        topk=None,
    )

    # class_weights = train_set.core_dataset.dataset.get_class_weights()

    # cls_weight_report = [
    #    f"{class_name} : {class_weights[i]}"
    #    for i, class_name in enumerate(train_set.core_dataset.dataset.label_to_id)
    # ]

    # logger.success(f"Calculated class weights: " + "\n".join(cls_weight_report))

    train_dataloader = DataLoader(
        train_set,
        batch_size=configs["batch_size"],
        shuffle=True,
        collate_fn=custom_collate_fn,
    )
    val_dataloader = DataLoader(
        validation_set,
        batch_size=configs["batch_size"],
        shuffle=False,
        collate_fn=custom_collate_fn,
    )
    logger.success("Datasets created!")

    logger.info("Initializing model...")

    # model = Perciever(
    #    img_size=patch_size,
    #    in_channels=3,
    #    out_channels=4,
    #    hidden_size=512,
    #    num_perceptions=1024,
    #    attenton_hidden_size=64,
    #    num_scaling_layers=3,
    #    num_perciever_steps=6,
    # )

    model = UNet(
        in_channels=3,
        num_classes=configs["num_segmentation_regions"] + 1,
        apply_softmax=False,
        hidden_channels=[32, 128, 512],
    )

    logger.success(
        "Model built {}!\nSending model to device {}\nThe model has {} parameters.".format(
            model, configs["device"], num_params(model, "million")
        )
    )

    model = model.to(configs["device"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(configs["lr"]),
        weight_decay=float(configs["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # loss = torch.nn.CrossEntropyLoss(
    #    reduction="none",
    #    # weight=torch.tensor(class_weights).to(configs["device"]),
    # )
    loss = RecallWeightedCrossEntropy(reduction="none", num_classes=4)

    logger.info("Initializing trainer...")
    trainer = Trainer(
        num_epochs=int(configs["num_epochs"]),
        loss_fn=loss,
        device=configs["device"],
        labels_path="configs/label_ids.yaml",
        log_every_n_steps=configs["log_every_steps"],
    )
    logger.success("Trainer initialzed!")

    with mlflow.start_run() as run:
        logger.info(f"Starting run {run.info.run_name}\nBeginning training...")
        trainer.train(
            model,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    logger.success("Training done!")


parser = ArgumentParser()
parser.add_argument("--config_path", type=str, default="configs/train_configs.yaml")

if __name__ == "__main__":
    args = parser.parse_args()
    main(config_path=args.config_path)
