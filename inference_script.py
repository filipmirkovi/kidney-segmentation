from pathlib import Path

import mlflow
import torch.nn as nn

from src.dataset.SegmentationDataset import SegemetationDataset


def run_inference(dataset_path: str | Path, model_run_name: str):
    """
    Inference script steps:
        1. Initialize dataset for prediction
        2. Load model
        3. Run inference
        4. Report predictions
    """
    mlflow.set_tracking_uri(uri="http://127.0.0.1:44555")
    # TODO: Build dataset
    # ....

    # TODO: Load Model
    model_uri = ""  # f"runs:/{}"
    model: nn.Module = mlflow.pytorch.load_model(model_uri)


if __name__ == "__main__":
    run_inference()
