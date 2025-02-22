import os
from pathlib import Path
import streamlit as st
import mlflow
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.dataset.SegmentationDataset import SegemetationDataset
from src.utils.visualization import make_images_with_masks

st.markdown(
    """
## Welcome!
### This is a simple inference page.
"""
)


save_dir = "app_data/output/"
model: torch.nn.Module = mlflow.pytorch.load_model(
    model_uri="app_data/model/final_model"
)
dataset = SegemetationDataset(images_path="app_data/data/")
loader = DataLoader(dataset=dataset, batch_size=4, num_workers=4)

progress_bar = st.progress(0)
n_batches = len(loader)
model.eval()
st.write("Runing model...")
with torch.no_grad():
    for i, batch in enumerate(loader):
        image, _ = batch
        segment_masks = model(image)
        make_images_with_masks(
            image=image,
            masks=segment_masks,
            save_path=Path(save_dir, f"_batch_{i+1}.png"),
        )
        progress_bar.progress(i + 1, f"Processing batch: {i+1}/{n_batches}")


# output_file_options= os.listdir()
# st.selectbox(label="Select the model output you want to show :smile:",options=)
