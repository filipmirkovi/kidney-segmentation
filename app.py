import os
import zipfile
from pathlib import Path
import streamlit as st
import mlflow
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


from src.app_utils import extract_zipfile, streamify_string
from src.dataset.SegmentationDataset import SegemetationDataset
from src.utils.visualization import make_images_with_masks


data_dir = "app_data/data"
save_dir = "app_data/output/"

st.markdown(
    """
    ## Kidney tissue segmentation!
    """
)


if "welcome_streamed" not in st.session_state:
    st.session_state.welcome_streamed = False
    # This variable will store the complete welcome message after streaming
    st.session_state.welcome_complete = ""

# Create a container to hold the welcome message
welcome_container = st.empty()
if not st.session_state.welcome_streamed:
    welcome_text = """
    ## Welcome! :smile:
    #### This is a page that allows you to perform detection and segmentation of various types of tissues present in human kidneys.
    """
    # Use the container to display the streaming text
    with welcome_container:
        st.write_stream(
            streamify_string(welcome_text, pause_time=0.01),
        )

    # After streaming, store the complete message and mark as streamed
    st.session_state.welcome_complete = welcome_text
    st.session_state.welcome_streamed = True
else:
    # If already streamed, just display the complete message immediately
    welcome_container.markdown(st.session_state.welcome_complete)


uploaded_zip = st.file_uploader(
    label="""**Upload a `.zip` file containing images you want to segment.
            (Supported formats: `.tif`, `.png`, `.jpg`, `.jpeg`)**
    """,
    type=".zip",
)

if uploaded_zip is not None:
    files_extracted = extract_zipfile(
        uploaded_zip=uploaded_zip, extraction_directory=data_dir
    )
    st.write("**Extracted files**:")
    for i, file in enumerate(files_extracted):
        st.write(f"`{file}`")
        if i > 3:
            st.write_stream(streamify_string("**....**"))
            break

    st.write(f"**to `{data_dir}`**")

    dataset = SegemetationDataset(
        images_path="app_data/data/", labels_yaml="configs/label_ids.yaml"
    )
    loader = DataLoader(dataset=dataset, batch_size=4, num_workers=4)

    if st.button(label="## Start infernce"):

        progress_bar = st.progress(0)
        n_batches = len(loader)

        model: torch.nn.Module = mlflow.pytorch.load_model(
            model_uri="app_data/model/final_model"
        )
        model.eval()
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

        st.write_stream(streamify_string("Results saved to `{save_dir}`."))
# st.selectbox(label="Select the model output you want to show ",options=)
