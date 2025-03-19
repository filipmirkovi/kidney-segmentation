from pathlib import Path
import streamlit as st
import mlflow
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.app_utils import extract_zipfile, streamify_string
from src.dataset.SegmentationDataset import SegmentationDataset
from src.utils.visualization import make_images_with_masks


data_dir = "app_data/data/"
save_dir = "app_data/output/"
model_dir = "app_data/model/"
config_dir = "configs"
device = "cuda:0" if torch.cuda.is_available() else "cpu"


st.set_page_config(
    page_title="Kidney Segmentation",  # This sets the browser tab title
    page_icon="💡",  # This sets the icon (can be an emoji or path to an image)
)

st.image("app_data/resources/logo.png")


if "welcome_streamed" not in st.session_state:
    st.session_state.welcome_streamed = False
    # This variable will store the complete welcome message after streaming
    st.session_state.welcome_complete = ""

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

    st.session_state.welcome_complete = welcome_text
    st.session_state.welcome_streamed = True
else:
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

    dataset = SegmentationDataset(
        images_path=data_dir, labels_yaml=Path(config_dir, "label_ids.yaml")
    )
    loader = DataLoader(dataset=dataset, batch_size=4, num_workers=4)

    if st.button(label="## Start infernce"):

        progress_bar = st.progress(0)
        n_batches = len(loader)

        model: torch.nn.Module = mlflow.pytorch.load_model(
            model_uri=Path(model_dir, "final_model")
        ).to(device)
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                image, _ = batch
                segment_masks = model(image.to(device))
                figure: plt.Figure = make_images_with_masks(
                    image=image,
                    masks=segment_masks,
                    num_classes=3,
                    colors=["orange", "blue", "green"],
                )
                figure.savefig(f"{save_dir}/results_{i}.png")
        progress_bar.progress(i + 1, f"Processing batch: {i+1}/{n_batches}")

        st.write_stream(streamify_string(f"Results saved to `{save_dir}`."))
