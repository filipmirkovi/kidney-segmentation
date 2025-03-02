import os
import zipfile
from pathlib import Path
from io import BytesIO
import time


def extract_zipfile(
    uploaded_zip: BytesIO, extraction_directory: str | Path
) -> list[str]:
    temp_zip_path = os.path.join("temp", uploaded_zip.name)
    os.makedirs(os.path.dirname(temp_zip_path), exist_ok=True)
    with open(temp_zip_path, "wb") as f:
        f.write(uploaded_zip.getbuffer())
    with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
        # Get list of files before extraction
        file_list = zip_ref.namelist()

        # Extract all files to the extraction directory
        zip_ref.extractall(extraction_directory)
    os.remove(temp_zip_path)
    return file_list


def streamify_string(
    string: str,
    pause_time: float = 0.02,
):
    for s in string:
        yield s
        time.sleep(pause_time)
