from typing import Optional
from collections import defaultdict
import os
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from cv2 import fillPoly
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


@dataclass
class SegDataItem:
    id: str
    image_path: str | Path
    # `annotations` is a dictionary the type of
    # segmentation area as keys (e.g. 'blood_vessel') and a list of np.ndarray as
    # values. Those values represent polygon points, and the poylgons are the
    # boundaries of segmetnation maps.
    annotations: Optional[dict[str, list[np.ndarray]]] = False


class SegemetationDataset(Dataset):
    def __init__(
        self, images_path: str | Path, labels_path: Optional[str | Path] = None
    ):
        """
        images_path: path to where the images are contained.
        labels_path:Optional: path to jsonl file where labels are contained.
        """
        self.images_path = Path(images_path)
        self.labels_path = labels_path
        self.label_to_id = {"blood_vessel": 0, "glomerulus": 1, "unsure": 2}
        self.id_to_label = {0: "blood_vessel", 1: "glomerulus", 2: "unsure"}
        self.set_up()

    def set_up(self) -> None:
        labels = self.load_labels(self.labels_path)
        if labels is None:
            self.data = [
                SegDataItem(
                    id=img_path.stem, image_path=Path(self.images_path, img_path)
                )
                for img_path in self.images_path.glob("*.tif")
            ]
        else:
            self.data = []
            for label in labels:
                image_path = Path(self.images_path, label["id"] + ".tif")
                if not os.path.isfile(image_path):
                    continue

                annotation = {lab: [] for lab in self.label_to_id.keys()}
                for annot in label["annotations"]:
                    annotation[annot["type"]].append(np.array(annot["coordinates"][0]))
                self.data.append(
                    SegDataItem(
                        id=label["id"], image_path=image_path, annotations=annotation
                    )
                )

    def load_labels(self, labels_path):
        if labels_path is None:
            return None
        with open(labels_path, "r") as labels_file:
            labels = [json.loads(line) for line in labels_file]
        return labels

    def get_target_mask(
        self, image_size, annotations: dict[str, list[np.ndarray]]
    ) -> torch.Tensor | None:
        if not annotations:
            return None
        label_masks = np.zeros([len(self.label_to_id), *image_size])
        for annot, polygons in annotations.items():
            for polygon in polygons:
                fillPoly(label_masks[self.label_to_id[annot]], pts=[polygon], color=1)

        return torch.tensor(label_masks)

    def __getitem__(self, index) -> tuple[torch.Tensor | None]:
        data_item: SegDataItem = self.data[index]
        input_image = Image.open(data_item.image_path)
        label_masks = self.get_target_mask(input_image.size, data_item.annotations)
        if label_masks:
            label_masks = label_masks.float()
        return pil_to_tensor(input_image).float(), label_masks

    def __len__(self):
        return len(self.data)
