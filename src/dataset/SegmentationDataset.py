from typing import Optional
import os
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import yaml
from cv2 import fillPoly
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from src.dataset.utils import ImageSplitter


@dataclass
class SegDataItem:
    id: str
    image_path: str | Path
    # `annotations` is a dictionary the type of
    # segmentation area as keys (e.g. 'blood_vessel') and a list of np.ndarray as
    # values. Those values represent polygon points, and the poylgons are the
    # boundaries of segmetnation maps.
    annotations: Optional[dict[str, list[np.ndarray]]] = False


def normalize_image(img: torch.Tensor) -> torch.Tensor:
    img.div_(255.0)
    img = (img - 0.5) / 0.5
    return img


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_path: str | Path,
        labels_path: Optional[str | Path] = None,
        labels_yaml: str | Path = "../../configs/label_ids.yaml",
    ):
        """
        images_path: path to where the images are contained.
        labels_path:Optional: path to jsonl file where labels are contained.
        """
        self.images_path = Path(images_path)
        self.labels_path = labels_path

        with open(labels_yaml, "r") as label_file:
            self.label_to_id = yaml.safe_load(label_file)

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.set_up()

    @property
    def image_size(self):
        return self[0][0].shape[-1]

    def set_up(self) -> None:
        labels = self.load_labels(self.labels_path)
        if labels is None:
            image_files = (
                list(self.images_path.glob("*.tif"))
                + list(self.images_path.glob("*.png"))
                + list(self.images_path.glob("*.jpg"))
                + list(self.images_path.glob("*.jpeg"))
            )
            self.data = [
                SegDataItem(id=img_path.stem, image_path=Path(img_path), annotations=[])
                for img_path in image_files
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
            return torch.zeros((1,))
        label_masks = np.zeros([len(self.label_to_id), *image_size])
        for annot, polygons in annotations.items():
            if annot not in self.label_to_id.keys():
                continue
            for polygon in polygons:
                fillPoly(label_masks[self.label_to_id[annot]], pts=[polygon], color=1)

        background = torch.ones(image_size) - sum(label_masks)
        label_masks[self.label_to_id["background"]] = background
        return torch.tensor(label_masks)

    def __getitem__(self, index) -> tuple[torch.Tensor | None]:
        data_item: SegDataItem = self.data[index]
        input_image = Image.open(data_item.image_path)
        label_masks = self.get_target_mask(input_image.size, data_item.annotations)
        if label_masks is not None:
            label_masks = label_masks.float()
        image = pil_to_tensor(input_image).float()
        image = normalize_image(image)
        return image, label_masks

    def __len__(self):
        return len(self.data)

    def get_class_weights(self) -> np.ndarray:
        """
        A class weight of class is calculated as the mean
        inverse area of the class (segmentation region), normalized
        by the image size.
        """
        assert (
            self.labels_path is not None
        ), "This instance of SegmentationDataset does not contain information about class masks!"
        class_weights = np.zeros(len(self.label_to_id))

        for i in tqdm(range(len(self)), desc="Calculating class_weights"):
            img, mask = self[i]
            _, H, W = mask.shape

            class_area = mask.sum(dim=(-1, -2))
            class_weights += np.where(
                class_area.numpy() > 0, H * W / (class_area.numpy()), 0
            )

        class_weights /= len(self)

        return class_weights


class ImageSplittingDatasetWrapper(Dataset):
    def __init__(
        self,
        core_dataset: Dataset,
        patch_size=128,
        image_channels=3,
        num_mask_regions=4,
        background_idx: int | None = 3,
        topk: int | None = 4,
    ):
        self.core_dataset = core_dataset
        self.image_splitter = ImageSplitter(
            patch_size=patch_size, num_channels=image_channels
        )
        self.mask_splitter = ImageSplitter(
            patch_size=patch_size, num_channels=num_mask_regions
        )
        self.mask_idx = [i for i in range(num_mask_regions) if i != background_idx]
        self.topk = topk
        self.patch_size = patch_size

    def __getitem__(self, index):

        image, mask = self.core_dataset[index]
        image_batch = self.image_splitter(image[None])
        mask_batch = self.mask_splitter(mask[None])
        if self.topk is None:
            return image_batch, mask_batch

        area_prob = mask_batch[:, self.mask_idx, ...].sum(dim=(-1, -2, -3)).add(1.0)
        area_prob = area_prob / area_prob.sum()
        idx = np.random.choice(
            range(mask_batch.shape[0]),
            replace=False,
            size=self.topk,
            p=area_prob.numpy(),
        )
        # mask_area_topk = torch.topk(
        #    mask_batch[:, self.mask_idx, ...].sum(dim=(-1, -2, -3)),
        #    k=self.topk if self.topk else image.shape[0],
        # )

        return (
            image_batch[idx, ...],  # mask_area_topk.indices, ...],
            mask_batch[idx, ...],  # mask_area_topk.indices, ...],
        )

    def __len__(self):
        return len(self.core_dataset)
