from typing import Literal
from pydantic import BaseModel
import torch.nn as nn


class ImageInfo(BaseModel):
    height: int
    width: int
    channels: int


def num_params(
    model: nn.Module,
    units: Literal["million", "billion", "thousand", "none"] = "million",
) -> str:
    num_params = sum([param.numel() for param in model.parameters()])
    match units:
        case "none":
            return f"{num_params}"
        case "thousand":
            return f"{num_params/1e3} thousand"
        case "million":
            return f"{num_params/1e6} million"
        case "billion":
            return f"{num_params/1e9} billion"
        case _:
            raise ValueError(
                f"{units} is an unkown type of unit. Please provide: 'million', 'billion', 'thousand', 'none'."
            )
