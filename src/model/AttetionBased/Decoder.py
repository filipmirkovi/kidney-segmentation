from typing import Optional
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from src.model.Perciever.Layers.ConvLike import UpLayer


class Decoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        input_dims: Optional[int] = None,
        in_channels: int = 128,
        out_channels: int = 3,
        num_scaling_layers: int = 3,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.img_size = img_size
        hidden_sizes = (
            [in_channels]
            + [int(in_channels // 2**i) for i in range(1, num_scaling_layers)]
            + [out_channels]
        )
        self.layers = nn.Sequential(
            *[
                UpLayer(
                    in_channels=hidden_sizes[i],
                    out_channels=hidden_sizes[i + 1],
                    kernel_size=3,
                    pool=True,
                )
                for i in range(len(hidden_sizes) - 1)
            ]
        )
        if not self.is_upsampling_correct(num_scaling_layers):
            self.layers.append(
                nn.Upsample(size=(self.img_size, self.img_size), mode="bilinear")
            )
        elif input_dims and input_dims * 2**num_scaling_layers != img_size:
            self.layers.append(
                nn.Upsample(size=(self.img_size, self.img_size), mode="bilinear")
            )

    def is_upsampling_correct(self, num_scaling_layers: int) -> bool:
        im_size = int(self.img_size // 2**num_scaling_layers) * 2**num_scaling_layers
        return im_size == self.img_size

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.layers(embedding)


if __name__ == "__main__":
    from src.model.utils import num_params

    x = torch.randn(4, 64, 128, 128)

    dec = Decoder(
        img_size=512,
        in_channels=64,
        out_channels=3,
        num_scaling_layers=2,
    )
    print(dec)
    print(num_params(dec))
    print(dec(x).shape)
