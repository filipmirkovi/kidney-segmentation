import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange


class Decoder(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.projection = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, self.patch_size * self.patch_size * out_channels),
        )
        self.rearrange = Rearrange("b n CL -> b CL n")
        self.fold = nn.Fold(
            output_size=(img_size, img_size),
            kernel_size=patch_size,
            stride=self.patch_size // 2,
        )

        # self.rearrange = Rearrange(
        #    "b (nh nw) (pw ph c) -> b c (nh ph) (nw pw)",
        #    ph=self.patch_size,
        #    pw=self.patch_size,
        #    c=self.out_channels,
        #    nh=self.img_size // self.patch_size,
        #    nw=self.img_size // self.patch_size,
        # )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        out = self.projection(embedding)
        return self.fold(self.rearrange(out))


if __name__ == "__main__":
    from src.model.utils import num_params

    x = torch.randn(4, 128, 128, 6)
    ps = 16
    num_tokens = (128 // 16) ** 2
    x_in = torch.randn(4, num_tokens, 6)
    dec = Decoder(img_size=128, patch_size=16, in_channels=6, out_channels=3)
    print(num_params(dec))
    print(dec(x_in).shape)
