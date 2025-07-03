import torch
import torch.nn as nn


class ResidualConv(nn.Module):
    def __init__(
        self, channels: int, kernel_size: int, activation: nn.Module = nn.LeakyReLU()
    ):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.activation = activation
        self.conv = nn.Conv2d(
            kernel_size=kernel_size,
            in_channels=channels,
            out_channels=channels,
            padding="same",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x + skip


class UpLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, pool: bool):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualConv(channels=in_channels, kernel_size=kernel_size),
            ResidualConv(channels=in_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
                padding="same",
            ),
        )

        self.upsample = (
            nn.Upsample(scale_factor=2, mode="bilinear") if pool else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        # The x and skip_connection tensors are either
        # CxHxW or BxCxHxW shaped, which means that the dim=-3
        # corresponds to the channel dim regardless of a batched or
        # unbatched input.

        x = self.layers(x)
        return self.upsample(x)


class DownLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, pool):
        super().__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SELU(),
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
                padding="same",
            ),
            ResidualConv(channels=out_channels, kernel_size=kernel_size),
            ResidualConv(channels=out_channels, kernel_size=kernel_size),
        )
        self.pool = (
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) if pool else nn.Identity()
        )

    def forward(self, x):
        x = self.layers(x)
        return self.pool(x)


class StabileSoftmax(nn.Module):
    def __init__(self, dim: int = -3):
        super().__init__()
        self.softmax = nn.Softmax2d(dim=dim)

    def forward(self, x):
        x - torch.max(x, keepdim=True, dim=-3).values
        return self.softmax(x)
