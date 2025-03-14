import torch
import torch.nn as nn


class UpLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, pool: bool):
        super().__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SELU(),
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=in_channels,
                padding="same",
            ),
            nn.BatchNorm2d(in_channels),
            nn.SELU(),
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=in_channels,
                padding="same",
            ),
            nn.BatchNorm2d(in_channels),
            nn.SELU(),
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

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor):
        # The x and skip_connection tensors are either
        # CxHxW or BxCxHxW shaped, which means that the dim=-3
        # corresponds to the channel dim regardless of a batched or
        # unbatched input.
        x = x + skip_connection
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
            nn.BatchNorm2d(out_channels),
            nn.SELU(),
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=out_channels,
                out_channels=out_channels,
                padding="same",
            ),
            nn.BatchNorm2d(out_channels),
            nn.SELU(),
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=out_channels,
                out_channels=out_channels,
                padding="same",
            ),
        )
        self.pool = (
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) if pool else nn.Identity()
        )

    def forward(self, x):
        intermediate = self.layers(x)
        return self.pool(intermediate), intermediate


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: list[int] = [32, 64, 128],
        kernel_size=3,
        apply_softmax: bool = True,
    ):
        super().__init__()
        channels = [in_channels] + hidden_channels
        self.downlayers = nn.ModuleList(
            [
                DownLayer(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size,
                    pool=i < len(channels) - 2,
                )
                for i in range(len(channels) - 1)
            ]
        )
        channels.reverse()
        channels = channels[:-1] + [num_classes]
        self.uplayers = nn.ModuleList(
            [
                UpLayer(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size,
                    pool=i < len(channels) - 2,
                )
                for i in range(len(channels) - 1)
            ]
        )
        self.softmax = nn.Softmax(dim=-3) if apply_softmax else nn.Identity()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        intermediate: list[torch.Tensor] = []
        for down_layer in self.downlayers:
            image, downsampled = down_layer(image)
            intermediate.append(downsampled)

        for up_layer in self.uplayers:
            skip = intermediate.pop()
            # image = torch.cat([image, skip], dim=-3)
            image = up_layer(image, skip)
        # NOTE: We use BCEWithLogitsLoss which means
        # the loss itself implements the sigmoid.
        image = image - torch.max(image, keepdim=True, dim=-3).values
        return self.softmax(image)
