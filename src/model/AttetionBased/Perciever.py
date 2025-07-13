from loguru import logger
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from src.model.AttetionBased.Decoder import Decoder
from src.model.AttetionBased.Encoder import Encoder
from src.model.AttetionBased.PercieverProcessor import PercieverProcessor
from src.model.utils import num_params


class Perciever(nn.Module):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        num_perceptions: int,
        attenton_hidden_size: int,
        num_scaling_layers: int = 2,
        num_perciever_steps: int = 8,
    ):
        super().__init__()
        self.encoder = Encoder(
            img_size=img_size,
            in_channels=in_channels,
            num_scaling_layers=num_scaling_layers,
            hidden_size=hidden_size,
        )
        logger.info(f"Encoder params: {num_params(self.encoder)}")
        self.processor = PercieverProcessor(
            input_size=hidden_size,
            latent_size=hidden_size,
            num_perceptions=num_perceptions,
            attention_hidden_size=attenton_hidden_size,
            num_steps=num_perciever_steps,
        )
        logger.info(f"Processor params: {num_params(self.processor)}")

        self.inv_rearange = Rearrange(
            "b (h w) c -> b c h w",
            h=self.encoder.output_size[0],
            w=self.encoder.output_size[1],
        )
        self.decoder = Decoder(
            img_size=img_size,
            input_dims=self.encoder.output_size[0],
            out_channels=out_channels,
            in_channels=hidden_size,
            num_scaling_layers=num_scaling_layers,
        )
        logger.info(f"Decoder params: {num_params(self.decoder)}")

        self.softmax = StabileSoftmax()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(image)
        embedding = self.processor(embedding)
        embedding = self.inv_rearange(embedding)
        mask = self.decoder(embedding)
        # mask = self.softmax(mask)
        return mask


class StabileSoftmax(nn.Module):
    def __init__(self, dim: int = -3):
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x):
        x - torch.max(x, keepdim=True, dim=-3).values
        return self.softmax(x)


if __name__ == "__main__":

    model = Perciever(
        img_size=256,
        in_channels=3,
        out_channels=4,
        hidden_size=128,
        num_perceptions=256,
        attenton_hidden_size=32,
        num_scaling_layers=2,
    )
    print(model)
    print(num_params(model))

    x = torch.randn(2, 3, 256, 256)
    print(model(x).shape)
