import torch
import torch.nn as nn
from src.model.Perciever.Decoder import Decoder
from src.model.Perciever.Encoder import Encoder
from src.model.Perciever.PercieverProcessor import PercieverProcessor


class Perciever(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        hidden_size: int,
        num_perceptions: int,
        attenton_hidden_size: int,
    ):
        super().__init__()
        self.encoder = Encoder(
            img_size=img_size,
            in_channels=in_channels,
            patch_size=patch_size,
            hidden_size=hidden_size,
        )
        self.processor = PercieverProcessor(
            input_size=hidden_size,
            latent_size=hidden_size,
            num_perceptions=num_perceptions,
            attention_hidden_size=attenton_hidden_size,
        )
        self.decoder = Decoder(
            img_size=img_size,
            patch_size=patch_size,
            out_channels=in_channels,
            in_channels=hidden_size,
        )
        self.softmax = StabileSoftmax()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(image)
        embedding = self.processor(embedding)
        mask = self.decoder(embedding)
        return mask


class StabileSoftmax(nn.Module):
    def __init__(self, dim: int = -3):
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x):
        x - torch.max(x, keepdim=True, dim=-3).values
        return self.softmax(x)
