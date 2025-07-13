import numpy as np
import torch
import torch.nn as nn
import einops
from src.model.Perciever.Layers.ConvLike import DownLayer


def get_sinusoid_encoding(num_tokens: int, token_channels: int) -> torch.Tensor:
    """Make Sinusoid Encoding Table

    Args:
        num_tokens (int): number of tokens
        token_channels (int): num of dimensions of a token

    Returns:
        (torch.FloatTensor) sinusoidal position encoding table
    """

    def get_position_angle_vec(i):
        return [
            i / np.power(10000, 2 * (j // 2) / token_channels)
            for j in range(token_channels)
        ]

    sinusoid_table = np.array([get_position_angle_vec(i) for i in range(num_tokens)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    # Output shape: (1, token position, token dimension)
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for Vision Transformers and similar architectures.
    """

    def __init__(self, height: int, width: int, embed_dim: int):
        """
        Initialize the positional encoding.

        Args:
            height (int): Height of the input grid
            width (int): Width of the input grid
            embed_dim (int): Embedding dimension (must be divisible by 4)
            dropout (float): Dropout probability
            inv_freq (float): inv_freq parameter for the encoding
        """
        super().__init__()
        self.height = height
        self.width = width
        self.embed_dim = embed_dim

        # Create fixed positional encodings
        pos_embed = get_sinusoid_encoding(
            num_tokens=self.height * self.width, token_channels=embed_dim
        )
        self.register_buffer("pos_embed", pos_embed)

    def forward(self, x):
        """
        Add positional encoding to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, height*width, embed_dim)

        Returns:
            torch.Tensor: Output with positional encoding added
        """
        # Add positional encoding
        x = x + self.pos_embed
        return x


class Encoder(nn.Module):
    """
    Convert an image into patches and embed them.
    """

    def __init__(
        self,
        img_size=224,
        in_channels: int = 3,
        num_scaling_layers: int = 3,
        hidden_size=128,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_scaling_layers = num_scaling_layers
        self.grid_size = img_size // 2**num_scaling_layers
        self.num_patches = self.grid_size**2
        hidden_sizes = (
            [in_channels]
            + [
                int(hidden_size // 2 ** (num_scaling_layers - i))
                for i in range(num_scaling_layers - 1)
            ]
            + [hidden_size]
        )
        self.layers = nn.Sequential(
            *[
                DownLayer(
                    in_channels=hidden_sizes[i],
                    out_channels=hidden_sizes[i + 1],
                    kernel_size=3,
                    pool=True,
                )
                for i in range(len(hidden_sizes) - 1)
            ]
        )
        self.positional_encoding = PositionalEncoding(
            height=self.grid_size, width=self.grid_size, embed_dim=hidden_size
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input images of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Patch embeddings of shape (batch_size, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        assert (
            H == W == self.img_size
        ), f"Input image size ({H}*{W}) doesn't match expected size ({self.img_size}*{self.img_size})"

        # Extract patches using convolution and flatten
        x = self.layers(x)  # (B, embed_dim, grid_size, grid_size)
        x = einops.rearrange(
            x, "b c h w -> b (h w) c"
        )  # (B, grid_size*grid_size, embed_dim,)
        x = self.positional_encoding(x)
        return x

    @property
    def output_size(self) -> tuple[int]:
        s = self.img_size // 2**self.num_scaling_layers
        return s, s


if __name__ == "__main__":
    from src.model.utils import num_params

    embd = Encoder()
    print(embd.output_size)
    print(num_params(embd))
    x = torch.randn(1, 3, 224, 224)
    print(embd(x).shape)
