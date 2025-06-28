import torch.nn as nn
import torch
from src.model.Perciever.Layers.Attention import (
    CrossAttentionLayer,
    SelfAttentionLayer,
)
from src.model.Perciever.Layers.FCNN import ProjectionFCNN


class PercieverProcessor(nn.Module):

    def __init__(
        self,
        latent_size: int = 128,
        input_size: int = 64,
        num_perceptions: int = 256,
        attention_hidden_size: int = 32,
        num_steps: int = 8,
    ):
        super().__init__()

        self.latent_size = latent_size
        self.num_perceptions = num_perceptions
        self.perception_array = nn.Parameter(
            torch.randn([1, self.num_perceptions, self.latent_size]), requires_grad=True
        )

        self.cross_attn = nn.ModuleList()
        self.self_attn = nn.ModuleList()
        self.cross_projectios = nn.ModuleList()
        self.projectios = nn.ModuleList()

        for step in num_steps:
            self.cross_attn.append(
                CrossAttentionLayer(
                    query_sequence_size=self.latent_size,
                    key_value_sequence_size=input_size,
                    hidden_size=attention_hidden_size,
                    skip_connection=True,
                )
            )
            self.self_attn.append(
                SelfAttentionLayer(
                    input_size=latent_size,
                    hidden_size=attention_hidden_size,
                    skip_connection=True,
                )
            )
            self.cross_projectios.append(
                ProjectionFCNN(self.latent_size, self.latent_size)
            )
            self.projectios.append(ProjectionFCNN(self.latent_size, self.latent_size))

        self.inverse_cross_attention = CrossAttentionLayer(
            query_sequence_size=input_size,
            key_value_sequence_size=self.latent_size,
            hidden_size=attention_hidden_size,
            skip_connection=True,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.perception_array
        for cross_attn, self_attn, cross_proj, proj in zip(
            self.cross_attn, self.self_attn, self.cross_projectios, self.projectios
        ):
            output = cross_attn(output, input)
            output = cross_proj(output) + output
            output = self_attn(output)
            output = proj(output) + output

        # Inverse Cross Attention uses the input array as a query and
        # the perception array as key-value pairs.
        output = self.inverse_cross_attention(input, output)
        return output


if __name__ == "__main__":
    from src.model.utils import num_params

    model = PercieverProcessor()
    print(num_params(model))
    x = torch.randn(2, 128, 64)
    print(model(x).shape)
