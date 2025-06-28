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
    ):
        super().__init__()

        self.latent_size = latent_size
        self.num_perceptions = num_perceptions
        self.perception_array = nn.Parameter(
            torch.randn([1, self.num_perceptions, self.latent_size]), requires_grad=True
        )
        self.cross_attention1 = CrossAttentionLayer(
            query_sequence_size=self.latent_size,
            key_value_sequence_size=input_size,
            hidden_size=attention_hidden_size,
            skip_connection=True,
        )
        self.self_attention1 = SelfAttentionLayer(
            input_size=latent_size,
            hidden_size=attention_hidden_size,
            skip_connection=True,
        )
        self.cross_attention2 = CrossAttentionLayer(
            query_sequence_size=self.latent_size,
            key_value_sequence_size=input_size,
            hidden_size=attention_hidden_size,
            skip_connection=True,
        )
        self.self_attention2 = SelfAttentionLayer(
            input_size=latent_size,
            hidden_size=attention_hidden_size,
            skip_connection=True,
        )
        self.inverse_cross_attention = CrossAttentionLayer(
            query_sequence_size=input_size,
            key_value_sequence_size=self.latent_size,
            hidden_size=attention_hidden_size,
            skip_connection=True,
        )
        self.projection1 = ProjectionFCNN(self.latent_size, self.latent_size)
        self.projection2 = ProjectionFCNN(self.latent_size, self.latent_size)
        self.projection3 = ProjectionFCNN(self.latent_size, self.latent_size)
        self.projection4 = ProjectionFCNN(self.latent_size, self.latent_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.cross_attention1(self.perception_array, input)
        output = self.projection1(output) + output
        output = self.self_attention1(output)
        output = self.projection2(output) + output
        output = self.cross_attention2(output, input)
        output = self.projection3(output) + output
        output = self.self_attention2(output)
        output = self.projection4(output) + output
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
