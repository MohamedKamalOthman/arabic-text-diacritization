from math import sqrt

import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        max_seq_length=600,
        num_heads=8,
        hidden_size=256,
        num_layers=6,
    ):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.positional_encoding = self._generate_positional_encoding(
            max_seq_length, hidden_size
        )

        self.encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x, seq_lengths=None):
        x = self.embedding(x) * sqrt(self.hidden_size)
        x += self.positional_encoding[: x.size(0), :]

        x = x.permute(
            1, 0, 2
        )  # Change from [seq_len, batch, hidden] to [batch, seq_len, hidden]

        encoded = self.encoder(x)
        decoded = self.decoder(
            encoded.permute(1, 0, 2)
        )  # Change back to [batch, seq_len, hidden]

        return decoded

    def _generate_positional_encoding(self, max_seq_length, hidden_size):
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2)
            * -(torch.log(torch.tensor(10000.0)) / hidden_size)
        )
        pos_enc = torch.zeros(max_seq_length, hidden_size)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(1)
        return pos_enc.to("cuda")
