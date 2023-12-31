import torch
import torch.nn as nn

from modules.cbhg import CBHGModule
from modules.prenet import PrenetModule


class CBHGModel(nn.Module):
    def __init__(
        self,
        in_vocab_size: int,
        out_vocab_size: int,
        embedding_dim: int,
        use_prenet: bool = True,
        prenet_dims: list[int] = [256, 128],
        prenet_dropout: float = 0.5,
        cbhg_num_filters: int = 16,
        cbhg_proj_dims: list[int] = [128, 256],
        cbhg_num_highway_layers: int = 4,
        cbhg_gru_hidden_size: int = 256,
        lstm_hidden_dims: list[int] = [256, 256],
        use_batch_norm_post_lstm: bool = True,
    ):
        super(CBHGModel, self).__init__()
        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size
        self.embedding_dim = embedding_dim
        self.use_prenet = use_prenet
        self.lstm_hidden_dims = lstm_hidden_dims
        self.use_batch_norm_post_lstm = use_batch_norm_post_lstm

        # embedding
        self.embedding = nn.Embedding(in_vocab_size, embedding_dim)

        # prenet
        if self.use_prenet:
            self.prenet = PrenetModule(
                in_dim=embedding_dim,
                hidden_dim=prenet_dims,
                dropout=prenet_dropout,
            )

        # CBHG
        self.cbhg = CBHGModule(
            in_dim=embedding_dim if not self.use_prenet else prenet_dims[-1],
            out_dim=cbhg_gru_hidden_size,
            K=cbhg_num_filters,
            proj_dims=cbhg_proj_dims,
            num_highway_layers=cbhg_num_highway_layers,
        )

        # post CBHG LSTM
        self.LSTM = nn.LSTM(
            input_size=cbhg_gru_hidden_size * 2,
            hidden_size=lstm_hidden_dims[0],
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.LSTM2 = nn.LSTM(
            input_size=lstm_hidden_dims[0] * 2,
            hidden_size=lstm_hidden_dims[1],
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # post CBHG batch norm
        if self.use_batch_norm_post_lstm:
            self.batch_norm1 = nn.BatchNorm1d(lstm_hidden_dims[0] * 2)
            self.batch_norm2 = nn.BatchNorm1d(lstm_hidden_dims[1] * 2)

        # output projection
        self.linear = nn.Linear(lstm_hidden_dims[1] * 2, out_vocab_size)

    def forward(self, x, lengths=None):
        # x: (batch_size, seq_len)
        # lengths: (batch_size, )
        x = self.embedding(x)

        if self.use_prenet:
            x = self.prenet(x)

        x = self.cbhg(x, lengths)

        hn = torch.zeros((2, 2, 2))
        cn = torch.zeros((2, 2, 2))

        x, (hn, cn) = self.LSTM(x)
        if self.use_batch_norm_post_lstm:
            # ensure the batch norm is applied to the correct dimension
            x = self.batch_norm1(x.permute(0, 2, 1)).permute(0, 2, 1)

        x, (hn, cn) = self.LSTM2(x, (hn, cn))
        if self.use_batch_norm_post_lstm:
            # ensure the batch norm is applied to the correct dimension
            x = self.batch_norm2(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.linear(x)
        return x
