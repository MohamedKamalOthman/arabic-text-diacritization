import torch
import torch.nn as nn

from modules.batch_norm_conv1d import BatchNormConv1d


class CRNNModel(nn.Module):
    def __init__(
        self,
        in_vocab_size,
        out_vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        K,
        proj_dims,
    ):
        super(CRNNModel, self).__init__()

        # embedding
        self.embedding = nn.Embedding(in_vocab_size, embedding_dim)

        # convolutional banks
        self.relu = nn.ReLU()
        self.conv1d_banks = nn.ModuleList(
            [
                BatchNormConv1d(
                    in_channels=embedding_dim,
                    out_channels=embedding_dim,
                    kernel_size=k,
                    padding=k // 2,
                    stride=1,
                    activation=self.relu,
                )
                for k in range(1, K + 1)
            ]
        )
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        # projection
        in_channels = K * embedding_dim
        self.conv1d_proj1 = BatchNormConv1d(
            in_channels=in_channels,
            out_channels=proj_dims[0],
            kernel_size=3,
            padding=1,
            stride=1,
            activation=self.relu,
        )
        self.conv1d_proj2 = BatchNormConv1d(
            in_channels=proj_dims[0],
            out_channels=proj_dims[1],
            kernel_size=3,
            padding=1,
            stride=1,
            activation=None,
        )

        self.rnn = nn.LSTM(
            proj_dims[1],
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
        )

        # output projection
        self.linear = nn.Linear(hidden_dim * 2, out_vocab_size)

    def forward(self, x, seq_lengths=None):
        x = self.embedding(x)

        # convolutional banks
        x = x.transpose(1, 2)
        max_seq_len = x.size(2)
        x = torch.cat(
            [conv(x)[:, :, :max_seq_len] for conv in self.conv1d_banks], dim=1
        )
        x = self.max_pool1d(x)[:, :, :max_seq_len]
        # convolutional projection
        x = self.conv1d_proj1(x)
        x = self.conv1d_proj2(x)

        x = x.transpose(1, 2)

        if seq_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)

        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        if seq_lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return self.linear(x)
