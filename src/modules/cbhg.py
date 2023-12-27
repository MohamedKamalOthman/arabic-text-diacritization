import torch
import torch.nn as nn

from modules.batch_norm_conv1d import BatchNormConv1d
from modules.highway import Highway


class CBHGModule(nn.Module):
    """
    CBHG module is a stack of convolutional banks, highway networks, and a
    bidirectional GRU.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        K: int,
        proj_dims: list[int],
        num_highway_layers: int = 4,
    ):
        super(CBHGModule, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = K

        # convolutional banks
        self.relu = nn.ReLU()
        self.conv1d_banks = nn.ModuleList(
            [
                BatchNormConv1d(
                    in_channels=in_dim,
                    out_channels=in_dim,
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
        in_channels = K * in_dim
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

        # highway networks
        self.pre_highway = nn.Linear(proj_dims[1], in_dim, bias=False)
        self.highways = nn.ModuleList(
            [Highway(in_dim=in_dim, out_dim=in_dim) for _ in range(num_highway_layers)]
        )

        # bidirectional GRU
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=out_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x_in, input_lengths=None):
        x = x_in
        # x: (batch_size, seq_len, in_channels)
        x = x.transpose(1, 2)
        # x: (batch_size, in_channels, seq_len)
        max_seq_len = x.size(2)
        # convolutional banks
        x = torch.cat(
            [conv1d(x)[:, :, :max_seq_len] for conv1d in self.conv1d_banks], dim=1
        )
        assert x.size(1) == self.in_dim * self.K
        # max pooling
        x = self.max_pool1d(x)[:, :, :max_seq_len]
        # x: (batch_size, in_channels * K, seq_len)

        # projection
        x = self.conv1d_proj1(x)
        x = self.conv1d_proj2(x)
        # x: (batch_size, proj_dims[1], seq_len)

        # prepare for highway networks
        x = x.transpose(1, 2)
        # x: (batch_size, seq_len, proj_dims[1])
        # use pre-highway projection to match dimensions
        if x.size(2) != self.in_dim:
            x = self.pre_highway(x)

        # residual connection
        x += x_in

        # highway networks
        for highway in self.highways:
            x = highway(x)

        # pack sequences if lengths are given
        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

        # x: (batch_size, seq_len, in_channels * 2)

        # bidirectional GRU
        # flatten parameters to speed up computation
        self.gru.flatten_parameters()
        x, _ = self.gru(x)

        # unpack sequences if lengths are given
        if input_lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        return x
