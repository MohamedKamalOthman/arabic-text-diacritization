import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(
        self, in_vocab_size, out_vocab_size, embedding_dim, hidden_dim, num_layers
    ):
        super(RNNModel, self).__init__()

        self.embedding = nn.Embedding(in_vocab_size, embedding_dim)
        # add batch norm after each lstm layer
        self.rnn_layers = nn.ModuleList(
            [
                nn.LSTM(
                    embedding_dim if i == 0 else hidden_dim * 2,
                    hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
                for i in range(num_layers)
            ]
        )
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim * 2) for _ in range(num_layers)]
        )

        # output projection
        self.linear = nn.Linear(hidden_dim * 2, out_vocab_size)

    def forward(self, x, seq_lengths=None):
        x = self.embedding(x)

        # flatten parameters
        for rnn_layer in self.rnn_layers:
            rnn_layer.flatten_parameters()
        # run through rnn layers
        for i, rnn_layer in enumerate(self.rnn_layers):
            if seq_lengths is not None:
                x = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
            x, _ = rnn_layer(x)
            if seq_lengths is not None:
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            x = self.batch_norms[i](x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.linear(x)
