import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(
        self, in_vocab_size, out_vocab_size, embedding_dim, hidden_dim, num_layers
    ):
        super(RNNModel, self).__init__()

        self.embedding = nn.Embedding(in_vocab_size, embedding_dim)

        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
        )

        # output projection
        self.linear = nn.Linear(hidden_dim * 2, out_vocab_size)

    def forward(self, x, seq_lengths=None):
        x = self.embedding(x)

        if seq_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)

        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        if seq_lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return self.linear(x)
