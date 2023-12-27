import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, in_vocab_size, out_vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()

        self.embedding = nn.Embedding(in_vocab_size, embedding_dim)

        self.rnn = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )

        # output projection
        self.linear = nn.Linear(hidden_dim * 2, out_vocab_size)

    def forward(self, x):
        x = self.embedding(x)

        x, _ = self.rnn(x)
        return self.linear(x)
