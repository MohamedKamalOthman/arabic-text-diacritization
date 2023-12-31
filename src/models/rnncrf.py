import torch
import torch.nn as nn

from models.torchcrf import CRF


class RNNCRFModel(nn.Module):
    def __init__(
        self, in_vocab_size, out_vocab_size, embedding_dim, hidden_dim, num_layers
    ):
        super(RNNCRFModel, self).__init__()

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
        # crf
        self.crf = CRF(out_vocab_size, batch_first=True)

    def forward(self, x, tags, seq_lengths=None, mask=None):
        x = self.embedding(x)
        if seq_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)

        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)

        if seq_lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = self.linear(x)
        x = self.crf(x, tags, mask)
        # return loss
        return -x

    def decode(self, x, seq_lengths=None):
        x = self.embedding(x)
        if seq_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)

        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)

        if seq_lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = self.linear(x)
        x = self.crf.decode(x)
        return x
