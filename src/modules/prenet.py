import torch.nn as nn


class PrenetModule(nn.Module):
    """
    A prenet module is a stack of linear layers with ReLU activations and dropout.
    """

    def __init__(self, in_dim, hidden_dim=[256, 128], dropout=0.5):
        super(PrenetModule, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: (batch_size, in_channels)
        x = self.dropout(self.relu(self.linear1(x)))  # (batch_size, hidden_channels[0])
        x = self.dropout(self.relu(self.linear2(x)))  # (batch_size, hidden_channels[1])
        return x
