import torch.nn as nn


class Highway(nn.Module):
    """
    Highway networks are a special type of neural network layers that enable
    unimpeded information flow across the layers. They are used to control the
    flow of information through the network.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super(Highway, self).__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim

        self.proj = nn.Linear(in_dim, out_dim)
        # fill bias with 0 for the highway gate
        self.proj.bias.data.zero_()
        self.gate = nn.Linear(in_dim, out_dim)
        # fill bias with negative value for the highway gate
        # to allow the information to pass through initially
        self.gate.bias.data.fill_(-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, in_channels)
        x_proj = self.relu(self.proj(x))  # (batch_size, out_channels)
        x_gate = self.sigmoid(self.gate(x))  # (batch_size, out_channels)
        x_highway = x_proj * x_gate + x * (1 - x_gate)  # (batch_size, out_channels)
        return x_highway
