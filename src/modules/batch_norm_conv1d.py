import torch.nn as nn


class BatchNormConv1d(nn.Module):
    """
    Batch normalization followed by a 1D convolution.
    Takes an optional activation function.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        activation: nn.Module = None,
    ):
        super(BatchNormConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.activation = activation
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x: (batch_size, in_channels, seq_len)
        x = self.conv1d(x)  # (batch_size, out_channels, seq_len)
        # apply activation function if given
        if self.activation:
            x = self.activation(x)
        x = self.bn(x)  # (batch_size, out_channels, seq_len)
        return x
