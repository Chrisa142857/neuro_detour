import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs) -> None:
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU()
        )

    def forward(self, batch):
        return self.lin(batch.x)