import torch
from torch import nn

from typing import Optional, Tuple, Union
from torch_geometric.nn.dense.linear import Linear


class DetourTransformer(nn.Module):

    def __init__(self, 
        nlayer: int = 1,
        node_sz: int=116,
        in_channels: Union[int, Tuple[int, int]] = 10,
        out_channels: int = 10,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=out_channels, nhead=heads, dim_feedforward=out_channels, dropout=dropout),
            num_layers=nlayer,
            norm=nn.LayerNorm(out_channels)
        )
        self.nlayer = nlayer
        self.pe_lin = nn.Linear(node_sz*2, out_channels)
        self.dee_lin = nn.Linear(1, out_channels)
        self.lin_identifier = Linear(1, out_channels)
        self.extra_encs = nn.ModuleList([
            self.pe_lin, self.dee_lin, self.lin_identifier 
        ])
        self.in_bn = nn.BatchNorm1d(out_channels, affine=True)
        self.lin_in = nn.Linear(in_channels, out_channels)


    def forward(self, mdnn_input):
        x, extra_encodings, pad_mask = mdnn_input
        x = self.lin_in(x)
        x = self.in_bn(x)
        for ei, encoding in enumerate(extra_encodings):
            if encoding is not None:
                x = x + self.extra_encs[ei](encoding)
                
        x = self.net(x)
        x = x[pad_mask]
        return x
