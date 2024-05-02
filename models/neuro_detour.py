import torch
from torch import nn

from typing import Optional, Tuple, Union
from torch_geometric.nn.dense.linear import Linear


class DetourTransformer(nn.Module):

    def __init__(self, 
        nlayer: int = 2,
        node_sz: int=116,
        in_channel: Union[int, Tuple[int, int]] = 10,
        out_channel: int = 10,
        heads: int = 1,
        concat: bool = True,
        dek: int = 4,
        pek: int = 10,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=out_channel, nhead=heads, dim_feedforward=out_channel, dropout=dropout),
            num_layers=nlayer,
            norm=nn.LayerNorm(out_channel)
        )
        self.nlayer = nlayer
        self.pe_lin = nn.Linear(pek, out_channel)
        self.dee_lin = nn.Linear(dek, out_channel)
        self.lin_identifier = Linear(1, out_channel)
        self.extra_encs = nn.ModuleList([
            self.pe_lin, self.dee_lin, self.lin_identifier 
        ])
        self.in_bn = nn.BatchNorm1d(out_channel, affine=True)
        self.lin_in = nn.Linear(in_channel, out_channel)


    def forward(self, data):
        x, extra_encodings, pad_mask = data.token, (data.PE, data.DE, data.ID), data.mask
        x = self.lin_in(x)
        x = self.in_bn(x)
        for ei, encoding in enumerate(extra_encodings):
            if encoding is not None:
                x = x + self.extra_encs[ei](encoding)
                
        x = self.net(x)
        x = x[pad_mask]
        return x
