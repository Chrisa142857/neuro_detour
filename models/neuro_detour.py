import torch
from torch import nn

from typing import Optional, Tuple, Union
from torch_geometric.nn.dense.linear import Linear


class DetourTransformer(nn.Module):

    def __init__(self, 
        nlayer: int = 1,
        node_sz: int=116,
        in_channel: Union[int, Tuple[int, int]] = 10,
        out_channel: int = 10,
        heads: int = 2,
        concat: bool = False,
        dek: int = 4,
        pek: int = 10,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        hiddim: int = 1024,
        detour_type = 'node',
        batch_size = 32,
        device='cuda:0',
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.detour_type = detour_type
        self.nlayer = nlayer
        self.node_sz = node_sz
        if concat:
            lin_dim = in_channel//3
        else:
            lin_dim = in_channel
            
        self.pe_lin = nn.Sequential(nn.Linear(pek, lin_dim), nn.BatchNorm1d(lin_dim), nn.LeakyReLU())
        self.dee_lin = nn.Sequential(nn.Linear(dek, lin_dim), nn.BatchNorm1d(lin_dim), nn.LeakyReLU())
        self.lin_identifier = nn.Sequential(nn.Linear(1, lin_dim), nn.BatchNorm1d(lin_dim), nn.LeakyReLU())
        self.extra_encs = nn.ModuleList([
            # self.pe_lin, 
            self.dee_lin, 
            # self.lin_identifier 
        ])

        # in_channel = in_channel * heads
        # self.node_identity = nn.Parameter(torch.zeros(
        #     node_sz, out_channel), requires_grad=True)
        # in_channel = node_sz + out_channel
        # nn.init.kaiming_normal_(self.node_identity)

        self.lin_first = nn.Sequential(
            nn.Linear(in_channel, in_channel), 
            nn.BatchNorm1d(in_channel), 
            nn.LeakyReLU()
        )
        self.in_bn = nn.BatchNorm1d(in_channel, affine=True)
        # self.lin_in = nn.Linear(in_channel, out_channel)
        self.lin_in = nn.Sequential(
            nn.Linear(in_channel, out_channel), 
            nn.BatchNorm1d(out_channel), 
            nn.LeakyReLU(),
        )
        self.concat = concat
        if concat:
            self.lin_concat = nn.Sequential(
                nn.Linear(out_channel*2, out_channel),
                nn.BatchNorm1d(out_channel),
                nn.LeakyReLU()
            )
        # self.net = torch.nn.TransformerEncoder(
        #     torch.nn.TransformerEncoderLayer(d_model=in_channel, nhead=heads, dim_feedforward=in_channel, dropout=dropout, batch_first=True),
        #     num_layers=nlayer,
        #     norm=None#nn.LayerNorm(in_channel)
        # )
        self.net = nn.ModuleList([torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=in_channel, nhead=heads, dim_feedforward=hiddim, dropout=dropout, batch_first=True),
            num_layers=1,
            norm=None#nn.LayerNorm(in_channel)
        ) for _ in range(nlayer)])
        self.net_fc = nn.ModuleList([torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=in_channel, nhead=heads, dim_feedforward=hiddim, dropout=dropout, batch_first=True),
            num_layers=1,
            norm=None#nn.LayerNorm(in_channel)
        ) for _ in range(nlayer)])
        self.heads = heads
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mask_heldout = torch.zeros(batch_size, node_sz, node_sz) - torch.inf
        self.mask_heldout = self.mask_heldout.to(device)
        self.fcsc_loss = nn.MSELoss()
        self.loss = 0

    def forward(self, data):
        self.loss = 0
        node_feature = data.x
        node_feature = self.lin_first(node_feature)
        node_feature = node_feature.view(data.batch.max()+1, len(torch.where(data.batch==0)[0]), data.x.shape[1])
        node_feature_fc = node_feature
        bz, _, _, = node_feature.shape
        adj = data.adj_sc
        adj[:, torch.arange(self.node_sz), torch.arange(self.node_sz)] = True
        org_adj = adj
        multi_mask = []
        for _ in range(self.heads):
            mask = self.mask_heldout[:len(adj)]
            mask[adj] = 0
            adj = (adj.float() @ org_adj.float()) > 0
            multi_mask.append(mask)
        multi_mask = torch.cat(multi_mask)
        adj_fc = data.adj_fc
        adj_fc[:, torch.arange(self.node_sz), torch.arange(self.node_sz)] = True
        mask_fc = self.mask_heldout[:len(adj_fc)]
        mask_fc[adj_fc] = 0
        mask_fc = mask_fc.repeat(self.heads, 1, 1)
        for i in range(self.nlayer):
            node_feature = self.net[i](node_feature, mask=multi_mask)
            node_feature_fc = self.net_fc[i](node_feature_fc, mask=mask_fc)
            self.loss = self.loss + self.fcsc_loss(node_feature_fc, node_feature)
        if not self.training:
            node_feature = node_feature_fc
        return self.lin_in(node_feature.reshape(node_feature.shape[0] * node_feature.shape[1], self.in_channel))
        # x, readout_mask = data.token, data.mask
        # extra_encodings = (
        #     data.PE, 
        #     data.DE, 
        #     torch.arange(data.x.shape[0])[:, None].to(data.x.device).float() if self.detour_type=='node' else data.ID
        # )

        # bsz = data.batch.max()+1
        
        # x = self.lin_in(x)
        # x = self.in_bn(x)
        # for ei, encoding in enumerate(extra_encodings):
        #     if encoding is not None:
        #         if self.concat:
        #             x = torch.cat([x, self.extra_encs[ei](encoding)], -1)
        #         else:
        #             x = x + self.extra_encs[ei](encoding)
        # if self.concat:
        #     x = self.lin_concat(x)
        
        # if self.detour_type == 'node':
        #     # x.shape = [bsz * node_num, dim]
        #     x = x.view(bsz, self.node_sz, x.shape[1]) # bsz x node_num x dim
        #     # src_mask = torch.ones_like(x[..., 0]).bool()
        #     x = self.net(x).view(x.shape[0]*x.shape[1], x.shape[-1])
        
        # elif self.detour_type == 'edge':
        #     # x.shape = [bsz * edge_num, dim]
        #     spilt_ind = torch.where(readout_mask)[0]# len = bsz * node_num
        #     spilt_ind = spilt_ind[::self.node_sz] # len = bsz, aka, graph_num
        #     src_mask = torch.nn.utils.rnn.pad_sequence(torch.tensor_split(torch.ones_like(x[..., 0]).bool(), spilt_ind.cpu())[1:], batch_first=True, padding_value=False) # bsz x seq 
        #     x = torch.nn.utils.rnn.pad_sequence(torch.tensor_split(x, spilt_ind.cpu())[1:], batch_first=True) # bsz x seq x dim
        #     if not src_mask.all():
        #         x = self.net(x, src_key_padding_mask=src_mask) # bsz x seq x dim
        #     else:
        #         x = self.net(x)
        #     x = x[src_mask] # bsz*edge_num x dim
        
        # x = x[readout_mask]
        # return x
