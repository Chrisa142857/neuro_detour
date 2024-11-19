import torch
from datasets import dataloader_generator
from models.neuro_detour import DetourTransformer
from torch.nn.modules.transformer import TransformerEncoderLayer
import torch.nn.functional as F
import torch.nn as nn

class VisTransformerEncoderLayer(TransformerEncoderLayer):
    def forward(
            self,
            src,
            src_mask = None,
            src_key_padding_mask = None,
            is_causal = False):
        
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first :
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim :
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.device.type in _supported_device_type) for x in tensor_args):
                why_not_sparsity_fast_path = ("some Tensor argument's device is neither one of "
                                              f"{_supported_device_type}")
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                )


        x = src
        x, attn_weights = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
        return attn_weights

    # self-attention block
    def _sa_block(self, x,
                  attn_mask, key_padding_mask, is_causal = False):
        x, attn_weights = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True, is_causal=is_causal, average_attn_weights=False)
        return self.dropout1(x), attn_weights


class VisDetourTransformer(DetourTransformer):

    def __init__(self, 
        heads: int = 8,
        nlayer: int = 1,
        node_sz: int=116,
        in_channel = 10,
        out_channel: int = 10,
        concat: bool = False,
        dek: int = 4,
        pek: int = 10,
        dropout: float = 0.1,
        edge_dim = None,
        bias: bool = True,
        hiddim: int = 1024,
        detour_type = 'node',
        batch_size = 32,
        device='cpu',
        *args, **kwargs) -> None:
        
        super(DetourTransformer, self).__init__()
        org_in_channel = in_channel
        if in_channel % heads != 0:
            in_channel = in_channel  + heads - (in_channel % heads)
        self.detour_type = detour_type
        self.nlayer = nlayer
        self.node_sz = node_sz
            
        self.lin_first = nn.Sequential(
            nn.Linear(org_in_channel, in_channel), 
            nn.BatchNorm1d(in_channel), 
            nn.LeakyReLU()
        )
        self.lin_in = nn.Sequential(
            nn.Linear(in_channel, out_channel), 
            nn.BatchNorm1d(out_channel), 
            nn.LeakyReLU(),
        )
        self.net = nn.ModuleList([torch.nn.TransformerEncoder(
            VisTransformerEncoderLayer(d_model=in_channel, nhead=heads, dim_feedforward=hiddim, dropout=dropout, batch_first=True),
            num_layers=1,
            norm=None#nn.LayerNorm(in_channel)
        ) for _ in range(nlayer)])
        self.net_fc = nn.ModuleList([torch.nn.TransformerEncoder(
            VisTransformerEncoderLayer(d_model=in_channel, nhead=heads, dim_feedforward=hiddim, dropout=dropout, batch_first=True),
            num_layers=1,
            norm=None#nn.LayerNorm(in_channel)
        ) for _ in range(nlayer)])
        self.heads = heads
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mask_heldout = torch.zeros(batch_size, node_sz, node_sz) - torch.inf
        self.mask_heldout = self.mask_heldout.to(device)


    def forward(self, data):
        node_feature = data.x
        node_feature = self.lin_first(node_feature)
        node_feature = node_feature.view(data.batch.max()+1, len(torch.where(data.batch==0)[0]), self.in_channel)
        node_feature_fc = node_feature
        adj = data.adj_sc
        adj_fc = data.adj_fc
        adj[:, torch.arange(self.node_sz), torch.arange(self.node_sz)] = True
        adj_fc[:, torch.arange(self.node_sz), torch.arange(self.node_sz)] = True
        org_adj = adj
        multi_mask = []
        for _ in range(self.heads):
            mask = self.mask_heldout[:len(adj)]
            mask[torch.logical_and(adj, adj_fc)] = 0
            adj = (adj.float() @ org_adj.float()) > 0
            multi_mask.append(mask)
        multi_mask = torch.cat(multi_mask)
        mask_fc = self.mask_heldout[:len(adj_fc)]
        mask_fc[adj_fc] = 0
        mask_fc = mask_fc.repeat(self.heads, 1, 1)
        for i in range(self.nlayer):
            attn_weights = self.net[i](node_feature, mask=multi_mask)
            attn_weights_fc = self.net_fc[i](node_feature_fc, mask=mask_fc)

        return attn_weights, attn_weights_fc



node_sz = 116
atlas = 'AAL_116'
node_attr = 'FC'
foldi = 0
dname = 'adni'
# mweight_fn = f'model_weights/neurodetour_{atlas}_boldwin500_FC{node_attr}/fold{foldi}_2024-05-14-11-08-29-992532.pt' # HCPA
# mweight_fn = 'model_weights/neurodetour_D_160_boldwin500_FCFC/fold0_2024-05-14-12-55-24-702634.pt' # OASIS
mweight_fn = 'model_weights/neurodetour_AAL_116_boldwin500_FCFC/fold0_2024-05-14-12-53-34-133933.pt' # ADNI
# mweight_fn = 'model_weights/neurodetour_Gordon_333_boldwin500_FCFC/fold0_2024-05-14-12-07-29-167949.pt' # UKB
mweight = torch.load(mweight_fn, map_location='cpu')

model = VisDetourTransformer(node_sz=node_sz, in_channel=node_sz, out_channel=768)
model.load_state_dict(mweight)
model = model.cpu()
print(model)

w2 = model.net[0].layers[0].self_attn.out_proj.weight.detach()
# print(w2, w2.shape)
w2 = torch.tensor_split(w2, torch.LongTensor([i for i in range(w2.shape[0]//8, w2.shape[0], w2.shape[0]//8)]))
print([w.shape for w in w2])
w2 = torch.stack(w2).abs().sum(1).sum(1) / (42*336)
print(w2, w2.shape)
# exit()
tl, vl, ds = dataloader_generator(nfold=foldi, atlas_name=atlas, node_attr=node_attr, dname=dname)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm, trange
heads = 8
bsz = 4
multihop_path_dict = {
    'label': []
}
if dname == 'adni':
    tgt_c = [53, 59] # defualt [73, 87] subcortical [72, 86] cortical-optical [53, 59]
elif dname == 'oasis':
    tgt_c = [93, 138] # defualt [17, 110] subcortical [16, 36] cortical [93, 138]
path_n = 1
subi = 0
for data in tl:
    # print(data.x.shape, data.y)
    attn_td, attn_fc = model(data)
    fc = torch.stack(torch.tensor_split(data.x, data.ptr)[1:-1])
    mat1 = attn_td.detach()#.numpy()
    mat2 = attn_fc.detach()#.numpy()
    mat1 = torch.softmax(mat1,3)
    # print(mat1.sum(2), mat1.sum(3))
    # exit()
    data.adj_sc[:, torch.arange(node_sz), torch.arange(node_sz)] = False
    ## Get pathways   
    for bi in range(bsz): 
        subi += 1
        multihop_path_dict['label'].append(data.y[bi])
        # avgmat1 = mat1[bi].mean(0)
        multihop_path = {}
        for ci in tgt_c:
        # for ci in trange(mat1.shape[-1]):
            # if ci == 3: break
            pathways = []
            pathws = []
            for h in range(heads):
                headmat = mat1[bi, h]
                headmat[torch.arange(node_sz), torch.arange(node_sz)] = 0
                rowi = torch.where(headmat[:, ci]>0)[0]
                nodes = torch.LongTensor([ci]).repeat(len(rowi)) # node
                w1 = headmat[rowi, nodes] # weight after softmax
                pathw = w1*w2[h]
                if len(pathways) == 0: 
                    pathways = torch.stack([nodes, rowi], 1)
                    pathws = pathw[:, None]
                else:
                    curr_node = rowi.repeat(len(pathways),1).T.reshape(-1)
                    curr_w = pathw.repeat(len(pathways),1).T.reshape(-1)
                    pathways = pathways.repeat(len(rowi),1)
                    adjacent = data.adj_sc[bi, pathways[:, -1], curr_node]
                    no_repeat = (pathways != curr_node[:, None]).all(1)
                    adjacent = (adjacent & no_repeat).bool()
                    pathways = torch.cat([pathways[adjacent], curr_node[adjacent, None]],1)
                    pathways = torch.unique(pathways, dim=0)
                    pathws = torch.cat([pathws.repeat(len(rowi), 1)[adjacent], curr_w[adjacent, None]], 1)
                if pathways.shape[0] == 0: break
                if pathways.shape[1] > 2:
                    fc_ind = torch.LongTensor([i for i in range(len(pathways)) if pathways[i, -1] in tgt_c])
                    pathw = pathws.mean(1)[fc_ind]
                    ind = torch.argsort(pathw, descending=True)[:path_n]
                    path = torch.cat([
                        pathw[ind, None],
                        pathways[fc_ind][ind]
                    ], 1)
                    # print(path.shape)
                    # exit()
                    if f'{h+1}hop' not in multihop_path: multihop_path[f'{h+1}hop'] = []
                    multihop_path[f'{h+1}hop'].append(path)

        path_adj = torch.zeros(node_sz, node_sz)
        path_i = 0
        weights = []
        paths = []
        headi, pathi = [], []
        for ki, key in enumerate(multihop_path):
            multihop_path[key] = torch.cat(multihop_path[key], 0)
            weights.append(multihop_path[key][:, 0])
            # paths.append(torch.cat([multihop_path[key][:, 1:], torch.zeros(len(multihop_path[key]), pathways.shape[1]-multihop_path[key][:, 1:].shape[1])-ki-1],1))
            paths.append(multihop_path[key][:, 1:].long())
            headi.extend([ki for _ in range(len(multihop_path[key]))])
            pathi.extend([i for i in range(len(multihop_path[key]))])
        weights = torch.cat(weights)
        # paths = torch.cat(paths)
        # overlap = paths.repeat(len(paths), 1, 1) == paths.repeat(len(paths), 1, 1)
        # overlap = overlap.any(-1)
        # iso_path_id = []
        iso_paths = []
        for i in torch.argsort(weights, descending=True):
            # if overlap[i, iso_path_id].any() or path_i >= path_n: continue
            if path_i >= path_n: continue
            # iso_path_id.append(i.item())
            # path_adj[paths[headi[i]][pathi[i], :-1], paths[headi[i]][pathi[i], 1:]] = torch.stack([fc[bi, paths[headi[i]][pathi[i], j], paths[headi[i]][pathi[i], j+1]] for j in range(paths[headi[i]].shape[1]-1)])
            path_adj[paths[headi[i]][pathi[i], :-1], paths[headi[i]][pathi[i], 1:]] = (20-(path_i+1))/20#weights[i].repeat(paths[headi[i]].shape[1]-1)
            # path_adj[paths[headi[i]][pathi[i], 0], paths[headi[i]][pathi[i], -1]] = fc[bi, paths[headi[i]][pathi[i], 0], paths[headi[i]][pathi[i], -1]].abs()
            iso_paths.append(paths[headi[i]][pathi[i]])
            path_i += 1
        # iso_paths = paths[iso_path_id].long()
        # fc = torch.stack(torch.tensor_split(data.x, data.ptr)[1:-1])
        # pathweights = torch.cat([fc[bi, iso_paths[:, i], iso_paths[:, i+1]] for i in range(iso_paths.shape[1]-1)])
        # # pathweights = (pathweights-pathweights.min())/(pathweights.max()-pathweights.min())
        # path_adj[iso_paths[:, :-1], iso_paths[:, 1:]] = pathweights
        print(f'Sub {subi} Label {data.y[bi].item()} Got {path_i} paths', iso_paths)
        tgt_cstr = "-".join([str(tgt_ci) for tgt_ci in tgt_c])
        np.savetxt(f'resources/{atlas}_{dname}_top{path_n}Path_signFC{tgt_cstr}_sub{subi}_label{data.y[bi].item()}_adj.edge', path_adj.numpy(), delimiter='\t', fmt='%.5f')
#             if key not in multihop_path_dict:
#                 multihop_path_dict[key] = []
#             multihop_path_dict[key].append(multihop_path[key])
# torch.save(multihop_path_dict, f'resources/{dname}_{atlas}_FC{node_attr}_fold{foldi}_traindata_multiHopPath.pt')
exit() 
'''
Attention maps
'''
# for data in tl:
#     print(data.x.shape, data.y)
#     attn_td, attn_fc = model(data)
#     mat1 = attn_td.detach().numpy()
#     mat2 = attn_fc.detach().numpy()
#     ## Plot attn map
#     fig, axes = plt.subplots(bsz, 1, figsize=(3, bsz*3), sharex=True, sharey=True)
#     for bi in range(bsz):
#         sns.heatmap(mat1[bi].mean(0), ax=axes[bi])
#         axes[bi].set_title(f'label{data.y[bi]}')
#         axes[bi].axis("off")
#     plt.tight_layout()
#     plt.savefig('figs/attnTDavg_map.png')
#     plt.close()
#     fig, axes = plt.subplots(bsz, heads, figsize=(heads*3, bsz*3), sharex=True, sharey=True)
    
#     for bi in range(bsz):
#         for h in range(heads):
#             sns.heatmap(mat1[bi, h], ax=axes[bi, h])
#             axes[bi, h].set_title(f'label{data.y[bi]} head{h}')
#             axes[bi, h].axis("off")
#     plt.tight_layout()
#     plt.savefig('figs/attnTD_map.png')
#     plt.close()

#     fig, axes = plt.subplots(bsz, 1, figsize=(3, bsz*3), sharex=True, sharey=True)
#     for bi in range(bsz):
#         sns.heatmap(mat2[bi].mean(0), ax=axes[bi])
#         axes[bi].set_title(f'label{data.y[bi]}')
#         axes[bi].axis("off")
#     plt.tight_layout()
#     plt.savefig('figs/attnFCavg_map.png')
#     plt.close()
#     fig, axes = plt.subplots(bsz, heads, figsize=(heads*3, bsz*3), sharex=True, sharey=True)
#     for bi in range(bsz):
#         for h in range(heads):
#             sns.heatmap(mat2[bi, h], ax=axes[bi, h])
#             axes[bi, h].set_title(f'label{data.y[bi]} head{h}')
#             axes[bi, h].axis("off")
#     plt.tight_layout()
#     plt.savefig('figs/attnFC_map.png')
#     plt.close()
#     print(attn_td.shape, attn_fc.shape)
#     exit()
