import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from torch_scatter import scatter_add, scatter_max
from torch_geometric.utils import softmax

from torch_geometric.nn.inits import uniform


class HUNET(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        kwargs['hunet_depth'] = 3
        kwargs['drop_out'] = 0.5
        kwargs['pool_ratios'] = 0.5
        self.dim_feat = kwargs['in_channel']
        # self.n_categories = kwargs['n_categories']
        self.n_stack = 1#kwargs['n_layer']
        layer_spec = [64]
        self.dims_in = [self.dim_feat] + layer_spec
        self.dims_out = layer_spec# + [self.n_categories]
        self.out_act = nn.LogSoftmax(dim=-1)
        self.dim_reduce = HGNN_conv(self.dim_feat,layer_spec[0])
        # self.H = torch.Tensor(kwargs['H_for_hunet']).to(device='cuda:0')
        self.hunets = nn.ModuleList([HyperUNet(
            dim_in=layer_spec[i],
            dim_out=layer_spec[i+1],
            dropout_rate=kwargs['dropout_rate'],
            activation=nn.ReLU(),
            depth=kwargs["hunet_depth"],
            pool_ratios = kwargs['pool_ratios'],
            sum_res=True,
            should_drop=True,
            H_for_hunet=kwargs['H_for_hunet']) if i < kwargs['n_stack'] - 1 else HyperUNet(
            dim_in=layer_spec[i],
            dim_out=self.n_categories,
            dropout_rate=kwargs['dropout_rate'],
            activation=self.out_act,
            depth=kwargs["hunet_depth"],
            pool_ratios = kwargs['pool_ratios'],
            sum_res=True,
            should_drop=False,
            H_for_hunet=kwargs['H_for_hunet'])
                                 for i in range(kwargs['n_stack'])])

    def forward(self, data):
        x = data.x
        H = data.adj
        x = self.dim_reduce(x,H)
        for i in range(len(self.hunets)):
            x = self.hunets[i](x)
        return x
    
class HyperUNet(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.dim_in = kwargs['dim_in']
        self.dim_out = kwargs['dim_out']
        self.activation = kwargs['activation']
        self.depth = kwargs['depth']
        self.pool_ratios = kwargs['pool_ratios']

        # Only used when stacking HUNets
        self.should_drop = kwargs['should_drop']
        self.dropout = nn.Dropout(p=kwargs['dropout_rate'])

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.up_convs = torch.nn.ModuleList()

        self.hunet_act = F.relu
        self.sum_res = kwargs['sum_res']
        self.H = np.array(kwargs['H_for_hunet'], dtype=np.float32)

        for i in range(self.depth):
            self.pools.append(TopKPooling(self.dim_in, self.pool_ratios))
            self.down_convs.append(HGNN_conv(self.dim_in, self.dim_in))

        for i in range(self.depth - 1):
            self.up_convs.append(HGNN_conv(self.dim_in, self.dim_in))
        self.up_convs.append(HGNN_conv(self.dim_in, self.dim_out))

    def forward(self, feat):
        x = feat
        xsaved = [x]
        graphs = [torch.Tensor(self.H).to('cuda:0')]
        perms = [range(len(x))]
        for i in range(1, self.depth + 1):
            x, batch, perm, _ = self.pools[i - 1](x, None)

            H = torch.tensor(np.array([self.H[perm[i], perm.cpu().numpy()] for i in range(len(perm))]).reshape(
                (len(perm), len(perm)))).to('cuda:0')
            x = self.down_convs[i - 1](x, H)
            x = self.hunet_act(x)

            if i < self.depth:
                xsaved += [x]
                graphs += [H]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - i - 1

            res = xsaved[j]
            H = graphs[j]
            perm = perms[j + 1]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.stack((res, up), dim=1)
            x = self.up_convs[i](x, H)
            x = self.activation(x) if i == self.depth - 1 else self.hunet_act(x)
        if self.should_drop:
            x = self.dropout(x)
        return x


class HGNN_conv(nn.Module):

    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, H: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = H.matmul(x)
        return x
    



def topk(x, ratio, batch, min_score=None, tol=1e-7):
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter_max(x, batch)[0][batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = torch.nonzero(x > scores_min).view(-1)
    else:
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ), -2)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
        mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)

        perm = perm[mask]

    return perm

class TopKPooling(torch.nn.Module):
    #Modified Top K Pooling that uses H instead of Adjacency Matrix
    r""":math:`\mathrm{top}_k` pooling operator from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_, `"Towards Sparse
    Hierarchical Graph Classifiers" <https://arxiv.org/abs/1811.01287>`_
    and `"Understanding Attention and Generalization in Graph Neural
    Networks" <https://arxiv.org/abs/1905.02850>`_ papers

    if min_score :math:`\tilde{\alpha}` is None:

        .. math::
            \mathbf{y} &= \frac{\mathbf{X}\mathbf{p}}{\| \mathbf{p} \|}

            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot
            \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}

    if min_score :math:`\tilde{\alpha}` is a value in [0, 1]:

        .. math::
            \mathbf{y} &= \mathrm{softmax}(\mathbf{X}\mathbf{p})

            \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot \mathbf{y})_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},

    where nodes are dropped based on a learnable projection score
    :math:`\mathbf{p}`.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`.
            This value is ignored if min_score is not None.
            (default: :obj:`0.5`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        multiplier (float, optional): Coefficient by which features gets
            multiplied after pooling. This can be useful for large graphs and
            when :obj:`min_score` is used. (default: :obj:`1`)
        nonlinearity (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.tanh`)
    """

    def __init__(self, in_channels, ratio=0.5, min_score=None, multiplier=1,
                 nonlinearity=torch.tanh):
        super(TopKPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.weight = Parameter(torch.Tensor(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.weight)

    def forward(self, x, batch=None, attn=None):
        """"""

        if batch is None:
            batch = x.new_zeros(x.size(0),dtype=torch.int64)

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = (attn * self.weight).sum(dim=-1)

        if self.min_score is None:
            score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        #edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
         #                                  num_nodes=score.size(0))

        return x,  batch, perm, score[perm]

    def __repr__(self):
        return '{}({}, {}={}, multiplier={})'.format(
            self.__class__.__name__, self.in_channels,
            'ratio' if self.min_score is None else 'min_score',
            self.ratio if self.min_score is None else self.min_score,
            self.multiplier)