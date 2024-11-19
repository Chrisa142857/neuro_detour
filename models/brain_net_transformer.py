import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .BNT.ptdec import DEC
from typing import List
from .BNT.components import InterpretableTransformerEncoder
from omegaconf import DictConfig
from .BNT.base import BaseModel
import hydra
from torch_geometric.data import Data

class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True, freeze_center=False, project_assignment=True):
        super().__init__()
        self.transformer = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=4 if input_feature_size%4 == 0 else 3,
                                                           dim_feedforward=hidden_size,
                                                           batch_first=True)

        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


class BrainNetworkTransformer(BaseModel):
    
    def __init__(self, node_sz, in_channel, out_channel, pos_encoding=None, pos_embed_dim=360, hidden_size=1024, nlayer=1, **kargs):

        super().__init__()
        hidden_size = hidden_size
        self.node_sz = node_sz

        self.attention_list = nn.ModuleList()
        forward_dim = in_channel

        self.pos_encoding = pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(
                node_sz, pos_embed_dim), requires_grad=True)
            forward_dim = in_channel + pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        # sizes = [out_channel//8, out_channel//8]
        # # sizes = [out_channel//8 for _ in range(nlayer+1)]
        # sizes[0] = node_sz
        # in_sizes = [node_sz] + sizes[:-1]
        # do_pooling = [False, True]
        do_pooling = [False for _ in range(nlayer)] + [True]
        self.do_pooling = do_pooling
        # for index, size in enumerate(sizes):
        for index in range(nlayer+1):
            self.attention_list.append(
                TransPoolingEncoder(input_feature_size=forward_dim,
                                    input_node_num=node_sz,#in_sizes[index],
                                    hidden_size=hidden_size,
                                    output_node_num=node_sz,
                                    pooling=do_pooling[index],
                                    orthogonal=True,
                                    freeze_center=True,
                                    project_assignment=True))

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, out_channel),
            nn.LeakyReLU()
        )

        # self.fc = nn.Sequential(
        #     nn.Linear(8 * sizes[-1], 256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 32),
        #     nn.LeakyReLU(),
        #     nn.Linear(32, nclass)
        # )
        # self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch: Data, label=None):
        node_feature = batch.x.view(batch.batch.max()+1, len(torch.where(batch.batch==0)[0]), batch.x.shape[1])
        bz, _, _, = node_feature.shape

        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        assignments = []

        for atten in self.attention_list:
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)

        node_feature = self.dim_reduction(node_feature)
        # node_feature = node_feature.reshape((bz, -1))

        # out = self.fc(node_feature)
        # if label is not None:
        #     loss = self.loss_fn(out, label)
        #     return out, loss
        # else:
        #     return out
        return node_feature.view(node_feature.shape[0] * node_feature.shape[1], node_feature.shape[-1])

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    # def loss(self, assignments):
    #     """
    #     Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
    #     Inputs: assignments: [batch size, number of clusters]
    #     Output: KL loss
    #     """
    #     decs = list(
    #         filter(lambda x: x.is_pooling_enabled(), self.attention_list))
    #     assignments = list(filter(lambda x: x is not None, assignments))
    #     loss_all = None

    #     for index, assignment in enumerate(assignments):
    #         if loss_all is None:
    #             loss_all = decs[index].loss(assignment)
    #         else:
    #             loss_all += decs[index].loss(assignment)
    #     return loss_all