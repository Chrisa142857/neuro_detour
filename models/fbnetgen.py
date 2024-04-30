import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, MaxPool1d, Linear, GRU
from omegaconf import DictConfig
from BNT.base import BaseModel


class FBNETGEN(BaseModel):

    def __init__(self, node_sz, timeseries_sz, node_feature_sz, window_size=4, embedding_size=16, num_gru_layers=4, cnn_pool_size=16, graph_generation='product'):
        super().__init__()

        assert 'gru' in ['cnn', 'gru']
        assert graph_generation in ['linear', 'product']
        assert timeseries_sz % window_size == 0

        self.graph_generation = graph_generation
        if 'gru' == 'cnn':
            self.extract = ConvKRegion(
                out_size=embedding_size, kernel_size=window_size,
                time_series=timeseries_sz)
        elif 'gru' == 'gru':
            self.extract = GruKRegion(
                out_size=embedding_size, kernel_size=window_size,
                layers=num_gru_layers)
        if self.graph_generation == "linear":
            self.emb2graph = Embed2GraphByLinear(
                embedding_size, roi_num=node_sz)
        elif self.graph_generation == "product":
            self.emb2graph = Embed2GraphByProduct(
                embedding_size, roi_num=node_sz)

        self.predictor = GNNPredictor(
            node_feature_sz, roi_num=node_sz)

    def forward(self, time_seires, node_feature, label=None):
        x = self.extract(time_seires)
        x = F.softmax(x, dim=-1)
        m = self.emb2graph(x)
        m = m[:, :, :, 0]

        return self.predictor(m, node_feature), m
    


class GruKRegion(nn.Module):

    def __init__(self, kernel_size=8, layers=4, out_size=8, dropout=0.5):
        super().__init__()
        self.gru = GRU(kernel_size, kernel_size, layers,
                       bidirectional=True, batch_first=True)

        self.kernel_size = kernel_size

        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            Linear(kernel_size*2, kernel_size),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(kernel_size, out_size)
        )

    def forward(self, raw):

        b, k, d = raw.shape

        x = raw.view((b*k, -1, self.kernel_size))

        x, h = self.gru(x)

        x = x[:, -1, :]

        x = x.view((b, k, -1))

        x = self.linear(x)
        return x


class ConvKRegion(nn.Module):

    def __init__(self, k=1, out_size=8, kernel_size=8, pool_size=16, time_series=512):
        super().__init__()
        self.conv1 = Conv1d(in_channels=k, out_channels=32,
                            kernel_size=kernel_size, stride=2)

        output_dim_1 = (time_series-kernel_size)//2+1

        self.conv2 = Conv1d(in_channels=32, out_channels=32,
                            kernel_size=8)
        output_dim_2 = output_dim_1 - 8 + 1
        self.conv3 = Conv1d(in_channels=32, out_channels=16,
                            kernel_size=8)
        output_dim_3 = output_dim_2 - 8 + 1
        self.max_pool1 = MaxPool1d(pool_size)
        output_dim_4 = output_dim_3 // pool_size * 16
        self.in0 = nn.InstanceNorm1d(time_series)
        self.in1 = nn.BatchNorm1d(32)
        self.in2 = nn.BatchNorm1d(32)
        self.in3 = nn.BatchNorm1d(16)

        self.linear = nn.Sequential(
            Linear(output_dim_4, 32),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(32, out_size)
        )

    def forward(self, x):

        b, k, d = x.shape

        x = torch.transpose(x, 1, 2)

        x = self.in0(x)

        x = torch.transpose(x, 1, 2)
        x = x.contiguous()

        x = x.view((b*k, 1, d))

        x = self.conv1(x)

        x = self.in1(x)
        x = self.conv2(x)

        x = self.in2(x)
        x = self.conv3(x)

        x = self.in3(x)
        x = self.max_pool1(x)

        x = x.view((b, k, -1))

        x = self.linear(x)

        return x


class Embed2GraphByProduct(nn.Module):

    def __init__(self, input_dim, roi_num=264):
        super().__init__()

    def forward(self, x):

        m = torch.einsum('ijk,ipk->ijp', x, x)

        m = torch.unsqueeze(m, -1)

        return m


class GNNPredictor(nn.Module):

    def __init__(self, node_input_dim, roi_num=360):
        super().__init__()
        inner_dim = roi_num
        self.roi_num = roi_num
        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(inner_dim, inner_dim)
        )
        self.bn1 = torch.nn.BatchNorm1d(inner_dim)

        self.gcn1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = torch.nn.BatchNorm1d(inner_dim)
        self.gcn2 = nn.Sequential(
            nn.Linear(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = torch.nn.BatchNorm1d(inner_dim)

        self.fcn = nn.Sequential(
            nn.Linear(8*roi_num, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2)
        )

    def forward(self, m, node_feature):
        bz = m.shape[0]

        x = torch.einsum('ijk,ijp->ijp', m, node_feature)

        x = self.gcn(x)

        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn1(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum('ijk,ijp->ijp', m, x)

        x = self.gcn1(x)

        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn2(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum('ijk,ijp->ijp', m, x)

        x = self.gcn2(x)

        x = self.bn3(x)

        x = x.view(bz, -1)

        return self.fcn(x)


class Embed2GraphByLinear(nn.Module):

    def __init__(self, input_dim, roi_num=360):
        super().__init__()

        self.fc_out = nn.Linear(input_dim * 2, input_dim)
        self.fc_cat = nn.Linear(input_dim, 1)

        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot

        off_diag = np.ones([roi_num, roi_num])
        rel_rec = np.array(encode_onehot(
            np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(
            np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).cuda()
        self.rel_send = torch.FloatTensor(rel_send).cuda()

    def forward(self, x):

        batch_sz, region_num, _ = x.shape
        receivers = torch.matmul(self.rel_rec, x)

        senders = torch.matmul(self.rel_send, x)
        x = torch.cat([senders, receivers], dim=2)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)

        x = torch.relu(x)

        m = torch.reshape(
            x, (batch_sz, region_num, region_num, -1))
        return m



def continus_mixup_data(*xs, y=None, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = y.size()[0]
    index = torch.randperm(batch_size).to(device)
    new_xs = [lam * x + (1 - lam) * x[index, :] for x in xs]
    y = lam * y + (1-lam) * y[index]
    return *new_xs, y


def mixup_data_by_class(x, nodes, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    mix_xs, mix_nodes, mix_ys = [], [], []

    for t_y in y.unique():
        idx = y == t_y

        t_mixed_x, t_mixed_nodes, _, _, _ = continus_mixup_data(
            x[idx], nodes[idx], y[idx], alpha=alpha, device=device)
        mix_xs.append(t_mixed_x)
        mix_nodes.append(t_mixed_nodes)

        mix_ys.append(y[idx])

    return torch.cat(mix_xs, dim=0), torch.cat(mix_nodes, dim=0), torch.cat(mix_ys, dim=0)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_cluster_loss(matrixs, y, intra_weight=2):

    y_1 = y[:, 1]

    y_0 = y[:, 0]

    bz, roi_num, _ = matrixs.shape
    matrixs = matrixs.reshape((bz, -1))
    sum_1 = torch.sum(y_1)
    sum_0 = torch.sum(y_0)
    loss = 0.0

    if sum_0 > 0:
        center_0 = torch.matmul(y_0, matrixs)/sum_0
        diff_0 = torch.norm(matrixs-center_0, p=1, dim=1)
        loss += torch.matmul(y_0, diff_0)/(sum_0*roi_num*roi_num)
    if sum_1 > 0:
        center_1 = torch.matmul(y_1, matrixs)/sum_1
        diff_1 = torch.norm(matrixs-center_1, p=1, dim=1)
        loss += torch.matmul(y_1, diff_1)/(sum_1*roi_num*roi_num)
    if sum_0 > 0 and sum_1 > 0:
        loss += intra_weight * \
            (1 - torch.norm(center_0-center_1, p=1)/(roi_num*roi_num))

    return loss


def inner_loss(label, matrixs):

    loss = 0

    if torch.sum(label == 0) > 1:
        loss += torch.mean(torch.var(matrixs[label == 0], dim=0))

    if torch.sum(label == 1) > 1:
        loss += torch.mean(torch.var(matrixs[label == 1], dim=0))

    return loss


def intra_loss(label, matrixs):
    a, b = None, None

    if torch.sum(label == 0) > 0:
        a = torch.mean(matrixs[label == 0], dim=0)

    if torch.sum(label == 1) > 0:
        b = torch.mean(matrixs[label == 1], dim=0)
    if a is not None and b is not None:
        return 1 - torch.mean(torch.pow(a-b, 2))
    else:
        return 0

