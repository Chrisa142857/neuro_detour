import random
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# print(torch.cuda.is_available())
# from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch.utils.data import ConcatDataset, random_split, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm, trange
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
# import matplotlib as mpl
# mpl.use('Agg')
# from dataset2 import *
# from models import *
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.model_selection import KFold
# from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
import argparse
from torch_geometric.data import Data
import os, re

from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_mean, scatter_max

from models import neuro_detour

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--g', type=str, default='1')
    args = parser.parse_args()
    device = torch.device('cuda:'+args.g if torch.cuda.is_available() else 'cpu')

    node_sz = 333 # 360
    hiddim = 768
    all_features = []
    all_gts = []
    acc = []
    precision = []
    f1_scores = []
    splits = 10
    num_classes = 8
    input_dim = 39

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # input_dir = f'/ram/USERS/jiaqi/Mainfold_transformer/data/DFYANG_0914/LR_WM'
    input_dir = f'/ram/USERS/ziquanw/data/HCP-YA-SC_FC/Gordon_333_WM_LR'
    label_dir = '/ram/USERS/jiaqi/Mainfold_transformer/data/DFYANG_0914/LR_label'
    train_dataset = SeqFCDataset_WM(input_dir, label_dir, input_dim, skiprows=1)
    # input_dir = f'/ram/USERS/jiaqi/Mainfold_transformer/data/DFYANG_0914/RL_WM'
    input_dir = f'/ram/USERS/ziquanw/data/HCP-YA-SC_FC/Gordon_333_WM_RL'
    label_dir = '/ram/USERS/jiaqi/Mainfold_transformer/data/DFYANG_0914/RL_label'
    test_dataset = SeqFCDataset_WM(input_dir, label_dir, input_dim, skiprows=1)
    # val_dataset, test_dataset = random_split(test_dataset, [3459, 5189])
    val_dataset, test_dataset = random_split(test_dataset, [int(len(test_dataset)*0.4), len(test_dataset) - int(len(test_dataset)*0.4)])

    # kf = KFold(n_splits=splits, shuffle=False, random_state=None)
    # for i, (train_index, test_index) in enumerate(kf.split(ANDI_dataset)):
    #     print(f"Fold {i}:")

    batchsize = 16
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=False)

    # model = TF_mix(in_feas=input_dim, d_model=1024, n_heads=4, d_ff=1024 * 4, num_layers=6, num_classes=num_classes, device=device).to(device)
    # model = TF_fMRI(in_feas=140, d_model=512, n_heads=2, d_ff=512* 4, num_layers=4, num_classes=num_classes, device=device).to(device)
    # model = TF(in_feas=input_dim, d_model=1024, n_heads=4, d_ff=1024 * 4, num_layers=4, num_classes=num_classes, device=device).to(device)
    # model = CNN_1D(nrois=input_dim, f1=1024, f2=360, dilation_exponential=2, k1=3, dropout=0.2, readout='mean', num_classes=num_classes).to(device)
    # model = MLP_Mixer(input_dim=input_dim, length=39, tokens_mlp_dim=1024, channels_mlp_dim=1024, num_classes=num_classes, num_blocks=4).to(device)
    # model = LSTMClassifier(input_size=input_dim, hidden_size=1024, num_layers=2, num_classes=num_classes, device=device).to(device)
    # model = RNNClassifier(input_size=input_dim, hidden_size=1024, num_layers=4, num_classes=num_classes, device=device).to(device)
    # model = MLP(nrois=input_dim, f1=1024, f2=360, num_classes=num_classes, device=device).to(device)
    model = Model(
        neuro_detour.DetourTransformer(node_sz=node_sz, out_channel=hiddim, in_channel=input_dim, batch_size=batchsize, device=device, nlayer=1, heads=8, dropout=0.1), 
        feat_dim=hiddim,
        nclass=num_classes,
        node_sz=node_sz
    ).to(device)
    print("parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)


    best_acc = 0
    best_pre = 0
    best_rec = 0
    best_f1 = 0
    val_best_f1 = 0
    for epoch in range(200):
        # scheduler.step()
        loss = train(model, train_loader, criterion, optimizer, device)
        with torch.no_grad():
            val_acc, pre, rec, val_f1, fold_feature, fold_gts = test(model, val_loader, device)
        if val_f1 >= val_best_f1:
            val_best_f1 = val_f1
            with torch.no_grad():
                test_acc, pre, rec, f1, fold_feature, fold_gts = test(model, test_loader, device)
            best_f1 = f1
            best_acc = test_acc
            best_pre = pre
            best_rec = rec
            
        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     best_pre = pre
        #     best_f1 = f1
            # best_feas = fold_feature
            # best_gts = fold_gts

        print(f'Epoch: {epoch:03d}, Val Acc: {val_acc:.4f}, f1: {val_f1:.4f}, Test Acc: {best_acc:.4f}, pre: {best_pre:.4f}, rec: {best_rec:.4f}, f1: {best_f1:.4f}, Train Loss: {loss:.4f}')
    
    acc.append(best_acc)
    precision.append(best_pre)
    f1_scores.append(best_f1)
    # all_features.extend(best_feas)
    # all_gts.extend(best_gts)
    # print(f"Fold {i}:", best_acc, best_pre, best_f1)


    print(len(all_features), len(all_gts))


    # torch.save(all_features, '/ram/USERS/jiaqi/benchmark_fmri/attn_features/feas_vis/'+model_name+'.pt')
    # np.savetxt('/ram/USERS/jiaqi/benchmark_fmri/attn_features/feas_vis/'+model_name+'label.txt', all_gts, fmt='%d')

    # print("region:", r, "Total_acc:", total_acc, "Total_pre:", total_pre, "Total_f1:", total_f1)   
    print("parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))     
    print("ACC: ", acc)
    print("precision: ", precision)
    print("F1: ", f1_scores)

                

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    losses = []
    # bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, data in enumerate(train_loader):
        # data = Data.from_dict(data)
        target = data.y.to(device)
        logits = model(data.to(device)) 
        loss = criterion(logits, target.to(device)) 
        if hasattr(model, 'loss'):
            loss = loss + model.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # print(loss.item())

    return sum(losses)/len(losses)

def test(model, loader, device):
    model.eval()
    correct = 0
    fold_feature = []
    all_target = []
    all_test = []
    preds = []
    gts = []
    # bar = tqdm(enumerate(loader), total=len(loader))
    for batch_idx, data in enumerate(loader):
        target = data.y.to(device)
        out = model(data.to(device))
        # fold_feature.extend(feas.detach().cpu().numpy())
        pred = out.argmax(dim=-1) 
        
        correct += int((pred == target.to(device)).sum())
        preds.append(pred.cpu().numpy())
        gts.append(target.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)

    accuracy = accuracy_score(gts, preds)
    pre = precision_score(gts, preds, average='weighted')
    rec = recall_score(gts, preds, average='weighted')
    f1 = f1_score(gts, preds, average='weighted')
    
    return accuracy, pre, rec, f1, fold_feature, gts
        
class Model(nn.Module):

    def __init__(self, backbone, feat_dim, nclass=8, node_sz=360) -> None:
        super().__init__()
        self.backbone = backbone
        self.calssifier = Classifier(nn.Linear, feat_dim, nclass, node_sz)
        # GCNConv
    
    def forward(self, data):
        x = self.backbone(data)
        return self.calssifier(x, data.edge_index, data.batch)

class Classifier(nn.Module):

    def __init__(self, net: callable, feat_dim, nclass, node_sz, aggr='learn', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.ModuleList([
            net(feat_dim, feat_dim),
            nn.LeakyReLU(),
            net(feat_dim, nclass)
        ])
        if isinstance(self.net[0], MessagePassing):
            self.nettype = 'gnn'
        else:
            self.nettype = 'mlp'
        self.aggr = aggr
        if aggr == 'learn':
            self.pool = nn.Sequential(nn.Linear(node_sz, 1), nn.LeakyReLU())
        elif aggr == 'mean':
            self.pool = scatter_mean
        elif aggr == 'max':
            self.pool = scatter_max
        # self.head = net(feat_dim, nclass)
    
    def forward(self, x, edge_index, batch):
        if self.nettype == 'gnn':
            x = self.net[0](x, edge_index)
            x = self.net[1](x)
            x = self.net[2](x, edge_index)
        else:
            x = self.net[0](x)
            x = self.net[1](x)
            x = self.net[2](x)
    
        if self.aggr == 'learn':
            x = self.pool(x.view(batch.max()+1, len(torch.where(batch==0)[0]), x.shape[-1]).transpose(-1, -2))[..., 0]
        else:
            if self.aggr == 'max': 
                x = x.view(batch.max()+1, len(torch.where(batch==0)[0]), x.shape[-1]).transpose(-1, -2).max(-1)[0]
            else:
                x = self.pool(x, batch, dim=0)
        return x
    
class SeqFCDataset_WM(Dataset):

    def __init__(self, data_dir, label_dir, input_dim, skiprows=0):
        
        super(SeqFCDataset_WM, self).__init__()
        self.input_dim = input_dim
        self.base_path = data_dir
        self.data_path = [os.path.join(self.base_path, name) for name in
                          sorted_aphanumeric(os.listdir(data_dir)) if name.endswith('.txt') or name.endswith('.csv') or name.endswith('.tsv')]
        
        sc_root = '/ram/USERS/ziquanw/data/HCP-YA-SC_FC/HCP-YA-SC_yuyu'
        subn_p = 0
        subs = [fn.split('/')[-1].split('_')[subn_p] for fn in self.data_path]
        self.sc_common_rname = None
        all_sc = {}
        for fn in tqdm(os.listdir(sc_root), desc='Load SC'):
            subn = fn
            if subn in subs:
                sc, rnames, _ = load_sc(f"{sc_root}/{fn}", 'Gordon_333')
                if self.sc_common_rname is None: self.sc_common_rname = rnames
                if self.sc_common_rname is not None: 
                    _, rid, _ = np.intersect1d(rnames, self.sc_common_rname, return_indices=True)
                    all_sc[subn] = sc[rid, :][:, rid].clone()
                else:
                    all_sc[subn] = sc
        self.subs = []
        self.data = []
        self.fc_common_rname = None
        for path, subn in zip(self.data_path, subs):
            if subn not in all_sc: continue
            self.subs.append(subn)
            x = np.loadtxt(path, delimiter='\t', dtype=str, skiprows=skiprows)
            x[x=='n/a'] = 0
            x = x.astype(np.float32)
            if skiprows > 0:
                rnames = np.loadtxt(path, delimiter='\t', dtype=str)[0]
            else:
                rnames = None
            if self.fc_common_rname is None: self.fc_common_rname = rnames
            if self.fc_common_rname is not None: 
                _, rid, _ = np.intersect1d(rnames, self.fc_common_rname, return_indices=True)
                x = x[:, rid]
            self.data.append(x)
        self.label_path = label_dir
        self.labels_path = [os.path.join(self.label_path, name) for name in
                          sorted_aphanumeric(os.listdir(label_dir)) if name.endswith('.txt') or name.endswith('.csv')]
        # self.labels = [np.loadtxt(path, delimiter=',', dtype=np.float32) for path in self.labels_path] #1
        # self.labels = self.labels * len(self.data)
        self.labels = []
        
        labels = np.loadtxt(self.labels_path[0], delimiter=',', dtype=np.float32)
        for data in self.data:
            self.labels.append(labels[:data.shape[0]])
        
        self.temp_label = []
        self.temp_clip = []
        self.sc = []
        for sample, label, subn in zip(self.data, self.labels, self.subs):
            label_pos = self.pick_labels(label)
            if label_pos is None: continue
            batch_data = sample.squeeze() 
            for ll in label_pos:
                if self.sc_common_rname is not None and self.fc_common_rname is not None:
                    self.sc_common_rname = [rn.strip() for rn in self.sc_common_rname]
                    self.fc_common_rname = [rn.strip() for rn in self.fc_common_rname]
                    common_rname, sc_rid, fc_rid = np.intersect1d(self.sc_common_rname, self.fc_common_rname, return_indices=True)
                    bold = torch.from_numpy(batch_data[label == ll][:, fc_rid].copy()).clone()
                    sc = all_sc[subn][:, sc_rid][sc_rid, :].clone()
                else:
                    bold = torch.from_numpy(batch_data[label == ll]).clone()
                    sc = all_sc[subn].clone()

                if bold.shape[0] < 10: continue
                self.temp_clip.append(bold)
                self.temp_label.append(int(ll-1)) #3
                self.sc.append(sc)

        self.sc_common_rname = common_rname
        self.fc_common_rname = common_rname
        self.data = self.temp_clip
        self.labels = self.temp_label

        self.bold = []
        self.adj = []
        self.edge_index = []
        default_fc_th = 0.5
        default_fc_perc = 10
        for idx in trange(len(self.data), desc='Preload'):
            data = self.data[idx]            
            bold = torch.tensor(data.T).squeeze()
            adj = torch.corrcoef(bold)
            adj[adj.isnan()] = 0
            adj[torch.arange(adj.shape[0]), torch.arange(adj.shape[0])] = 0
            # adj = adj > default_fc_th
            adj = preprocess_adjacency_matrix(adj, default_fc_perc)
            edge_index = torch.stack(torch.where(adj))
            adj[torch.arange(adj.shape[0]), torch.arange(adj.shape[0])] = True
            if bold.shape[1] != self.input_dim:
                bold = torch.concat([bold, torch.zeros(bold.shape[0], self.input_dim-bold.shape[1])], 1)
            assert not bold.isnan().any() 
            assert not adj.isnan().any() 
            assert not self.sc[idx].isnan().any()
            self.bold.append(bold)
            self.adj.append(adj)
            self.edge_index.append(edge_index)

        self.n_data = len(self.data)
        print(self.n_data)
        print(data_dir)
        print(f'{self.data_path[0]},...,{self.data_path[-1]}')

    def __len__(self):
        return self.n_data

    def __add__(self, other):
        self.data_path += other.data_path
        self.labels = np.concatenate((self.labels, other.labels), axis=0)
        self.n_data += other.n_data
        return self

    def __getitem__(self, idx):
        bold = self.bold[idx]
        adj = self.adj[idx]
        edge_index = self.edge_index[idx]
        default_sc_th = 0.1
        adj_sc = self.sc[idx] > default_sc_th

        label = torch.tensor(self.labels[idx]).long()
        data = Data.from_dict({
            'x': bold,
            'edge_index': edge_index,
            'adj_fc': adj[None],
            'adj_sc': adj_sc[None],
            'y': label
        })
        # print(data)
        # data = {
        #     'x': bold,
        #     'edge_index': edge_index,
        #     'adj_fc': adj[None],
        #     'adj_sc': adj_sc[None],
        #     'y': label
        # }
        return data
        # return data, label

    def pick_labels(self, tensor):
        arr = np.unique(tensor)
        mask = np.ones(len(arr), dtype=bool)
        mask[0] = False  # resting #4
        result = arr[mask]
        # assert len(result) == 8
        if len(result) != 8: return None
        return result
  
def load_sc(path, atlas_name):
    if not path.endswith('.mat') and not path.endswith('.txt'):
        matfns = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.mat')]
        txtfns = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]
        return load_sc(matfns[0] if len(matfns) > 0 else txtfns[0], atlas_name)
    if path.endswith('.mat'):
        fpath = f"{path}"
        sc_mat = loadmat(fpath)
        mat = sc_mat[f"{atlas_name.lower().replace('_','')}_sift_radius2_count_connectivity"]
        mat = torch.from_numpy(mat.astype(np.float32))
        mat = (mat + mat.T) / 2
        mat = (mat - mat.min()) / (mat.max() - mat.min())
        rnames = sc_mat[f"{atlas_name.lower().replace('_','')}_region_labels"]
    elif path.endswith('.txt'):
        fpath = f"{path}"
        mat = np.loadtxt(fpath)
        mat = torch.from_numpy(mat.astype(np.float32))
        mat = (mat + mat.T) / 2
        mat = (mat - mat.min()) / (mat.max() - mat.min())
        rnames = None
    return mat, rnames, path.split('/')[-1]
  
def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def preprocess_adjacency_matrix(adjacency_matrix, percent):
    top_percent = np.percentile(adjacency_matrix.flatten(), 100-percent)
    adjacency_matrix[adjacency_matrix < top_percent] = 0
    adjacency_matrix = adjacency_matrix != 0
    return adjacency_matrix

if __name__ == '__main__': main()