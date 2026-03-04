from datasets import dataloader_generator
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from models import brain_net_transformer, neuro_detour, brain_gnn, brain_identity, bolt, graphormer, nagphormer, vanilla_model
from models import brain_net_transformer
# from models.classifier import Classifier
# from torch_geometric.nn import GCNConv#, GATConv, SAGEConv, SGConv
from tqdm import trange, tqdm
import torch.optim as optim
import torch.nn as nn
import torch, math
import argparse, os
import numpy as np
from datetime import datetime
from torch_geometric.data import Batch, Data
import random
from copy import deepcopy
from sklearn.model_selection import KFold
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
epochs = 30
max_patience = 5
lr = 0.0001
decay = 0
batch_size = 5000
device = 'cuda:2'
saver = 'baseline_BNT-beh_out'
os.makedirs(saver, exist_ok=True)
no_train = False
# Suppress only this specific warning
data = pd.read_pickle('data/ukb-nimg-dwi_icd10-beh_dated.pkl')
data = data[~data['FC'].isna()]
ukb_nimg_data = pd.read_csv('/ram/USERS/ziquanw/data/meta_data/UKB-nimg_phenotype.tsv', sep='\t', index_col='eid', usecols=['eid'])
fnlist = os.listdir('../data/UKB-SC-FC/ALL_SC')
fnlist = [fn for fn in fnlist if 'S' not in fn.split('_')[0][4:]]
fnlist = np.array(fnlist)
fneid = np.array([int(fn.split('_')[0][4:]) for fn in fnlist])
dwi_path = ukb_nimg_data.index.to_series().map(lambda x: fnlist[fneid==x].tolist())
dwi_path = dwi_path.map(lambda x: x if len(x) > 0 else np.nan)
dwi_path = dwi_path.dropna()

kf = KFold(n_splits=5, shuffle=True, random_state=142857)


class GraphNet(nn.Module):
    def __init__(self, in_channel, out_channel, hid_channel=768, **kwargs) -> None:
        super().__init__()
        self.node_sz = 116
        self.lin_in = nn.Linear(in_channel, hid_channel)
        self.net = brain_net_transformer.BrainNetworkTransformer(node_sz=self.node_sz, in_channel=in_channel, out_channel=hid_channel, hidden_size=hid_channel, nlayer=2)
        self.lin_node = nn.Linear(self.node_sz, 1)
        self.lin_out = nn.Linear(hid_channel, out_channel)

    def forward(self, batch):
        x = self.lin_in(batch['x'])
        x = self.net(batch)
        x = torch.stack(x.split(self.node_sz)).permute(0, 2, 1)
        x = self.lin_node(x)[..., 0]
        return self.lin_out(x)


def multiclass_train(classifier, device, loader, optimizer, epoch):
    # classifier.to(device)
    classifier.train()
    losses = []
    batches = train_batches[epoch-1]
    for batch in batches:
        gt = batch['y']
        optimizer.zero_grad()
        assert not batch['x'].isnan().any() and not batch['x'].isinf().any(), batch['x']
        y = classifier(batch) 
        loss = loss_fn(y.float(), gt)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())
        
    return np.mean(losses)

def multiclass_eval(classifier, device, loader, return_pred=False):
    classifier.eval()
    y_true = []
    y_scores = []
    losses = []
    
    for batch in val_batches:
        gt = batch['y']
        with torch.no_grad():
            y = classifier(batch) 
            loss = loss_fn(y.float(), gt)
        if return_pred:
            y_scores.append(y.detach().cpu())
            y_true.append(batch['y'])
        losses.append(loss.detach().cpu().item())

    if return_pred:
        y_true = torch.cat(y_true, dim = 0).detach().cpu().numpy()
        y_scores = torch.cat(y_scores, dim = 0).detach().cpu().numpy()
        return np.mean(losses), y_true, y_scores
    else:
        return np.mean(losses)

all_tgt_tasks = ['num_mem_acc', 'fluid_intel', 'trail_error', 'puzzle_solved', 'symbol-digit_match', 'broken_letter', 'vocab_level', 'tower_game', 'word_match', 'alcohol_freq', 'smoking', 'sleepless']
all_loss_fn = [nn.MSELoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]
save_out = {tgt_task: [] for tgt_task in all_tgt_tasks}
for tgt_task in all_tgt_tasks:
    loss_fn = all_loss_fn[all_tgt_tasks.index(tgt_task)]
    is_mse = isinstance(loss_fn, nn.MSELoss)
    if is_mse: continue
    
    fc_thr = 0.5
    all_subjects = []
    labels = []
    tasks = []
    xs = []
    xedges = []
    xages = []
    valid_mask = []
    # nan_idx = data.index[data['ICD10_grouped'].isna()].tolist()
    for i, row in tqdm(data.iterrows(), total=len(data)):
        if i not in dwi_path.index: continue
        all_subjects.extend([n.split('_')[0] for n in row['FC_name']])
        tasks.extend([n.split('_')[2] for n in row['FC_name']])
        xs.extend(row['FC'])
        xages.extend(row['FC_age (days)'])
        xedges.extend([np.stack(np.where(fc>fc_thr), -1) for fc in row['FC']])
        valid_mask.extend(np.logical_and(~np.isnan(row['FC']).any(-1).any(-1), ~np.isnan(row[tgt_task])))
        labels.extend(row[tgt_task])
    
    valid_mask = np.stack(valid_mask)
    xs = np.stack(xs)[valid_mask]
    if len(xs) == 0: continue
    xages = np.stack(xages)[valid_mask]
    xedges = [xedges[i] for i in np.where(valid_mask)[0]]
    labels = np.array(labels)[valid_mask]
    all_subjects = np.array(all_subjects)[valid_mask]
    tasks = np.array(tasks)[valid_mask]
    if not is_mse:
        nclass = len(np.unique(labels))
        for i, li in enumerate(np.unique(labels)):
            labels[labels==li] = i
    print(tgt_task, xs.shape, labels.shape, np.histogram(labels.reshape(-1)) if is_mse else [np.unique(labels), np.bincount(labels.astype(int).reshape(-1))])
    if no_train:
        model_weights = torch.load(f'{saver}/ckpt_{tgt_task}.ckpt', weights_only=False)['ckpt']
    
    
    best_models = []
    best_losses = []
    train_losses = []
    val_losses = []
    for foldi, (train_index, val_index) in enumerate(kf.split(all_subjects)):
        timestamp = datetime.now()
        train_subjects, val_subjects =all_subjects[train_index], all_subjects[val_index]
        train_idx = [di for di, subj in enumerate(all_subjects) if subj in train_subjects]
        val_idx = [di for di, subj in enumerate(all_subjects) if subj in val_subjects]
        
        print('len(train_subjects), len(val_subjects), len(train_idx), len(val_idx)', len(train_subjects), len(val_subjects), len(train_idx), len(val_idx))
    
        xs_tensor = torch.from_numpy(xs).float().to(device)
        ys_tensor = torch.from_numpy(labels).float().to(device)
        if not is_mse: ys_tensor = ys_tensor.long()
        xyedge_datalist = []
        for i in trange(len(xs), desc='convert to pyg data'):
            xyedge_datalist.append(Data(x=xs_tensor[i], edge_index=torch.from_numpy(xedges[i]).long().to(device).T, y=ys_tensor[i:i+1]))
        
        val_batches = [Batch.from_data_list([xyedge_datalist[idxi] for idxi in val_idx[bi : bi+batch_size]]) for bi in trange(0, len(val_idx), batch_size, desc='prepare val batches')]

        best_loss = 1e+10
        best_epoch = 0
        patience = 0
        model = GraphNet(in_channel=xs[0].shape[1], out_channel=1 if is_mse else nclass).to(device)
        if no_train:
            model.load_state_dict(model_weights[foldi])
            _, y_true, y_scores = multiclass_eval(model, device, None, return_pred=True)
            save_out[tgt_task].append({
                'y_true': y_true, 
                'y_scores': y_scores
            })
            continue
        else:
            train_batches = []
            for _ in trange(epochs, desc='prepare train batches'):
                random.shuffle(train_idx)
                train_batches.append([Batch.from_data_list([xyedge_datalist[idxi] for idxi in train_idx[bi : bi+batch_size]]) for bi in range(0, len(train_idx), batch_size)])

            optimizer = optim.Adam(list(model.parameters()), lr=lr, weight_decay=decay)
            for epoch in (pbar := trange(1, epochs+1)):
                train_loss = multiclass_train(model, device, None, optimizer, epoch)
                val_loss = multiclass_eval(model, device, None, return_pred=False)
                if val_loss <= best_loss: 
                    best_loss = val_loss
                    best_model = deepcopy(model.cpu().state_dict())
                    best_epoch = epoch
                    model.to(device)
                    patience = 0
                log = f'Train {train_loss:.5f} Val {val_loss:.5f} (Best {best_loss:.5f}@e{best_epoch:03d})'
                pbar.set_description(log)
                patience += 1
                if patience >= max_patience: break
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            best_models.append(best_model)
        
    
    if not no_train:
        torch.save({
            'ckpt': best_models,
            'best_loss': best_losses,
            'last_train_loss': train_losses,
            'last_val_loss': val_losses,
        }, f'{saver}/ckpt_{tgt_task}.ckpt')

if no_train:
    torch.save(save_out, f'mia-exp_eval_out/{saver}.pth')