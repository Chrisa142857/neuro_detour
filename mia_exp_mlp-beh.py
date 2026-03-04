# conda activate pyg24


import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, balanced_accuracy_score
from torchsampler import ImbalancedDatasetSampler
# from models import BECA
# from model_bank import brain_net_transformer, vanilla_model, brain_identity, brain_mass_models
from tqdm import trange, tqdm
import torch.optim as optim
import torch.nn as nn
import torch, math
import argparse, os, sys
import numpy as np
from datetime import datetime
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from torch.utils.data import Dataset, Subset, DataLoader, TensorDataset

from copy import deepcopy
from sklearn.model_selection import KFold
import random

warnings.filterwarnings('ignore')
no_train = True
device = 'cuda:0'
epochs = 30
lr = 0.00001
decay = 0
batch_size = 5120
max_patience = 5
saver = 'baseline_MLP-beh_out'
os.makedirs(saver, exist_ok=True)

# data = pd.read_pickle('/ram/USERS/ziquanw/brain_env_ukb/data/ukb-nimg_icd10_dated.pkl')
data = pd.read_pickle('data/ukb-nimg-dwi_icd10-beh_dated.pkl')
data = data[~data['FC'].isna()]
ukb_nimg_data = pd.read_csv('/ram/USERS/ziquanw/data/meta_data/UKB-nimg_phenotype.tsv', sep='\t', index_col='eid', usecols=['eid'])
fnlist = os.listdir('/ram/USERS/ziquanw/data/UKB-SC-FC/ALL_SC')
fnlist = [fn for fn in fnlist if 'S' not in fn.split('_')[0][4:]]
fnlist = np.array(fnlist)
fneid = np.array([int(fn.split('_')[0][4:]) for fn in fnlist])
dwi_path = ukb_nimg_data.index.to_series().map(lambda x: fnlist[fneid==x].tolist())
dwi_path = dwi_path.map(lambda x: x if len(x) > 0 else np.nan)
dwi_path = dwi_path.dropna()

kf = KFold(n_splits=5, shuffle=True, random_state=142857)
    
class MLP(nn.Module):
    def __init__(self, in_channel, out_channel, hid_channel=768, **kwargs) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
                nn.Linear(in_channel, hid_channel),
                nn.BatchNorm1d(hid_channel),
                nn.LeakyReLU(),
                nn.Linear(hid_channel, hid_channel),
                nn.BatchNorm1d(hid_channel),
                nn.LeakyReLU(),
                nn.Linear(hid_channel, out_channel)
            )

    def forward(self, batch):
        return self.layers(batch['x'])

def multiclass_train(classifier, device, loader, optimizer, epoch):
    # classifier.to(device)
    classifier.train()
    losses = []
    random.shuffle(train_idx)

    for bi in range(0, len(train_idx), batch_size):
        idx = train_idx[bi : bi+batch_size]
        x, y = xs_tensor[idx], ys_tensor[idx]
        batch = {'x': x.reshape(len(x), -1).float(), 'y': y}
        optimizer.zero_grad()
        assert not batch['x'].isnan().any() and not batch['x'].isinf().any(), batch['x']
        y = classifier(batch) 
        loss = loss_fn(y.float(), batch['y'])
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())

    return np.mean(losses)

def multiclass_eval(classifier, device, loader, return_pred=False):
    # classifier.to(device)
    classifier.eval()
    y_true = []
    y_scores = []
    losses = []
    for bi in range(0, len(val_idx), batch_size):
        idx = val_idx[bi : bi+batch_size]
        x, y = xs_tensor[idx], ys_tensor[idx]
        batch = {'x': x.reshape(len(x), -1).float(), 'y': y}
        with torch.no_grad():
            y = classifier(batch) 
            loss = loss_fn(y.float(), batch['y'])
        if return_pred:
            # y_scores.append(y[torch.arange(len(y)), batch['y']].detach().cpu())
            y_scores.append(y.detach().cpu())
            y_true.append(batch['y'])
        losses.append(loss.detach().cpu().item())

    if return_pred:
        y_true = torch.cat(y_true, dim = 0).detach().cpu().numpy()
        y_scores = torch.cat(y_scores, dim = 0).detach().cpu().numpy()
        return np.mean(losses), y_true, y_scores
    else:
        return np.mean(losses)


# In[3]:
all_tgt_tasks = ['num_mem_acc', 'fluid_intel', 'trail_error', 'puzzle_solved', 'symbol-digit_match', 'broken_letter', 'vocab_level', 'tower_game', 'word_match', 'alcohol_freq', 'smoking', 'sleepless']
all_loss_fn = [nn.MSELoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]
save_out = {tgt_task: [] for tgt_task in all_tgt_tasks}
for tgt_task in all_tgt_tasks:
    loss_fn = all_loss_fn[all_tgt_tasks.index(tgt_task)]
    is_mse = isinstance(loss_fn, nn.MSELoss)
    
    ### 
    all_subjects = []
    labels = []
    tasks = []
    # first_occurs = []
    # last_occurs = []
    xs = []
    xages = []
    valid_mask = []
    # nan_idx = data.index[data['ICD10_grouped'].isna()].tolist()
    for i, row in tqdm(data.iterrows(), total=len(data)):
        if i not in dwi_path.index: continue
        all_subjects.extend([n.split('_')[0] for n in row['FC_name']])
        tasks.extend([n.split('_')[2] for n in row['FC_name']])
        xs.extend(row['FC'])
        xages.extend(row['FC_age (days)'])
        valid_mask.extend(np.logical_and(~np.isnan(row['FC']).any(-1).any(-1), ~np.isnan(row[tgt_task])))
        labels.extend(row[tgt_task])
    
    valid_mask = np.stack(valid_mask)
    xs = np.stack(xs)[valid_mask]
    if len(xs) == 0: continue
    xages = np.stack(xages)[valid_mask]
    labels = np.array(labels)[valid_mask].astype(float)
    all_subjects = np.array(all_subjects)[valid_mask]
    tasks = np.array(tasks)[valid_mask]
    if not is_mse:
        nclass = len(np.unique(labels))
        for i, li in enumerate(np.unique(labels)):
            labels[labels==li] = i
        
    # In[10]:
    
    print(tgt_task, xs.shape, labels.shape, np.histogram(labels.reshape(-1)) if is_mse else [np.unique(labels), np.bincount(labels.astype(int).reshape(-1))])
    
    # In[11]:
    if no_train:
        model_weights = torch.load(f'{saver}/ckpt_{tgt_task}.ckpt', weights_only=False)['ckpt']
    
    # In[84]:
    best_models = []
    best_losses = []
    train_losses = []
    val_losses = []
    for foldi, (train_index, val_index) in enumerate(kf.split(all_subjects)):
        timestamp = datetime.now()
        train_subjects, val_subjects =all_subjects[train_index], all_subjects[val_index]
        train_idx = [di for di, subj in enumerate(all_subjects) if subj in train_subjects]
        val_idx = [di for di, subj in enumerate(all_subjects) if subj in val_subjects]
        xs_tensor = torch.from_numpy(xs).float().to(device)
        ys_tensor = torch.from_numpy(labels).float().to(device)
        if not is_mse: ys_tensor = ys_tensor.long()
        
        best_loss = 1e+10
        best_epoch = 0
        patience = 0
        
        # for mi in (pbar := trange(len(all_tgt_tasks))):
        model = MLP(in_channel=xs[0].shape[1]**2, out_channel=1 if is_mse else nclass).to(device)
        if no_train:
            model.load_state_dict(model_weights[foldi])
            _, y_true, y_scores = multiclass_eval(model, device, None, return_pred=True)
            save_out[tgt_task].append({
                'y_true': y_true, 
                'y_scores': y_scores
            })
            continue
        else:
            optimizer = optim.Adam(list(model.parameters()), lr=lr, weight_decay=decay)
            for epoch in (pbar := trange(1, epochs+1)):
                train_loss = multiclass_train(model, device, None, optimizer, epoch)
                val_loss = multiclass_eval(model, device, None, return_pred=False)
                if val_loss <= best_loss: 
                    best_loss = val_loss
                    # best_model = deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                    best_model = deepcopy(model.cpu().state_dict())
                    best_epoch = epoch
                    model.to(device)
                    patience = 0
                log = f'{timestamp}, Fold {foldi}, Train {train_loss:.5f} Val {val_loss:.5f} (Best {best_loss:.5f}@e{best_epoch:03d})'
                pbar.set_description(log)
                patience += 1
                if patience >= max_patience: break
    
        best_models.append(best_model)
        best_losses.append(best_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    if not no_train:
        torch.save({
            'ckpt': best_models,
            'best_loss': best_losses,
            'last_train_loss': train_losses,
            'last_val_loss': val_losses,
        }, f'{saver}/ckpt_{tgt_task}.ckpt')
if no_train:
    torch.save(save_out, f'mia-exp_eval_out/{saver}.pth')