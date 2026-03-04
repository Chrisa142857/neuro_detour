# In[1]:


import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, balanced_accuracy_score
from torchsampler import ImbalancedDatasetSampler
# from models import BECA
# from model_bank import brain_net_transformer, vanilla_model, brain_identity, brain_mass_models
from tqdm import trange, tqdm
import torch.optim as optim
import torch.nn as nn
import torch, math
import argparse, os
import numpy as np
from datetime import datetime
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from torch.utils.data import Dataset, Subset, DataLoader, TensorDataset

# Suppress only this specific warning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

device = 'cuda:1'
epochs = 30
lr = 0.0001
decay = 0
batch_size = 5120
max_patience = 5
saver = 'baseline_MLP-grouped_out'
os.makedirs(saver, exist_ok=True)

# data = pd.read_pickle('/ram/USERS/ziquanw/brain_env_ukb/data/ukb-nimg_icd10_dated.pkl')
data = pd.read_pickle('data/ukb-nimg-dwi_icd10-beh_dated.pkl')


# In[3]:


data = data[~data['FC'].isna()]


# In[4]:

delphi_train_data = np.memmap('../brain_env_ukb/data/delphi_train.bin', dtype=np.uint32, mode='r').reshape(-1, 3)
delphi_val_data = np.memmap('../brain_env_ukb/data/delphi_val.bin', dtype=np.uint32, mode='r').reshape(-1, 3)


# ### MLP

# In[79]:


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


# In[80]:

######### No group ################################################
# vocab_grouped = []
# vocab = []
# for icd in data['ICD10']:
#     if isinstance(icd, list):
#         vocab_grouped.extend([i[:3] for i in icd])
#         vocab.extend(icd)

# vocab_grouped = np.unique(vocab_grouped)
# vocab = np.unique(vocab)
# data['ICD10_grouped'] = data['ICD10'].map(lambda x: [xi[:3] for xi in x] if isinstance(x, list) else np.nan)
######################################################################

######## Group to chapters ###########################################
# remap_coding = pd.read_csv(f"/ram/USERS/ziquanw/data/meta_data/ukb_phenotype_info/data_coding/coding_000019.tsv", sep='\t', encoding_errors='ignore')
# nodeid2coding = {row['node_id']: row['coding'] for _, row in remap_coding.iterrows()}
# rootid = [i for i in remap_coding['parent_id'].unique() if i not in remap_coding['node_id'].tolist()]
# topid = remap_coding[remap_coding['parent_id'].isin(rootid)]['node_id'].unique().tolist()
# topid = [i for i in topid if i != 999999999]
# node2top = {}
# for i in trange(len(remap_coding), desc=f'parse the tree of coding'):
#     if remap_coding.iloc[i]['node_id'] == 999999999: continue
#     if remap_coding.iloc[i]['node_id'] in topid: continue
#     j = i
#     while True:
#         if remap_coding.iloc[j]['parent_id'].item() in topid:
#             node2top[remap_coding.iloc[i]['coding']] = nodeid2coding[remap_coding.iloc[j]['parent_id']].replace(' ', '-')
#             break
#         j = np.where(remap_coding['node_id'] == remap_coding.iloc[j]['parent_id'])[0].item()
# for k in list(node2top.keys()):
#     if isinstance(k, str):
#         try:
#             float(k)
#             node2top[str(int(k))+'.0'] = node2top[k]
#         except:
#             pass
#     else:
#         node2top[str(k)] = node2top[k]
#         node2top[str(k)+'.0'] = node2top[k]

vocab_grouped = []
# vocab = []
for icd in data['ICD10_chapter']:
    if isinstance(icd, list):
        vocab_grouped.extend([node2top[i] for i in icd])
        # vocab.extend(icd)

vocab_grouped = np.unique(vocab_grouped)
# vocab = np.unique(vocab)
data['ICD10_grouped'] = data['ICD10_chapter']
# data['ICD10_grouped'] = data['ICD10'].map(lambda x: [node2top[xi] for xi in x] if isinstance(x, list) else np.nan)
######################################################################

# In[82]:


ukb_nimg_data = pd.read_csv('/ram/USERS/ziquanw/data/meta_data/UKB-nimg_phenotype.tsv', sep='\t', index_col='eid', usecols=['eid'])
fnlist = os.listdir('/ram/USERS/ziquanw/data/UKB-SC-FC/ALL_SC')
fnlist = [fn for fn in fnlist if 'S' not in fn.split('_')[0][4:]]
fnlist = np.array(fnlist)
fneid = np.array([int(fn.split('_')[0][4:]) for fn in fnlist])
dwi_path = ukb_nimg_data.index.to_series().map(lambda x: fnlist[fneid==x].tolist())
dwi_path = dwi_path.map(lambda x: x if len(x) > 0 else np.nan)
dwi_path = dwi_path.dropna()


### 
all_subjects = []
multiclass_labels = []
tasks = []
first_occurs = []
last_occurs = []
xs = []
xages = []
valid_mask = []
nan_idx = data.index[data['ICD10_grouped'].isna()].tolist()
for i, row in tqdm(data.iterrows(), total=len(data)):
    if i not in dwi_path.index: continue
    all_subjects.extend([n.split('_')[0] for n in row['FC_name']])
    tasks.extend([n.split('_')[2] for n in row['FC_name']])
    xs.extend(row['FC'])
    xages.extend(row['FC_age (days)'])
    valid_mask.extend(~np.isnan(row['FC']).any(-1).any(-1))

    if i in nan_idx:
        multiclass_labels.extend([np.zeros(len(vocab_grouped)).tolist() for x in row['FC']])
        first_occurs.extend([np.zeros(len(vocab_grouped)).tolist() for x in row['FC']])
        last_occurs.extend([np.zeros(len(vocab_grouped)).tolist() for x in row['FC']])
    else:
        for x, xage in zip(row['FC'], row['FC_age (days)']):
            has_icd = np.isin(vocab_grouped, row['ICD10_grouped'])
            first_occur = np.zeros(len(vocab_grouped))
            last_occur = np.zeros(len(vocab_grouped))
            for l in np.unique(row['ICD10_grouped']):
                first_occur[vocab_grouped.tolist().index(l)] = int(np.array(row['ICD10_age (days)'])[np.array(row['ICD10_grouped'])==l].min())
                last_occur[vocab_grouped.tolist().index(l)] = int(np.array(row['ICD10_age (days)'])[np.array(row['ICD10_grouped'])==l].max())

            first_occurs.append(first_occur.tolist())
            last_occurs.append(last_occur.tolist())
            multiclass_labels.append(has_icd.astype(int).tolist())

    # if len(xs) >= 100: break
# valid_mask = np.array([~np.isnan(x).any() for x in xs])
valid_mask = np.stack(valid_mask)
xs = np.stack(xs)[valid_mask]
xages = np.stack(xages)[valid_mask]
multiclass_labels = np.array(multiclass_labels)[valid_mask]
first_occurs = np.array(first_occurs)[valid_mask]
last_occurs = np.array(last_occurs)[valid_mask]
all_subjects = np.array(all_subjects)[valid_mask]
tasks = np.array(tasks)[valid_mask]


# In[10]:

print(last_occurs.shape, xs.shape, multiclass_labels.shape)

# In[11]:

import random
def multiclass_train(i, classifier, device, loader, optimizer, epoch):
    # classifier.to(device)
    classifier.train()
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    random.shuffle(train_idx)

    for bi in range(0, len(train_idx), batch_size):
        idx = train_idx[bi : bi+batch_size]
        x, y = xs_tensor[idx], ys_tensor[idx]
        batch = {'x': x.reshape(len(x), -1).float(), 'y': y[:, i]}
        optimizer.zero_grad()
        assert not batch['x'].isnan().any() and not batch['x'].isinf().any(), batch['x']
        y = classifier(batch) 
        loss = loss_fn(y.float(), batch['y'].long())
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())

    return np.mean(losses)

def multiclass_eval(i, classifier, device, loader, return_pred=False):
    # classifier.to(device)
    classifier.eval()
    y_true = []
    y_pred = []
    y_scores = []
    losses = []
    loss_fn = nn.CrossEntropyLoss()
    for bi in range(0, len(val_idx), batch_size):
        idx = val_idx[bi : bi+batch_size]
        x, y = xs_tensor[idx], ys_tensor[idx]
        batch = {'x': x.reshape(len(x), -1).float(), 'y': y[:, i]}
        with torch.no_grad():
            y = classifier(batch) 
            loss = loss_fn(y.float(), batch['y'].long())
        if return_pred:
            # y_scores.append(y[torch.arange(len(y)), batch['y']].detach().cpu())
            y_scores.append(y.detach().cpu())
            y_pred.append(y.argmax(1).detach().cpu())
            y_true.append(batch['y'])
        losses.append(loss.detach().cpu().item())

    if return_pred:
        y_true = torch.cat(y_true, dim = 0).detach().cpu().numpy()
        y_pred = torch.cat(y_pred, dim = 0).detach().cpu().numpy()
        y_scores = torch.cat(y_scores, dim = 0).detach().cpu().numpy()
        return np.mean(losses), y_true, y_pred, y_scores
    else:
        return np.mean(losses)


# In[84]:


train_subjects = [f'sub-{i}' for i in np.unique(delphi_train_data[:, 0])]
val_subjects = [f'sub-{i}' for i in np.unique(delphi_val_data[:, 0])]
train_idx = [di for di, subj in enumerate(all_subjects) if subj in train_subjects]
val_idx = [di for di, subj in enumerate(all_subjects) if subj in val_subjects]


# #### MLP-train
# 

# In[85]:


xs_tensor = torch.from_numpy(xs).float().to(device)
ys_tensor = torch.from_numpy(multiclass_labels).long().to(device)


# In[89]:


from copy import deepcopy
best_losses = [1e+10 for _ in vocab_grouped]
best_models = [None for _ in vocab_grouped]
# for epoch in (pbar := trange(1, epochs+1, desc='Epoch')):
train_losses, val_losses = [], []
best_epoch = 0
patience = 0

for mi in (pbar := trange(len(vocab_grouped))):
    model = MLP(in_channel=xs[0].shape[1]**2, out_channel=2).to(device)
    optimizer = optim.Adam(list(model.parameters()), lr=lr, weight_decay=decay)
    for epoch in range(1, epochs+1):
        train_loss = multiclass_train(mi, model, device, None, optimizer, epoch)
        val_loss = multiclass_eval(mi, model, device, None, return_pred=False)
        if val_loss <= best_losses[mi]: 
            best_losses[mi] = val_loss
            # best_models[mi] = deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            best_models[mi] = deepcopy(model.cpu())
            best_epoch = epoch
            model.to(device)
            patience = 0
        log = f'Train {train_loss:.5f} Val {val_loss:.5f} (Best {best_losses[mi]:.5f}@e{best_epoch:03d})'
        pbar.set_description(log)
        patience += 1
        if patience >= max_patience: break
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    # log = f'{datetime.now()}, Epoch {epoch:03d} Train {np.mean(train_losses)} Val {np.mean(val_losses)}'

    # pbar.set_description(log)

for mi in trange(len(best_models)):
    torch.save({
        'ckpt': best_models[mi],
        'best_loss': best_losses[mi],
        'last_train_loss': train_losses[mi],
        'last_val_loss': val_losses[mi],
    }, f'{saver}/ckpt_{vocab_grouped[mi]}.ckpt')
