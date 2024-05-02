import os, torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm, trange
from statannotations.Annotator import Annotator
from scipy.stats import ttest_rel, ttest_ind
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch.nn.utils.rnn import pad_sequence


ATLAS_FACTORY = ['AAL_116', 'Aicha_384', 'Gordon_333', 'Brainnetome_264', 'Shaefer_100', 'Shaefer_200', 'Shaefer_400', 'D_160']
BOLD_FORMAT = ['.csv', '.csv', '.tsv', '.csv', '.tsv', '.tsv', '.tsv']
DATAROOT = {
    'adni': '/ram/USERS/ziquanw/detour_hcp/data',
    'oasis': '/ram/USERS/ziquanw/detour_hcp/data',
    'hcpa': '/ram/USERS/bendan/ACMLab_DATA',
    'ukb': '/ram/USERS/ziquanw/data'
}
DATANAME = {
    'adni': 'ADNI_BOLD_SC',
    'oasis': 'OASIS_BOLD_SC',
    'hcpa': 'HCP-A-SC_FC',
    'ukb': 'UKB-SC-FC'
}

LABEL_REMAP = {
    'adni': {'CN': 'CN', 'SMC': 'CN', 'EMCI': 'CN', 'LMCI': 'AD', 'AD': 'AD'},
    'oasis': {'CN': 'CN', 'AD': 'AD'},
}

def dataloader_generator(batch_size=4, num_workers=8, nfold=0, total_fold=5, dataset=None, **kargs):
    kf = KFold(n_splits=total_fold, shuffle=True, random_state=142857)
    if dataset is None:
        dataset = NeuroNetworkDataset(**kargs)
    all_subjects = dataset.data_subj
    train_index, index = list(kf.split(all_subjects))[nfold]
    train_subjects = [all_subjects[i] for i in train_index]
    subjects = [all_subjects[i] for i in index]
    # Filter dataset based on training and validation subjects
    train_data = [di for di, subj in enumerate(dataset.subject) if subj in train_subjects]
    data = [di for di, subj in enumerate(dataset.subject) if subj in subjects]
    print(f'Fold {nfold + 1}, Train {len(train_subjects)} subjects, Val {len(subjects)} subjects, len(train_data)={len(train_data)}, len(data)={len(data)}')
    train_dataset = torch.utils.data.Subset(dataset, train_data)
    valid_dataset = torch.utils.data.Subset(dataset, data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, loader, dataset

class NeuroNetworkDataset(Dataset):

    def __init__(self, atlas_name='AAL_116',
                 dname='hcpa',
                node_attr = 'SC', adj_type = 'FC',
                transform = None,
                fc_winsize = 500,
                fc_winoverlap = 0,
                fc_th = 0.5,
                sc_th = 0.1) -> None:
        
        data_root = DATAROOT[dname]
        data_name = DATANAME[dname]
        self.transform = transform
        self.data_root = f"{data_root}/{data_name}"
        self.fc_winsize = fc_winsize
        self.fc_th = fc_th
        self.sc_th = sc_th
        subn_p = 0
        subtask_p = 1 if dname not in ['adni', 'oasis'] else -1
        # subdir_p = 2
        # bold_format = BOLD_FORMAT[ATLAS_FACTORY.index(atlas_name)]
        # fc_format = '.csv'
        assert atlas_name in ATLAS_FACTORY, atlas_name
        bold_root = f'{self.data_root}/{atlas_name}/BOLD'
        fc_root = f'{self.data_root}/{atlas_name}/FC'
        sc_root = f'{self.data_root}/ALL_SC'
        atlas_name = CORRECT_ATLAS_NAME(atlas_name)
        data_dir = f'{dname}-{atlas_name}-BOLDwin{fc_winsize}'
        os.makedirs(f'data/{data_dir}', exist_ok=True)
        if not os.path.exists(f'data/{data_dir}/raw.pt'):
            fc_subs = [fn.split('_')[subn_p] for fn in os.listdir(fc_root)]
            fc_subs = np.unique(fc_subs)
            sc_subs = os.listdir(sc_root)
            subs = np.intersect1d(fc_subs, sc_subs)
            self.all_sc = {}
            self.all_fc = {}
            self.label_name = []
            self.sc_common_rname = None
            for subn in tqdm(os.listdir(sc_root), desc='Load SC'):
                if subn in subs:
                    sc, rnames, subn = load_sc(f"{sc_root}/{subn}", atlas_name)
                    if self.sc_common_rname is None: self.sc_common_rname = rnames
                    if self.sc_common_rname is not None: 
                        _, rid, _ = np.intersect1d(rnames, self.sc_common_rname, return_indices=True)
                        self.all_sc[subn] = sc[rid, :][:, rid]
                    else:
                        self.all_sc[subn] = sc
            self.fc_common_rname = None
            # compute FC in getitem
            self.data = {'bold': [], 'subject': [], 'label': [], 'winid': []}
            for fn in tqdm(os.listdir(bold_root), desc='Load BOLD'):
                if fn.split('_')[subn_p] in subs:
                    bolds, rnames, fn = bold2fc(f"{bold_root}/{fn}", self.fc_winsize, fc_winoverlap, onlybold=True)
                    subn = fn.split('_')[subn_p]
                    if self.fc_common_rname is None: self.fc_common_rname = rnames
                    if self.fc_common_rname is not None: 
                        _, rid, _ = np.intersect1d(rnames, self.fc_common_rname, return_indices=True)
                        bolds = [b[rid] for b in bolds]
                
                    label = Path(fn).stem.split('_')[subtask_p]
                    if dname in ['adni', 'oasis']:
                        if label not in LABEL_REMAP[dname]: continue
                        label = LABEL_REMAP[dname][label]
                    if label not in self.label_name: self.label_name.append(label)
                    self.data['bold'].extend(bolds) # N x T
                    self.data['subject'].extend([subn for _ in bolds])
                    self.data['label'].extend([self.label_name.index(label) for _ in bolds])
                    self.data['winid'].extend([i for i in range(len(bolds))])

            if self.sc_common_rname is not None and self.fc_common_rname is not None:
                self.sc_common_rname = [rn.strip() for rn in self.sc_common_rname]
                self.fc_common_rname = [rn.strip() for rn in self.fc_common_rname]
                common_rname, sc_rid, fc_rid = np.intersect1d(self.sc_common_rname, self.fc_common_rname, return_indices=True)
                for sub in self.all_sc:
                    self.all_sc[sub] = self.all_sc[sub][:, sc_rid][sc_rid, :]
                for i in range(len(self.data['subject'])):
                    self.data['bold'][i] = self.data['bold'][i][fc_rid]
                self.sc_common_rname = common_rname
                self.fc_common_rname = common_rname
            self.data['all_sc'] = self.all_sc
            torch.save(self.data, f'data/{data_dir}/raw.pt')
        
        self.data = torch.load(f'data/{data_dir}/raw.pt')
        self.all_sc = self.data['all_sc']
        self.adj_type = adj_type
        self.node_attr = node_attr
        self.atlas_name = atlas_name
        self.subject = np.array(self.data['subject'])
        self.data_subj = np.unique(self.subject)
        self.node_num = len(self.data['bold'][0])
        self.cached_data = [None for _ in range(len(self))]
        print("Data num", len(self), "BOLD shape (N x T)", self.data['bold'][0].shape)
        if self.transform is not None:
            processed_fn = f'processed_adj{self.adj_type}x{self.node_attr}_FCth{self.fc_th}SCth{self.sc_th}_{type(self.transform).__name__}{self.transform.k}'.replace('.', '')
            if not os.path.exists(f'data/{data_dir}/{processed_fn}.pt'):
                for _ in tqdm(self, desc='Processing'):
                    pass
                
                torch.save(self.cached_data, f'data/{data_dir}/{processed_fn}.pt')
            self.cached_data = torch.load(f'data/{data_dir}/{processed_fn}.pt')
        
    def __getitem__(self, index):
        if self.cached_data[index] is None:
            subjn = self.subject[index]
            fc = torch.corrcoef(self.data['bold'][index])
            sc = self.all_sc[subjn]
            edge_index_fc = torch.stack(torch.where(fc > self.fc_th))
            edge_index_sc = torch.stack(torch.where(sc > self.sc_th))
            if self.adj_type == 'FC':
                edge_index = edge_index_fc
                # adj = torch.sparse_coo_tensor(indices=edge_index_fc, values=fc[edge_index_fc[0], edge_index_fc[1]], size=(self.node_num, self.node_num))
            else:
                edge_index = edge_index_sc
                # adj = torch.sparse_coo_tensor(indices=edge_index_sc, values=sc[edge_index_sc[0], edge_index_sc[1]], size=(self.node_num, self.node_num))
            if self.node_attr=='FC':
                x = fc
            elif self.node_attr=='BOLD':
                x = self.data['bold'][index]
            elif self.node_attr=='SC':
                x = sc
            elif self.node_attr=='ID':
                x = torch.arange(self.node_num).float()[:, None]
        
            x[x.isnan()] = 0
            x[x.isinf()] = 0
            data = {
                'edge_index': edge_index,
                'x': x,
                'y': self.data['label'][index],
                'edge_index_fc': edge_index_fc,
                'edge_index_sc': edge_index_sc
            }
            if self.transform is not None:
                new_data = self.transform(Data.from_dict(data))
                for key in new_data:
                    data[key] = new_data[key]
            self.cached_data[index] = Data.from_dict(data)
        return self.cached_data[index]

    def __len__(self):
        return len(self.subject)


def load_fc(fpath):
    mat = pd.read_csv(fpath)
    mat = torch.from_numpy(mat[:, 1:].astype(np.float32))
    rnames = mat[:, 0]
    return mat, rnames, fpath.split('/')[-1]

def load_sc(path, atlas_name):
    matfns = [f for f in os.listdir(path) if f.endswith('.mat')]
    txtfns = [f for f in os.listdir(path) if f.endswith('.txt')]
    if len(matfns) > 0:
        fpath = f"{path}/{matfns[0]}"
        sc_mat = loadmat(fpath)
        mat = sc_mat[f"{atlas_name.lower().replace('_','')}_sift_radius2_count_connectivity"]
        mat = torch.from_numpy(mat.astype(np.float32))
        mat = (mat + mat.T) / 2
        mat = (mat - mat.min()) / (mat.max() - mat.min())
        rnames = sc_mat[f"{atlas_name.lower().replace('_','')}_region_labels"]
    elif len(txtfns) > 0:
        fpath = f"{path}/{txtfns[0]}"
        mat = np.loadtxt(fpath)
        mat = torch.from_numpy(mat.astype(np.float32))
        mat = (mat + mat.T) / 2
        mat = (mat - mat.min()) / (mat.max() - mat.min())
        rnames = None
    return mat, rnames, path.split('/')[-1]

def bold2fc(path, winsize, overlap, onlybold=False):
    if not path.endswith('.txt'):
        bold_pd = pd.read_csv(path) if not path.endswith('.tsv') else pd.read_csv(path, sep='\t')
        if isinstance(np.array(bold_pd)[0, 0], str):
            rnames = list(bold_pd.columns[1:])
            bold = torch.from_numpy(np.array(bold_pd)[:, 1:]).float().T
        else:
            rnames = list(bold_pd.columns)
            bold = torch.from_numpy(np.array(bold_pd)).float().T
    else:
        rnames = None
        bold = torch.from_numpy(np.loadtxt(path)).float().T
    # bold = bold[torch.logical_not(bold.isnan().any(dim=1))]
    # rnames = [rnames[i] for i in torch.where(torch.logical_not(bold.isnan().any(dim=1)))[0]]
    # bold = (bold - bold.min()) / (bold.max() - bold.min())
    timelen = bold.shape[1]
    steplen = int(winsize*(1-overlap))
    fc = []
    if onlybold:
        bolds = []
    for tstart in range(0, timelen, steplen):
        b = bold[:, tstart:tstart+winsize]
        if b.shape[1] < winsize: 
            # b = bold[:, -winsize:]
            b = torch.cat([b, torch.zeros([b.shape[0], winsize-b.shape[1]], dtype=b.dtype)], dim=1)
        if onlybold: 
            bolds.append(b)
            continue
        fc.append(torch.corrcoef(b))#.cpu()
    if onlybold:
        return bolds, rnames, path.split('/')[-1]
    fc = torch.stack(fc)
    return fc, rnames, path.split('/')[-1]

def CORRECT_ATLAS_NAME(n):
    if n == 'Brainnetome_264': return 'Brainnetome_246'
    if 'Shaefer_' in n: return n.replace('Shaefer', 'Schaefer')
    return n

def Schaefer_SCname_match_FCname(scn, fcn):
    pass

if __name__ == '__main__':
    tl, vl, ds = dataloader_generator(atlas_name='Gordon_333')
    # for data in tl:
    #     print(data.x.shape)
        