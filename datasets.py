import os, torch, difflib
from scipy.io import loadmat
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
# import networkx as nx
from tqdm import tqdm, trange
# from statannotations.Annotator import Annotator
# from scipy.stats import ttest_rel, ttest_ind
from torch_geometric.data import Data
# from torch_geometric.utils import remove_self_loops, add_self_loops
# from torch.nn.utils.rnn import pad_sequence


ATLAS_FACTORY = ['AAL_116', 'Aicha_384', 'Gordon_333', 'Brainnetome_264', 'Shaefer_100', 'Shaefer_200', 'Shaefer_400', 'D_160']
BOLD_FORMAT = ['.csv', '.csv', '.tsv', '.csv', '.tsv', '.tsv', '.tsv', '.txt']
DATAROOT = {
    'adni': '/ram/USERS/ziquanw/detour_hcp/data',
    'oasis': '/ram/USERS/ziquanw/detour_hcp/data',
    'hcpa': '/ram/USERS/bendan/ACMLab_DATA',
    'ukb': '/ram/USERS/ziquanw/data',
    'hcpya': '/ram/USERS/ziquanw/data',
}
DATANAME = {
    'adni': 'ADNI_BOLD_SC',
    'oasis': 'OASIS_BOLD_SC',
    'hcpa': 'HCP-A-SC_FC',
    'ukb': 'UKB-SC-FC',
    'hcpya': 'HCP-YA-SC_FC',
}
LABEL_NAME_P = {
    'adni': -1, 'oasis': -1, 
    'hcpa': 1, 'hcpya': 1, 
    'ukb': 2,
}

LABEL_REMAP = {
    'adni': {'CN': 'CN', 'SMC': 'CN', 'EMCI': 'CN', 'LMCI': 'AD', 'AD': 'AD'},
    'oasis': {'CN': 'CN', 'AD': 'AD'},
}

def dataloader_generator(batch_size=4, num_workers=8, nfold=0, total_fold=5, dataset=None, testset='None', **kargs):
    kf = KFold(n_splits=total_fold, shuffle=True, random_state=142857)
    if dataset is None:
        dataset = NeuroNetworkDataset(**kargs)
    if isinstance(testset, str):
        if testset != 'None':
            # if testset != 'oasis' and testset != 'adni' and kargs['dname'] != 'adni' and kargs['dname'] != 'oasis':
            #     del kargs['dname']
            #     testset = NeuroNetworkDataset(dname=testset, **kargs)
            # else:
            del kargs['atlas_name'], kargs['dname']
            atlas_name = {'adni': 'AAL_116', 'oasis': 'D_160', 'ukb': 'Gordon_333', 'hcpa': 'Gordon_333'}
            testset = NeuroNetworkDataset(dname=testset, atlas_name=atlas_name[testset], **kargs)
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
    if testset == 'None':
        return train_loader, loader, dataset
    else:
        _, index = list(kf.split(testset.data_subj))[nfold]
        subjects = [testset.data_subj[i] for i in index]
        data = [di for di, subj in enumerate(testset.subject) if subj in subjects]
        print(f'Fold {nfold + 1}, Test {len(subjects)} subjects, len(test_data)={len(data)}')
        test_dataset = torch.utils.data.Subset(testset, data)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, loader, dataset, test_loader, testset

class NeuroNetworkDataset(Dataset):

    def __init__(self, atlas_name='AAL_116',
                 dname='hcpa',
                node_attr = 'SC', adj_type = 'FC',
                transform = None,
                fc_winsize = 500,
                fc_winoverlap = 0,
                fc_th = 0.5,
                sc_th = 0.1) -> None:
        default_fc_th = 0.5
        default_sc_th = 0.1
        data_root = DATAROOT[dname]
        data_name = DATANAME[dname]
        self.transform = transform
        self.data_root = f"{data_root}/{data_name}"
        self.fc_winsize = fc_winsize
        self.fc_th = fc_th
        self.sc_th = sc_th
        subn_p = 0
        subtask_p = LABEL_NAME_P[dname]
        # subdir_p = 2
        # bold_format = BOLD_FORMAT[ATLAS_FACTORY.index(atlas_name)]
        # fc_format = '.csv'
        assert atlas_name in ATLAS_FACTORY, atlas_name
        bold_root = f'{self.data_root}/{atlas_name}/BOLD'
        fc_root = f'{self.data_root}/{atlas_name}/FC'
        sc_root = f'{self.data_root}/ALL_SC'
        atlas_name = CORRECT_ATLAS_NAME(atlas_name)
        if self.fc_th == default_fc_th and self.sc_th == default_sc_th:
            data_dir = f'{dname}-{atlas_name}-BOLDwin{fc_winsize}'
        else:
            data_dir = f'{dname}-{atlas_name}-BOLDwin{fc_winsize}-FCth{str(self.fc_th).replace('.', '')}SCth{str(self.sc_th).replace('.', '')}'
        os.makedirs(f'data/{data_dir}', exist_ok=True)
        if not os.path.exists(f'data/{data_dir}/raw.pt'):
            fc_subs = [fn.split('_')[subn_p] for fn in os.listdir(fc_root)]
            fc_subs = np.unique(fc_subs)
            sc_subs = [fn.split('_')[subn_p] for fn in os.listdir(sc_root)]
            subs = np.intersect1d(fc_subs, sc_subs)
            self.all_sc = {}
            self.all_fc = {}
            self.label_name = []
            self.sc_common_rname = None
            for fn in tqdm(os.listdir(sc_root), desc='Load SC'):
                subn = fn.split('_')[subn_p]
                if subn in subs:
                    sc, rnames, _ = load_sc(f"{sc_root}/{fn}", atlas_name)
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
            self.data['label_name'] = self.label_name
            torch.save(self.data, f'data/{data_dir}/raw.pt')
        
        self.data = torch.load(f'data/{data_dir}/raw.pt')
        self.all_sc = self.data['all_sc']
        self.adj_type = adj_type
        self.node_attr = node_attr
        self.atlas_name = atlas_name
        self.subject = np.array(self.data['subject'])
        # self.data['label'] = np.array(self.data['label'])
        self.data_subj = np.unique(self.subject)
        self.node_num = len(self.data['bold'][0])
        self.cached_data = [None for _ in range(len(self))]
        self.label_remap = None
        if 'task-rest' in self.data['label_name'] or 'task-REST' in self.data['label_name']:
            restli = [i for i, l in enumerate(self.data['label_name']) if 'rest' in l.lower()]
            assert len(restli) == 1, self.data['label_name']
            restli = restli[0]
            nln = list(self.data['label_name'])
            nln[0] = self.data['label_name'][restli]
            nln[restli] = self.data['label_name'][0]
            self.data['label_name'] = nln
            self.label_remap = {restli: 0, 0: restli}

        print("Data num", len(self), "BOLD shape (N x T)", self.data['bold'][0].shape, "Label name", self.data['label_name'])
        if self.transform is not None:
            processed_fn = f'processed_adj{self.adj_type}x{self.node_attr}_FCth{self.fc_th}SCth{self.sc_th}_{type(self.transform).__name__}{self.transform.k}'.replace('.', '')
            if not os.path.exists(f'data/{data_dir}/{processed_fn}.pt'):
                for _ in tqdm(self, desc='Processing'):
                    pass
                
                torch.save(self.cached_data, f'data/{data_dir}/{processed_fn}.pt')
            self.cached_data = torch.load(f'data/{data_dir}/{processed_fn}.pt')
        for _ in tqdm(self, desc='Preloading'):
            pass
        
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
                # self.cached_data[index] = new_data
                for key in new_data:
                    data[key] = new_data[key]
                    
            adj_fc = torch.zeros(x.shape[0], x.shape[0]).bool()
            adj_fc[edge_index_fc[0], edge_index_fc[1]] = True
            adj_sc = torch.zeros(x.shape[0], x.shape[0]).bool()
            adj_sc[edge_index_sc[0], edge_index_sc[1]] = True
            adj_fc[torch.arange(self.node_num), torch.arange(self.node_num)] = True
            adj_sc[torch.arange(self.node_num), torch.arange(self.node_num)] = True
            data['adj_fc'] = adj_fc[None]
            data['adj_sc'] = adj_sc[None]
            
            self.cached_data[index] = Data.from_dict(data)
        data = self.cached_data[index]
        if self.label_remap is not None:
            if data.y in self.label_remap:
                data.y = self.label_remap[data.y]
        return data

    def __len__(self):
        return len(self.subject)


def load_fc(fpath):
    mat = pd.read_csv(fpath)
    mat = torch.from_numpy(mat[:, 1:].astype(np.float32))
    rnames = mat[:, 0]
    return mat, rnames, fpath.split('/')[-1]

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

def Schaefer_SCname_match_FCname(scns, fcns):
    '''
    TODO: Align Schaefer atlas region name of SC and FC
    '''
    match = []
    def get_overlap(s1, s2):
        s = difflib.SequenceMatcher(None, s1, s2)
        pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
        return s1[pos_a:pos_a+size]
    
    for fcn in fcns:
        fcn = fcn.replace('17Networks_', '')
        fcn_split = fcn.split('_')
        sc_overlap_len = []
        for scn in scns:
            scn_split = scn.split('_')
            if scn_split[0] != fcn_split[0] or scn_split[-1] != fcn_split[-1]:
                continue
            sc_overlap_len.append(sum([len(get_overlap(scn_split[i], fcn_split[i])) for i in range(1, len(scn_split)-1)]))
        match.append()

    return match

def tsne_spdmat(mats):
    tril_ind = torch.tril_indices(mats.shape[1], mats.shape[2])
    X = mats[:, tril_ind[0], tril_ind[1]]
    X_embedded = TSNE(n_components=2, random_state=142857).fit_transform(X.numpy())
    return X_embedded

def ttest_fc(fcs1, fcs2, thr=0.05):
    from scipy import stats
    print(fcs1.shape, fcs2.shape)
    significant_fc = []
    ps = []
    for i in trange(fcs1.shape[1]):
        for j in range(i+1, fcs1.shape[2]):
            a = fcs1[:, i, j].numpy()
            b = fcs2[:, i, j].numpy()
            p = stats.ttest_ind(a, b).pvalue
            if p < thr: 
                significant_fc.append([i, j])
                ps.append(p)
    significant_fc = torch.LongTensor(significant_fc)
    ps = torch.FloatTensor(ps)
    print(significant_fc.shape)
    return significant_fc, ps
    


if __name__ == '__main__':
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns
    import random
    dname = 'oasis'
    # from models.graphormer import ShortestDistance
    # from models.nagphormer import NAGdataTransform
    tl, vl, ds = dataloader_generator(dname=dname, atlas_name='D_160', node_attr='FC', fc_winsize=100)#, transform=NAGdataTransform(), transform=NeuroDetourNode(k=5, node_num=333)
    d = []
    for data in tqdm(ds):
        # print(data.x.shape)
        d.append(data.adj_fc.sum(-1).float().mean())
        # exit()
    print(sum(d)/len(d))
    exit()
    sc_list = []
    fc_list = {}
    for data in tqdm(ds):
        sc_list.append(data.adj_sc[0])
        if data.y not in fc_list: fc_list[data.y] = []
        fc_list[data.y].append(data.adj_fc[0])
    subis = list(range(min([len(fc_list[l]) for l in fc_list])))
    random.shuffle(subis)
    subi = subis[:10]
    sc_list = torch.stack(sc_list).float()
    sc_list[:, torch.arange(sc_list.shape[1]), torch.arange(sc_list.shape[1])] = 0
    # plt.matshow(sc_list.mean(0))
    # plt.colorbar()
    # plt.savefig('sc_avg.png')
    # plt.savefig('sc_avg.svg')
    # plt.close()
    # plt.matshow(sc_list.std(0))
    # plt.colorbar()
    # plt.savefig('sc_std.png')
    # plt.savefig('sc_std.svg')
    # plt.close()
    # new_sc = ((sc_list@sc_list) > 0).float() - sc_list
    # new_sc[new_sc<0] = 0
    # new_sc[:, torch.arange(116), torch.arange(116)] = 0
    # plt.matshow(new_sc.mean(0))
    # plt.colorbar()
    # plt.savefig('sc^2_avg.png')
    # plt.savefig('sc^2_avg.svg')
    # plt.close()
    # plt.matshow(new_sc.std(0))
    # plt.colorbar()
    # plt.savefig('sc^2_std.png')
    # plt.savefig('sc^2_std.svg')
    # plt.close()
    # new_sc = ((sc_list@sc_list@sc_list) > 0).float() - new_sc - sc_list
    # new_sc[new_sc<0] = 0
    # new_sc[:, torch.arange(116), torch.arange(116)] = 0
    # plt.matshow(new_sc.mean(0))
    # plt.colorbar()
    # plt.savefig('sc^3_avg.png')
    # plt.savefig('sc^3_avg.svg')
    # plt.close()
    # plt.matshow(new_sc.std(0))
    # plt.colorbar()
    # plt.savefig('sc^3_std.png')
    # plt.savefig('sc^3_std.svg')
    # plt.close()
    # new_sc = ((sc_list@sc_list@sc_list@sc_list) > 0).float() - new_sc - sc_list
    # new_sc[new_sc<0] = 0
    # new_sc[:, torch.arange(116), torch.arange(116)] = 0
    # plt.matshow(new_sc.mean(0))
    # plt.colorbar()
    # plt.savefig('sc^4_avg.png')
    # plt.savefig('sc^4_avg.svg')
    # plt.close()
    # plt.matshow(new_sc.std(0))
    # plt.colorbar()
    # plt.savefig('sc^4_std.png')
    # plt.savefig('sc^4_std.svg')
    # plt.close()
    # for i in subi:
    #     plt.matshow(sc_list[i])
    #     plt.colorbar()
    #     plt.savefig(f'figs/sc_sub{i}.png')
    #     plt.savefig(f'figs/sc_sub{i}.svg')
    #     plt.close()
    for l in fc_list:
        # if l == 0: continue
        fc_list[l] = torch.stack(fc_list[l]).float()
        fc_list[l][:, torch.arange(sc_list.shape[1]), torch.arange(sc_list.shape[1])] = 0
    
    if dname == 'oasis':
        adj = torch.zeros(148, 148)
        # fc_ind = torch.LongTensor([17,18,37,43,49,91,92,111,117,123])-1 # Subcortical - entorhinal 
        fc_ind = torch.LongTensor([5,19,20,25,42,65,71,79,93,94,99,116,131,139,145])-1 # Occipital - Parietal
        meshx, meshy = torch.meshgrid(fc_ind, fc_ind)
        sig_fc, p = ttest_fc(fc_list[0][:, meshx, meshy], fc_list[1][:, meshx, meshy], thr=0.1)
    elif dname == 'adni':
        adj = torch.zeros(sc_list.shape[1], sc_list.shape[1])
        # fc_ind = torch.LongTensor([71,72,73,74,75,76,77,78,83,84,87,88])-1
        fc_ind = torch.LongTensor([i for i in range(43,71)])-1 # Occipital - Parietal
        meshx, meshy = torch.meshgrid(fc_ind, fc_ind)
        sig_fc, p = ttest_fc(fc_list[0][:,meshx, meshy], fc_list[1][:,meshx, meshy], thr=0.1)
    top10_ind = torch.argsort(p)[:10]
    adj[fc_ind[sig_fc[top10_ind, 0]], fc_ind[sig_fc[top10_ind, 1]]] = 1-p[top10_ind]
    print(torch.where(adj))
    np.savetxt(f'resources/{dname}_significant_fc_top10.edge', adj.numpy())
    exit()
    #     for i in subi:
    #         plt.matshow(fc_list[l][i])
    #         plt.colorbar()
    #         plt.savefig(f'figs/fc_sub{i}_label{l}.png')
    #         plt.savefig(f'figs/fc_sub{i}_label{l}.svg')
    #         plt.close()

    # scs = tsne_spdmat(sc_list)
    # fig, axes = plt.subplots(2, 3, sharey=True, sharex=True)
    # axes = axes.reshape(-1)
    # sns.scatterplot(x=scs[:, 0], y=scs[:, 1], ax=axes[0])
    # sns.displot(x=scs[:, 0], y=scs[:, 1], kind="kde")
    # plt.xlim([])
    # plt.ylim([])
    # plt.savefig(f'figs/sc_tsne.png')
    # plt.savefig(f'figs/sc_tsne.svg')
    # plt.close()
    for li, l in enumerate(fc_list):
        # if l == 0: continue
        # fcs = tsne_spdmat(fc_list[l])
        # sns.scatterplot(x=fcs[:, 0], y=fcs[:, 1], ax=axes[li+1])
        # axes[li+1].set_title(f'label{l}')
        # sns.displot(x=fcs[:, 0], y=fcs[:, 1], kind="kde")
        # plt.savefig(f'figs/fc_label{l}_tsne.png')
        # plt.savefig(f'figs/fc_label{l}_tsne.svg')
        # plt.close()
        # fc_list[l] = torch.nn.functional.interpolate(fc_list[l][None], size=(26,26), mode='bilinear')[0]
        plt.matshow(fc_list[l][:, 50:90, 50:90].mean(0), cmap='jet', vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(f'figs/{dname}fc_label{l}_avg.png')
        plt.savefig(f'figs/{dname}fc_label{l}_avg.svg')
        plt.close()
        plt.matshow(fc_list[l][:, 50:90, 50:90].std(0), cmap='jet', vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(f'figs/{dname}fc_label{l}_std.png')
        plt.savefig(f'figs/{dname}fc_label{l}_std.svg')
        plt.close()

    # plt.savefig(f'figs/scfc_tsne.png')
    # plt.savefig(f'figs/scfc_tsne.svg')
    # plt.close()
