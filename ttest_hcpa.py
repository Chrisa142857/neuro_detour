
import os, torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from multiprocessing import get_context
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm, trange
from statannotations.Annotator import Annotator
from scipy.stats import ttest_rel, ttest_ind
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch.nn.utils.rnn import pad_sequence
import seaborn as sns
import matplotlib.pyplot as plt


ATLAS_FACTORY = ['AAL_116', 'Aicha_384', 'Gordon_333', 'Brainnetome_264', 'Shaefer_100', 'Shaefer_200', 'Shaefer_400']
BOLD_FORMAT = ['.csv', '.csv', '.tsv', '.csv', '.tsv', '.tsv', '.tsv']
THREAD_N = 30
BOXPLOT_ORDER = None


def main():
    # dset = HCPAScFcDataset(ATLAS_FACTORY[2], fc_th=-1, dek=5)
    # dset_aicha384 = HCPAScFcDataset(ATLAS_FACTORY[1], fc_th=-1, dek=5)
    # dset_aal116 = HCPAScFcDataset(ATLAS_FACTORY[0], fc_th=-1, dek=5)
    dset_b264 = HCPAScFcDataset(ATLAS_FACTORY[3], fc_th=-1, dek=5)

def CORRECT_ATLAS_NAME(n):
    if n == 'Brainnetome_264': return 'Brainnetome_246'
    if 'Shaefer_' in n: return n.replace('Shaefer', 'Schaefer')
    return n

def Schaefer_SCname_match_FCname(scn, fcn):
    pass

class HCPAScFcDataset(Dataset):
    data_root = '/ram/USERS/bendan/ACMLab_DATA/HCP-A-SC_FC'
    work_root = '/ram/USERS/ziquanw/detour_hcp'
    def __init__(self, atlas_name,
                node_attr = 'FC', fctype='DynFC',
                direct_filter = [],
                fc_winsize = 100,
                fc_winoverlap = 0.1,
                fc_th = 0.5,
                sc_th = 0.1,
                dek = 5) -> None:
        self.fc_winsize = fc_winsize
        self.fc_th = fc_th
        self.sc_th = sc_th
        self.dek = dek
        subn_p = 0
        subtask_p = 1
        subdir_p = 2
        bold_format = BOLD_FORMAT[ATLAS_FACTORY.index(atlas_name)]
        assert atlas_name in ATLAS_FACTORY, atlas_name
        fc_root = f'{self.data_root}/{atlas_name}/BOLD'
        sc_root = f'{self.data_root}/ALL_SC'
        atlas_name = CORRECT_ATLAS_NAME(atlas_name)
        fc_subs = [fn.split('_')[subn_p] for fn in os.listdir(fc_root) if fn.endswith(bold_format)]
        fc_subs = np.unique(fc_subs)
        sc_subs = os.listdir(sc_root)
        subs = np.intersect1d(fc_subs, sc_subs)
        # print(subs)
        self.all_sc = {}
        region = {}
        sclist = []
        for subn in tqdm(os.listdir(sc_root), desc='Load SC'):
            if subn in subs:
                sclist.append(load_sc(f"{sc_root}/{subn}", atlas_name))
        for mat, rnames, subn in sclist:
            self.all_sc[subn] = mat
            region[subn] = [r.rstrip() for r in rnames]
        self.node_attr = node_attr
        self.atlas_name = atlas_name
        
        self.all_fc = []
        self.fc_task = []
        self.fc_direc = []
        self.fc_subject = []
        self.fc_winind = []
        self.task_name = []
        self.direc_name = []
        self.de = []
        if self.fc_winsize != 100:
            fc_zip_fn = f'{self.work_root}/data/{self.data_root.split("/")[-1]}_{atlas_name}_FC_winsize{self.fc_winsize}.zip'
        else:
            fc_zip_fn = f'{self.work_root}/data/{self.data_root.split("/")[-1]}_{atlas_name}_FC.zip'
        if os.path.exists(fc_zip_fn):
            fclist = torch.load(fc_zip_fn)
        else:
            fclist = [bold2fc(f"{fc_root}/{fn}", self.fc_winsize, fc_winoverlap) for fn in tqdm(os.listdir(fc_root), desc='Load BOLD') if fn.endswith(bold_format) and fn.split('_')[subn_p] in subs]
            # with Pool(2) as p:
            #     fclist = list(p.starmap(bold2fc, tqdm([[f"{fc_root}/{fn}", fc_winsize, fc_winoverlap] for fn in os.listdir(fc_root) if fn.endswith(bold_format) and fn.split('_')[subn_p] in subs], desc='Load BOLD')))
            torch.save(fclist, fc_zip_fn)
        if node_attr == 'BOLD':
            self.bolds = []
            for fn in tqdm(os.listdir(fc_root), desc='Load BOLD'):
                if fn.endswith(bold_format) and fn.split('_')[subn_p] in subs:
                    bolds, rnames, fn = bold2fc(f"{fc_root}/{fn}", self.fc_winsize, fc_winoverlap, onlybold=True)
                    subn = fn.split('_')[subn_p]
                    assert subn in region, subn
                    _, _, fc_ind = np.intersect1d(region[subn], rnames, return_indices=True)
                    self.bolds.extend([b[fc_ind] for b in bolds])
            
        for fc, rnames, fn in fclist:
            subn = fn.split('_')[subn_p]
            task = fn.split('_')[subtask_p]
            direc = fn.split('_')[subdir_p]
            if direc in direct_filter: continue
            assert subn in region, subn
            if task not in self.task_name: self.task_name.append(task)
            if direc not in self.direc_name: self.direc_name.append(direc)
            regions, sc_ind, fc_ind = np.intersect1d(region[subn], rnames, return_indices=True)
            self.all_sc[subn] = self.all_sc[subn][sc_ind, :][:, sc_ind]
            region[subn] = regions
            self.all_fc.extend(list(fc[:, fc_ind, :][:, :, fc_ind]))
            self.fc_winind.extend(torch.arange(len(fc)).tolist())
            self.fc_task.extend([self.task_name.index(task) for _ in range(len(fc))])
            self.fc_direc.extend([self.direc_name.index(direc) for _ in range(len(fc))])
            self.fc_subject.extend([subn for _ in range(len(fc))])
        
        assert len(np.unique([len(v) for v in region.values()])) == 1
        self.regions = list(region.values())[0]
        self.all_fc = torch.stack(self.all_fc)
        self.fc_winind = torch.LongTensor(self.fc_winind)
        self.fc_task = torch.LongTensor(self.fc_task)
        self.fc_direc = torch.LongTensor(self.fc_direc)
        self.fc_subject = np.array(self.fc_subject)
        self.data_subj = np.unique(self.fc_subject)
        fc_de_fn_tail = f'_FCth{fc_th}' if fc_th != 0.5 else ''
        if self.fc_winsize != 100:
            fc_de_fn = f'{self.work_root}/data/{self.data_root.split("/")[-1]}_{atlas_name}_DeN_k{dek}_FCwinsize{self.fc_winsize}{fc_de_fn_tail}.zip'
        else:
            fc_de_fn = f'{self.work_root}/data/{self.data_root.split("/")[-1]}_{atlas_name}_DeN_k{dek}{fc_de_fn_tail}.zip'
        if os.path.exists(fc_de_fn.replace('DeN', 'DirPath')) and os.path.exists(fc_de_fn.replace('DeN', 'DePath')):
            self.de_path = torch.load(fc_de_fn.replace('DeN', 'DePath'))
            self.dir_path = torch.load(fc_de_fn.replace('DeN', 'DirPath'))
        else:
            with get_context("spawn").Pool(THREAD_N) as p:
                self.de_path = []
                self.dir_path = []
                subi = 0
                for de_path in tqdm(p.imap(fc_detour, [[self.all_fc[i]>fc_th, self.all_sc[self.fc_subject[i]]>sc_th, dek] for i in range(len(self.all_fc))], chunksize=10), total=len(self.all_fc), desc='Prepare FC and Detour'):
                    dir_path = torch.stack(torch.where(self.all_fc[subi]>fc_th)).T
                    self.de_path.append(de_path)
                    self.dir_path.append(dir_path)
                    subi += 1
            torch.save(self.de_path, fc_de_fn.replace('DeN', 'DePath'))
            torch.save(self.dir_path, fc_de_fn.replace('DeN', 'DirPath'))
        
        if os.path.exists(fc_de_fn.replace('DeN', 'DeAdj')):
            self.de = torch.load(fc_de_fn.replace('DeN', 'DeAdj'))
        else:
            self.de = []
            # for fc, de_path, dir_path in zip(self.all_fc, self.de_path, self.dir_path):
            for i in trange(len(self.all_fc), desc='Prepare Detour adj'):
                fc, de_path, dir_path = self.all_fc[i], self.de_path[i], self.dir_path[i]
                de = torch.zeros_like(self.all_fc[0]).long()
                de[dir_path[:, 0], dir_path[:, 1]] = torch.LongTensor([len(p) for p in de_path])
                self.de.append(de)
            self.de = torch.stack(self.de)
            torch.save(self.de, fc_de_fn.replace('DeN', 'DeAdj'))
            
        if os.path.exists(fc_de_fn.replace('DeN', 'DeVecAdj')):
            self.de_vec = torch.load(fc_de_fn.replace('DeN', 'DeVecAdj'))
        else:
            de_vec = []
            for i in trange(len(self.all_fc), desc='Prepare Detour vector adj'):
                fc, de_path, dir_path = self.all_fc[i], self.de_path[i], self.dir_path[i]
                de = torch.zeros_like(self.all_fc[0]).long().repeat(self.dek-1, 1, 1)
                plen_ls = [torch.zeros(len(de_path)).long() for _ in range(self.dek-1)]
                for pi, ps in enumerate(de_path):
                    for p in ps:
                        plen_ls[len(p)-3][pi] = len(p)
                for k, plen in enumerate(plen_ls):
                    de[k, dir_path[:, 0], dir_path[:, 1]] = torch.LongTensor(plen)
                de_vec.append(de)
            self.de_vec = torch.stack(de_vec)
            torch.save(self.de_vec, fc_de_fn.replace('DeN', 'DeVecAdj'))
            
        self.fctype = fctype
        self.node_num = len(self.regions)

    def group_boxplot_analysis(self):
        global BOXPLOT_ORDER
        data = {'Task&Direct': [], 'Subject': [], 'Win ID': [], 
                'Ratio (~FC SC / SC)': [], 'Ratio (FC SC / FC)': [], 'Ratio (FC ~SC / FC)': [], 
                'Ratio (~De FC ~SC / FC)': [], 
                # 'Ratio (De FC ~SC / FC)': [], 
                'Ratio (~De FC SC / FC)': [],
                'Number (De FC ~SC)': [], 'Number (De FC SC)': [], 'Number (De)': []
                }
        for task, direc, subj, fc, de, winid in tqdm(zip(self.fc_task, self.fc_direc, self.fc_subject, self.all_fc, self.de, self.fc_winind), total=len(self.fc_task), desc='Group analysis'):
            fc = fc > self.fc_th
            sc = self.all_sc[subj] > self.sc_th
            data['Task&Direct'].append(f'{self.task_name[task]}&{self.direc_name[direc]}')
            data['Subject'].append(subj)
            data['Win ID'].append(winid.item())
            data['Ratio (~FC SC / SC)'].append(((torch.logical_not(fc) & sc).sum()/sc.sum()).item())
            data['Ratio (FC SC / FC)'].append(((fc & sc).sum()/fc.sum()).item())
            data['Ratio (FC ~SC / FC)'].append(((fc & torch.logical_not(sc)).sum()/fc.sum()).item())
            data['Ratio (~De FC ~SC / FC)'].append(((de[fc & torch.logical_not(sc)] == 0).sum()/fc.sum()).item())
            data['Ratio (~De FC SC / FC)'].append(((de[fc & sc] == 0).sum()/fc.sum()).item())
            data['Number (De FC ~SC)'].append(de[fc & torch.logical_not(sc)].sum().item()) 
            data['Number (De FC SC)'].append(de[fc & sc].sum().item())
            data['Number (De)'].append(de.sum().item()) 
        df = pd.DataFrame(data)
        if BOXPLOT_ORDER is None:
            BOXPLOT_ORDER = list(np.unique(data['Task&Direct']))
        pairs=[(BOXPLOT_ORDER[i], BOXPLOT_ORDER[j]) for i in range(len(BOXPLOT_ORDER)) for j in range(i+1, len(BOXPLOT_ORDER))]
        ratio_num = len([key for key in data.keys() if 'Ratio' in key or 'Number' in key])
        fig, axes = plt.subplots(2, ratio_num, figsize=(ratio_num*5, 10), sharex=True)
        axi = 0
        for key in data.keys():
            if 'Ratio' in key or 'Number' in key:
                ax = sns.boxplot(data=df, x='Task&Direct', y=key, ax=axes[0, axi], hue='Win ID', showfliers=False, order=BOXPLOT_ORDER)
                axi += 1
        axi = 0
        for key in data.keys():
            if 'Ratio' in key or 'Number' in key:
                ax = sns.boxplot(data=df, x='Task&Direct', y=key, ax=axes[1, axi], showfliers=False, order=BOXPLOT_ORDER)
                ax.set_xticklabels(BOXPLOT_ORDER, rotation=30)
                annotator = Annotator(ax, pairs, data=df, x='Task&Direct', y=key, order=BOXPLOT_ORDER, verbose=False)
                annotator.configure(test=None, text_format='simple', loc='inside', hide_non_significant=True, show_test_name=True)
                pvalues = []
                for pair in pairs:
                    paired_a, paired_b = get_paired_data_df(df, key, pair)
                    pvalues.append(ttest_rel(paired_a, paired_b).pvalue)
                annotator.set_pvalues_and_annotate(pvalues=pvalues)
                axi += 1
        plt.tight_layout()
        plt.savefig(f'HCP-A_Dek{self.dek}-FC_ws{self.fc_winsize}-SC_{self.atlas_name}_group_boxplot.png')
        plt.close()

    def group_avg_analysis(self):
        dir_uni = self.fc_direc.unique()
        task_uni = self.fc_task.unique()
        fig, axes = plt.subplots(len(dir_uni), len(task_uni), figsize=(len(task_uni)*6, len(dir_uni)*5))
        empty_ax = [[i,j] for i in range(len(dir_uni)) for j in range(len(task_uni))]
        for di, direc in enumerate(dir_uni):
            for ti, task in enumerate(task_uni):
                f1 = self.fc_direc==direc
                f2 = self.fc_task==task
                f = f1&f2
                if f.any():
                    hmap = self.de[f].float().mean(0)
                    sns.heatmap(hmap.numpy(), ax=axes[di,ti])
                    axes[di,ti].set_title(f'{self.task_name[task]}-{self.direc_name[direc]}')
                    empty_ax.remove([di,ti])
        if len(empty_ax)>0:
            i,j = empty_ax[0]
            for ri, rn in enumerate(self.regions):
                rowi = ri % 50 + 1
                coli = ri//50
                axes[i,j].text((len(self.regions)/50)*coli, 0.02*rowi, rn, horizontalalignment='center', verticalalignment='center', transform=axes[i,j].transAxes)
        plt.tight_layout()
        plt.savefig(f'HCP-A_De_k{self.dek}-FC_ws{self.fc_winsize}_{self.atlas_name}_group_avg.png')
        plt.close()
        
        fig, axes = plt.subplots(len(dir_uni), len(task_uni), figsize=(len(task_uni)*6, len(dir_uni)*5))
        for di, direc in enumerate(dir_uni):
            for ti, task in enumerate(task_uni):
                f1 = self.fc_direc==direc
                f2 = self.fc_task==task
                f = f1&f2
                if f.any():
                    hmap = torch.stack([self.all_sc[subj] for subj in self.fc_subject[f]]).float().mean(0)
                    sns.heatmap(hmap.numpy(), ax=axes[di,ti])
                    axes[di,ti].set_title(f'{self.task_name[task]}-{self.direc_name[direc]}')
        plt.tight_layout()
        plt.savefig(f'HCP-A_SC_{self.atlas_name}_group_avg.png')
        plt.close()
        fig, axes = plt.subplots(len(dir_uni), len(task_uni), figsize=(len(task_uni)*6, len(dir_uni)*5))
        for di, direc in enumerate(dir_uni):
            for ti, task in enumerate(task_uni):
                f1 = self.fc_direc==direc
                f2 = self.fc_task==task
                f = f1&f2
                if f.any():
                    hmap = torch.stack([self.all_sc[subj] for subj in self.fc_subject[f]]).float().std(0)
                    sns.heatmap(hmap.numpy(), ax=axes[di,ti])
                    axes[di,ti].set_title(f'{self.task_name[task]}-{self.direc_name[direc]}')
        plt.tight_layout()
        plt.savefig(f'HCP-A_SC_{self.atlas_name}_group_std.png')
        plt.close()
        
        fig, axes = plt.subplots(len(dir_uni), len(task_uni), figsize=(len(task_uni)*6, len(dir_uni)*5))
        for di, direc in enumerate(dir_uni):
            for ti, task in enumerate(task_uni):
                f1 = self.fc_direc==direc
                f2 = self.fc_task==task
                f = f1&f2
                if f.any():
                    hmap = self.all_fc[f].float().mean(0)
                    sns.heatmap(hmap.numpy(), ax=axes[di,ti])
                    axes[di,ti].set_title(f'{self.task_name[task]}-{self.direc_name[direc]}')
        plt.tight_layout()
        plt.savefig(f'HCP-A_FC_ws{self.fc_winsize}_{self.atlas_name}_group_avg.png')
        plt.close()
        fig, axes = plt.subplots(len(dir_uni), len(task_uni), figsize=(len(task_uni)*6, len(dir_uni)*5))
        for di, direc in enumerate(dir_uni):
            for ti, task in enumerate(task_uni):
                f1 = self.fc_direc==direc
                f2 = self.fc_task==task
                f = f1&f2
                if f.any():
                    hmap = self.all_fc[f].float().std(0)
                    sns.heatmap(hmap.numpy(), ax=axes[di,ti])
                    axes[di,ti].set_title(f'{self.task_name[task]}-{self.direc_name[direc]}')
        plt.tight_layout()
        plt.savefig(f'HCP-A_FC_{self.atlas_name}_group_std.png')
        plt.close()

    def group_edge_significance_analysis(self):
        task_uni = self.fc_task.unique()
        groups = []
        group_ids = []
        for task in task_uni:
            group_de = self.de[self.fc_task==task]
            group_sc = torch.stack([self.all_sc[subj] for subj in self.fc_subject[self.fc_task==task]])
            group_fc = self.all_fc[self.fc_task==task]
            group_de[group_de.isnan()] = 0
            group_sc[group_sc.isnan()] = 0
            group_fc[group_fc.isnan()] = 0
            group_winid = self.fc_winind[self.fc_task==task]
            group_subj = self.fc_subject[self.fc_task==task]
            group_ids.append([group_winid, group_subj])
            groups.append([group_de,group_sc,group_fc])
        node_num = self.node_num
        fig, axes = plt.subplots(1, 3, figsize=(3*10, 10))
        titles = ['DE', 'SC', 'FC']
        for i in range(3):
            sig_mat = torch.zeros(len(groups)*node_num, len(groups)*node_num)
            for gi in trange(len(groups), desc='Group Analysis'):
                for gj in range(len(groups)):
                    if gi == gj: continue
                    paired_a, paired_b = get_paired_data(groups[gi][i], groups[gj][i], group_ids[gi][0], group_ids[gj][0], group_ids[gi][1], group_ids[gj][1])
                    for mi in range(node_num):
                        for mj in range(node_num):
                            sig_mat[gi*node_num+mi, gj*node_num+mj] = ttest_rel(paired_a[:, mi, mj], paired_b[:, mi, mj]).pvalue
            sns.heatmap(sig_mat.numpy(), ax=axes[i])
            axes[i].set_title(titles[i])
            axes[i].set_xticks([node_num/2 + gi*node_num for gi in range(len(groups))], labels=[self.task_name[task_uni[gi]] for gi in range(len(groups))])
            axes[i].set_yticks([node_num/2 + gi*node_num for gi in range(len(groups))], labels=[self.task_name[task_uni[gi]] for gi in range(len(groups))])
        plt.tight_layout()
        plt.savefig(f'HCP-A_ROI_dek{self.dek}-{self.fctype}_{self.atlas_name}_group_ttest.png')
        plt.close()

    def group_node_significance_analysis(self, subj_filter=None, savetag=''):
        task_uni = self.fc_task.unique()
        self.fc_task = self.fc_task.numpy()
        groups = []
        group_ids = []
        if subj_filter is None:
            subj_filter = list(self.fc_subject)
        
        for task in task_uni:
            group_mask = (self.fc_task==task) & (np.array([subj in subj_filter for subj in self.fc_subject]))
            print(subj_filter[:10], self.fc_subject[:10])
            group_mask = np.where(group_mask)[0].tolist()
            print(len(group_mask))
            group_de = self.de[group_mask]
            group_sc = torch.stack([self.all_sc[subj] for subj in self.fc_subject[group_mask]])
            group_fc = self.all_fc[group_mask]
            group_de[group_de.isnan()] = 0
            group_sc[group_sc.isnan()] = 0
            group_fc[group_fc.isnan()] = 0
            group_winid = self.fc_winind[group_mask]
            group_subj = self.fc_subject[group_mask]
            group_ids.append([group_winid, group_subj])
            groups.append([group_de.sum(-1),group_sc.sum(-1),group_fc.mean(-1)])
        fig, axes = plt.subplots(1, 3, figsize=(3*10, 10))
        titles = ['DE', 'SC', 'FC']
        for i in range(3):
            sig_mat = torch.zeros(len(groups)**2, self.node_num)
            for gi in trange(len(groups), desc='Group Analysis'):
                for gj in range(len(groups)):
                    print(groups[gi][i].shape, group_ids[gi][0].shape, group_ids[gi][1].shape)
                    sig = torch.zeros(self.node_num)
                    if gi != gj:
                        paired_a, paired_b = get_paired_data(groups[gi][i], groups[gj][i], group_ids[gi][0], group_ids[gj][0], group_ids[gi][1], group_ids[gj][1])
                    else:
                        paired_a, paired_b = groups[gi][i], groups[gj][i]
                    for mi in range(self.node_num):
                        # pvalues = ttest_rel(paired_a[:, mi], paired_b[:, mi]).pvalue
                        pvalues = ttest_ind(paired_a[:, mi], paired_b[:, mi]).pvalue
                        sig_mat[gi*len(groups) + gj, mi] = pvalues
                        sig[mi] = pvalues
                    out = {
                        'significance': sig,
                        'regions': self.regions
                    }
                    if i == 0:
                        torch.save(out, f'HCP-A_DeN_{titles[i]}_dek{self.dek}-{self.fctype}_{self.atlas_name}_{self.task_name[task_uni[gi]]}-{self.task_name[task_uni[gj]]}{savetag}_ttest.pth')
                    else:
                        torch.save(out, f'HCP-A_DeN_{titles[i]}-{self.fctype}_{self.atlas_name}_{self.task_name[task_uni[gi]]}-{self.task_name[task_uni[gj]]}{savetag}_ttest.pth')
            sns.heatmap(sig_mat.numpy(), ax=axes[i])
            axes[i].set_title(titles[i])
            axes[i].set_yticks([gi for gi in range(len(groups)**2)])
            axes[i].set_xticks([ni for ni in range(self.node_num)])
            axes[i].set_xticklabels(self.regions, rotation = 90)
            axes[i].set_yticklabels([f'{self.task_name[task_uni[gi]]}-{self.task_name[task_uni[gj]]}' for gi in range(len(groups)) for gj in range(len(groups))], rotation = 0)
        plt.tight_layout()
        plt.savefig(f'HCP-A_DeN_dek{self.dek}-{self.fctype}_{self.atlas_name}{savetag}_group_ttest.png')
        plt.close()

    def __getitem__(self, index):
        # subjn = self.data_subj[index]
        subjn = self.fc_subject[index]
        fc = self.all_fc[index]
        de = self.de[index]
        # de_path = self.de_path[index]
        fc_edge_index = self.dir_path[index].long().T
        
        if self.node_attr=='FC':
            x = fc
        elif self.node_attr=='BOLD':
            x = self.bolds[index]
        elif self.node_attr=='SC':
            x = self.all_sc[subjn]
        elif self.node_attr=='ID':
            x = torch.arange(fc.shape[0]).float()[:, None]
        elif self.node_attr=='DE':
            x = de
        elif self.node_attr=='FC+DE':
            x = torch.cat([fc,de], dim=1) 
        data = Data(x=x, edge_index=fc_edge_index)

        return {
            'data':data,
            'subject':subjn,
            'label':self.fc_task[index]
        }

    def __len__(self):
        return len(self.all_fc)

def get_paired_data(a, b, a_winid, b_winid, a_subj, b_subj):
    subjs = np.intersect1d(a_subj, b_subj)
    paired_a, paired_b = [], []
    for subj in subjs:
        f1 = a_subj==subj
        f2 = b_subj==subj
        a_max = a_winid[f1].max()
        b_max = b_winid[f2].max()
        winid_max = min(a_max, b_max)
        a_w, a_ind = np.unique(a_winid[f1], return_index=True)
        b_w, b_ind = np.unique(b_winid[f2], return_index=True)
        a_ind = a_ind[:winid_max+1]
        b_ind = b_ind[:winid_max+1]
        paired_a.append(a[np.where(f1)[0][a_ind]])
        paired_b.append(b[np.where(f2)[0][b_ind]])
    paired_a = np.concatenate(paired_a)
    paired_b = np.concatenate(paired_b)
    return paired_a, paired_b

def get_paired_data_df(df, datakey, pairkeys):
    a = df[df['Task&Direct']==pairkeys[0]][datakey].to_numpy()
    a_winid = df[df['Task&Direct']==pairkeys[0]]['Win ID'].to_numpy()
    a_subj = df[df['Task&Direct']==pairkeys[0]]['Subject'].to_numpy()
    b = df[df['Task&Direct']==pairkeys[1]][datakey].to_numpy()
    b_winid = df[df['Task&Direct']==pairkeys[1]]['Win ID'].to_numpy()
    b_subj = df[df['Task&Direct']==pairkeys[1]]['Subject'].to_numpy()
    subjs = np.intersect1d(a_subj, b_subj)
    paired_a, paired_b = [], []
    for subj in subjs:
        f1 = a_subj==subj
        f2 = b_subj==subj
        a_max = a_winid[f1].max()
        b_max = b_winid[f2].max()
        winid_max = min(a_max, b_max)
        a_w, a_ind = np.unique(a_winid[f1], return_index=True)
        b_w, b_ind = np.unique(b_winid[f2], return_index=True)
        a_ind = a_ind[:winid_max+1]
        b_ind = b_ind[:winid_max+1]
        paired_a.append(a[np.where(f1)[0][a_ind]])
        paired_b.append(b[np.where(f2)[0][b_ind]])
    paired_a = np.concatenate(paired_a)
    paired_b = np.concatenate(paired_b)
    return paired_a, paired_b

def fc_detour(args):
    if len(args) == 2:
        fc, sc = args
        k = 3
    elif len(args) == 3:
        fc, sc, k = args
    # de = torch.zeros_like(sc).long()
    G = nx.Graph(sc.numpy())
    de_paths = []
    for i, j in torch.stack(torch.where(fc)).T:
        de_path = get_de(G, i.item(), j.item(), k)
        # de[i, j] = len([p for p in de_path if len(p) > 2])
        de_paths.append([p for p in de_path if len(p) > 2])
    # output = [de, de_paths, torch.stack(torch.where(fc)).T]
    # return output
    return de_paths

def get_de(G, ni, nj, k):
    de_paths = list(nx.all_simple_paths(G, source=ni, target=nj, cutoff=k))
    return de_paths

def load_sc(path, atlas_name):
    fpath = f"{path}/{[f for f in os.listdir(path) if f.endswith('.mat')][0]}"
    sc_mat = loadmat(fpath)
    mat = sc_mat[f"{atlas_name.lower().replace('_','')}_sift_radius2_count_connectivity"]
    mat = torch.from_numpy(mat.astype(np.float32))
    mat = (mat + mat.T) / 2
    mat = (mat - mat.min()) / (mat.max() - mat.min())
    rnames = sc_mat[f"{atlas_name.lower().replace('_','')}_region_labels"]
    return mat, rnames, path.split('/')[-1]

def bold2fc(path, winsize, overlap, onlybold=False):
    bold_pd = pd.read_csv(path) if path.endswith('.csv') else pd.read_csv(path, sep='\t')
    rnames = list(bold_pd.columns[1:])
    bold = torch.from_numpy(np.array(bold_pd)[:, 1:]).float().T
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


def segment_node_with_neighbor(edge_index, node_attrs=[], edge_attrs=[], pad_value=0):
    edge_attr_ch = [edge_attr.shape[1] for edge_attr in edge_attrs]
    edge_index, edge_attrs = remove_self_loops(edge_index, torch.cat(edge_attrs, -1) if len(edge_attrs)>0 else None)
    edge_index, edge_attrs = add_self_loops(edge_index, edge_attrs)
    if len(node_attrs[0]) > edge_index.max()+1:
        if edge_attrs is not None:
            edge_attrs = torch.cat([edge_attrs] + [torch.zeros(1, edge_attrs.shape[1]) for i in range(edge_index.max()+1, len(node_attrs[0]))], 0)
        edge_index = torch.cat([edge_index] + [torch.LongTensor([[i, i]]).T for i in range(edge_index.max()+1, len(node_attrs[0]))], 1)

    sortid = edge_index[0].argsort()
    edge_index = edge_index[:, sortid]
    if edge_attrs is not None:
        edge_attrs = edge_attrs[sortid]
    edge_attr_ch = [0] + torch.LongTensor(edge_attr_ch).cumsum(0).tolist()
    edge_attrs = [edge_attrs[:, edge_attr_ch[i]:edge_attr_ch[i+1]] for i in range(len(edge_attr_ch)-1)]
    id_mask = edge_index[0] == edge_index[1]
    edge_attrs.append(id_mask.float()[:, None])
    for i in range(len(node_attrs)):
        node_attrs[i] = torch.cat([node_attrs[i][edge_index[0]], node_attrs[i][edge_index[1]]], -1)
    attrs = node_attrs + edge_attrs
    segment = [torch.where(edge_index[0]==e)[0][0].item() for e in edge_index.unique()] + [edge_index.shape[1]]
    seq = [[] for _ in range(len(attrs))]
    seq_mask = []
    for i in range(len(segment)-1):
        for j in range(len(attrs)):
            attr = attrs[j][segment[i]:segment[i+1]]
            selfloop = torch.where(edge_index[0, segment[i]:segment[i+1]]==edge_index[1, segment[i]:segment[i+1]])[0].item()
            attr = torch.cat([attr[selfloop:selfloop+1], attr[:selfloop], attr[selfloop+1:]]) # Move self loop to the first place
            seq[j].append(attr)
        seq_mask.append(torch.ones(seq[j][0].shape[0], 1))
    seq = [pad_sequence(s, batch_first=True, padding_value=pad_value) for s in seq] # [(N, S, C)]
    seq_mask = pad_sequence(seq_mask, batch_first=True, padding_value=0).float()
    return seq, seq_mask, edge_index


if __name__ == '__main__':
    main()