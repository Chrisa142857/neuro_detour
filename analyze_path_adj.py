import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from datetime import datetime

# r = 'resources_nimg'
# noden = 160
# tgt_dn = 'oasis'
# # noden = 116
# # tgt_dn = 'adni'
# meta_tgt = 'MMSE'
# if tgt_dn == 'oasis':
#     meta = pd.read_csv('/ram/USERS/ziquanw/data/meta_data/OASIS_metadata.csv')
#     meta['Subject'] = meta['Subject'].str.split('_').str[0]

# elif tgt_dn == 'adni':
#     meta = pd.read_csv('/ram/USERS/ziquanw/data/meta_data/ADNI_all_metadata.csv')
#     meta['Subject'] = 'sub-' + meta['Subject'].str.replace('_', '').astype(str)

# labeln = ['CN', 'AD']
# path_index_age = []
# path_index_sex = []
# path_index_meta_score = []
# age = []
# meta_score = []
# labels = []
# # adj = []
# sex = []
# fc_edges = []
# # path_bincount = []
# path_index = []
# path_index_label = []
# path_index_pathlen = []
# path_index_fcedge = []
# path_len = []

# for fn in tqdm(os.listdir(r)):
#     # if '93-138' not in fn and '53-59' not in fn: continue
#     if tgt_dn not in fn: continue
#     fpath_list.append(f'{r}/{fn}')
#     label = int(fn.split('_')[-2].replace('label', ''))
#     a = np.loadtxt(f'{r}/{fn}', delimiter='\t')
#     if len(np.unique(a)) == 1: continue
#     # print(f'{r}/{fn}', np.unique(a))
#     # adj.append(a)
#     path_len.append(a.sum())
#     labels.append(labeln[label])
#     subj = fn.split('_')[5].split('d')[0]
#     meta_score.append(meta[meta_tgt][meta['Subject']==subj].iloc[0])    
#     age.append(meta['Age'][meta['Subject']==subj].iloc[0])
#     sex.append(meta['Sex'][meta['Subject']==subj].iloc[0])
#     age[-1] = (age[-1] // 5) * 5
#     fc_edge = fn.split('_')[4].replace('sign', '')
#     fc_edges.append(fc_edge)
#     # path_bincount.append(np.bincount(np.stack(np.where(a)).reshape(-1), minlength=noden))
#     path_index.append(np.stack(np.where(a)).reshape(-1))
#     path_index_label.extend([labeln[label] for _ in range(len(np.stack(np.where(a)).reshape(-1)))])
#     path_index_age.extend([age[-1] for _ in range(len(np.stack(np.where(a)).reshape(-1)))])
#     path_index_sex.extend([sex[-1] for _ in range(len(np.stack(np.where(a)).reshape(-1)))])
#     path_index_meta_score.extend([meta[meta_tgt][meta['Subject']==subj].iloc[0] for _ in range(len(np.stack(np.where(a)).reshape(-1)))])
#     # path_index_pathlen.extend([a.sum() for _ in range(len(np.stack(np.where(a)).reshape(-1)))])
#     path_index_fcedge.extend([fc_edge for _ in range(len(np.stack(np.where(a)).reshape(-1)))])

def main():
    r = 'resources_nimg'
    noden = 160
    tgt_dn = 'oasis'
    # noden = 116
    # tgt_dn = 'adni'
    meta_tgt = 'MMSE'
    if tgt_dn == 'oasis':
        meta = pd.read_csv('/ram/USERS/ziquanw/data/meta_data/OASIS_metadata.csv')
        meta['Subject'] = meta['Subject'].str.split('_').str[0]

    elif tgt_dn == 'adni':
        meta = pd.read_csv('/ram/USERS/ziquanw/data/meta_data/ADNI_all_metadata.csv')
        meta['Subject'] = 'sub-' + meta['Subject'].str.replace('_', '').astype(str)

    path_lens = []
    ages = []
    meta_scores = []
    labels = []
    sexs = []
    fc_edges = []
    
    path_index = []
    path_index_ages = []
    path_index_sexs = []
    path_index_meta_scores = []
    path_index_labels = []
    path_index_fcedges = []
    fpath_list = []
    output_list = []
    for fn in tqdm(os.listdir(r), desc='Prepare'):
        # if '93-138' not in fn and '53-59' not in fn: continue
        if tgt_dn not in fn: continue
        fpath_list.append([f'{r}/{fn}', meta, meta_tgt])
        # output_list.append(data_analyze([f'{r}/{fn}', meta, meta_tgt]))
    
    with Pool(processes=20) as loader_pool:
        output_list = list(loader_pool.imap(data_analyze, tqdm(fpath_list, desc=f'{datetime.now()} Loading path adj {tgt_dn}')))

    for output in tqdm(output_list, desc=f'{datetime.now()} Get output'):
        if output is None: continue
        label, age, sex, path_len, fc_edge, meta_score, path_index_node_id, path_index_label, path_index_age, path_index_sex, path_index_meta_score, path_index_fcedge = output
        path_lens.append(path_len)
        ages.append(age)
        meta_scores.append(meta_score)
        labels.append(label)
        sexs.append(sex)
        fc_edges.append(fc_edge)

        # path_index.append(path_index_node_id)
        # path_index_ages.extend(path_index_age)
        # path_index_sexs.extend(path_index_sex)
        # path_index_meta_scores.extend(path_index_meta_score)
        # path_index_labels.extend(path_index_label)
        # path_index_fcedges.extend(path_index_fcedge)
   
    fc_edges = np.array(fc_edges)
    path_lens = np.array(path_lens)
    labels = np.array(labels)
    ages = np.array(ages)
    sexs = np.array(sexs)
    meta_scores = np.array(meta_scores)
    path_data = {
        'label': labels,
        'age': ages,
        'sex': sexs,
        'path_len': path_lens,
        'fc_edge': fc_edges,
        meta_tgt: meta_scores
    }

    path_data = pd.DataFrame(path_data)
    path_data.to_csv(f'analyze_pathdata_{tgt_dn}.csv')
    exit()

    path_index = np.concatenate(path_index)
    path_index_label = np.array(path_index_labels)
    path_index_age = np.array(path_index_ages)
    path_index_sex = np.array(path_index_sexs)
    path_index_fcedge = np.array(path_index_fcedges)
    path_index_meta_score = np.array(path_index_meta_scores)
    node_data = {
        'path_index': path_index,
        'label': path_index_label,
        'age': path_index_age,
        'sex': path_index_sex,
        'fc_edge': path_index_fcedge,
        meta_tgt: path_index_meta_score,
    }
        
    node_data = pd.DataFrame(node_data)
    node_data.to_csv(f'analyze_nodedata_{tgt_dn}.csv')
    exit()

def get_sex(args):
    meta, fn = args
    subj = fn.split('_')[5].split('d')[0]
    sex = meta['Sex'][meta['Subject']==subj].iloc[0]
    return sex

def data_analyze(args):
    fpath, meta, meta_tgt = args
    
    labeln = ['CN', 'AD']
    fn = fpath.split('/')[-1]
    label = int(fn.split('_')[-2].replace('label', ''))
    a = np.loadtxt(fpath, delimiter='\t')
    if len(np.unique(a)) == 1: 
        return None

    path_len = a.sum()
    labels = labeln[label]
    subj = fn.split('_')[5].split('d')[0]
    meta_score = meta[meta_tgt][meta['Subject']==subj].iloc[0] 
    age = meta['Age'][meta['Subject']==subj].iloc[0]
    sex = meta['Sex'][meta['Subject']==subj].iloc[0]
    age = (age // 5) * 5

    fc_edge = fn.split('_')[4].replace('sign', '')
    return labels, age, sex, path_len, fc_edge, meta_score, None, None, None, None, None, None
    # fc_edges.append(fc_edge)
    # path_bincount.append(np.bincount(np.stack(np.where(a)).reshape(-1), minlength=noden))
    path_index_node_id = np.stack(np.where(a)).reshape(-1)
    path_index_label = [labeln[label] for _ in range(len(path_index_node_id))]
    path_index_age = [age for _ in range(len(path_index_node_id))]
    path_index_sex = [sex for _ in range(len(path_index_node_id))]
    path_index_meta_score = [meta[meta_tgt][meta['Subject']==subj].iloc[0] for _ in range(len(path_index_node_id))]
    path_index_fcedge = [fc_edge for _ in range(len(path_index_node_id))]

    return labels, age, sex, path_len, fc_edge, meta_score, path_index_node_id, path_index_label, path_index_age, path_index_sex, path_index_meta_score, path_index_fcedge


def random_sample_data(pd_data):
    valid_fcedge = np.unique(pd_data['fc_edge'])
    np.random.shuffle(valid_fcedge)
    valid_fcedge = valid_fcedge[:20]
    valid_ind = np.zeros(len(pd_data['fc_edge']), dtype=bool)
    for fcedge in valid_fcedge:
        valid_ind = np.logical_or(valid_ind, pd_data['fc_edge'] == fcedge)
    for k in pd_data:
        pd_data[k] = pd_data[k][valid_ind]

# path_index = np.concatenate(path_index)
# fc_edges = np.array(fc_edges)
# path_len = np.array(path_len)
# labels = np.array(labels)
# age = np.array(age)
# sex = np.array(sex)
# meta_score = np.array(meta_score)
# path_data = {
#     'label': labels,
#     'age': age,
#     'sex': sex,
#     'path_len': path_len,
#     'fc_edge': fc_edges,
#     meta_tgt: meta_score
# }

# path_data = pd.DataFrame(path_data)
# path_data.to_csv(f'analyze_pathdata_{tgt_dn}.csv')

# path_index_label = np.array(path_index_label)
# path_index_age = np.array(path_index_age)
# path_index_sex = np.array(path_index_sex)
# path_index_fcedge = np.array(path_index_fcedge)
# path_index_meta_score = np.array(path_index_meta_score)
# node_data = {
#     'path_index': path_index,
#     'label': path_index_label,
#     'age': path_index_age,
#     'sex': path_index_sex,
#     'fc_edge': path_index_fcedge,
#     meta_tgt: path_index_meta_score,
# }
    
# node_data = pd.DataFrame(node_data)
# node_data.to_csv(f'analyze_nodedata_{tgt_dn}.csv')
# exit()

# fig, axes = plt.subplots(1, 4, figsize=(30, 5))
# sns.boxplot(data=node_data, y='path_index', hue='label', x='fc_edge', ax=axes[0])
# sns.boxplot(data=node_data, y='path_index', hue=meta_tgt, x='fc_edge', ax=axes[1])
# sns.boxplot(data=node_data, hue='sex', y='path_index', x='fc_edge', ax=axes[2])
# sns.boxplot(data=node_data, hue='age', y='path_index', x='fc_edge', ax=axes[3])
# for ax in axes:
#     ax.tick_params(axis='x', rotation=90)

# plt.tight_layout()
# plt.savefig(f'analyze_path_adj_{tgt_dn}.png')
# plt.close()

# fig, axes = plt.subplots(1, 4, figsize=(30, 5))
# sns.boxplot(data=path_data, y='path_len', hue='label', x='fc_edge', ax=axes[0])
# sns.boxplot(data=path_data, y='path_len', hue=meta_tgt, x='fc_edge', ax=axes[1])
# sns.boxplot(data=path_data, hue='sex', y='path_len', x='fc_edge', ax=axes[2])
# sns.boxplot(data=path_data, hue='age', y='path_len', x='fc_edge', ax=axes[3])
# for ax in axes:
#     ax.tick_params(axis='x', rotation=90)

# plt.tight_layout()
# plt.savefig(f'analyze_path_len_{tgt_dn}.png')

if __name__ == '__main__': main()