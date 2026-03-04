import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from datetime import datetime

meta_tgt = 'MMSE'
# noden = 160
# tgt_dn = 'oasis'
noden = 116
tgt_dn = 'adni'

node_data = pd.read_csv(f'analyze_nodedata_{tgt_dn}.csv')
print(node_data)
exit()

fig, axes = plt.subplots(1, 4, figsize=(30, 5))
sns.boxplot(data=node_data, y='path_index', hue='label', x='fc_edge', ax=axes[0])
sns.boxplot(data=node_data, y='path_index', hue=meta_tgt, x='fc_edge', ax=axes[1])
sns.boxplot(data=node_data, hue='sex', y='path_index', x='fc_edge', ax=axes[2])
sns.boxplot(data=node_data, hue='age', y='path_index', x='fc_edge', ax=axes[3])
for ax in axes:
    ax.tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.savefig(f'analyze_path_adj_{tgt_dn}.png')
plt.close()

# fig, axes = plt.subplots(1, 4, figsize=(30, 5))
# sns.boxplot(data=path_data, y='path_len', hue='label', x='fc_edge', ax=axes[0])
# sns.boxplot(data=path_data, y='path_len', hue=meta_tgt, x='fc_edge', ax=axes[1])
# sns.boxplot(data=path_data, hue='sex', y='path_len', x='fc_edge', ax=axes[2])
# sns.boxplot(data=path_data, hue='age', y='path_len', x='fc_edge', ax=axes[3])
# for ax in axes:
#     ax.tick_params(axis='x', rotation=90)

# plt.tight_layout()
# plt.savefig(f'analyze_path_len_{tgt_dn}.png')
