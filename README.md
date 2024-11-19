[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neuropath-a-neural-pathway-transformer-for/graph-classification-on-adni)](https://paperswithcode.com/sota/graph-classification-on-adni?p=neuropath-a-neural-pathway-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neuropath-a-neural-pathway-transformer-for/graph-classification-on-hcp-aging)](https://paperswithcode.com/sota/graph-classification-on-hcp-aging?p=neuropath-a-neural-pathway-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neuropath-a-neural-pathway-transformer-for/graph-classification-on-oasis)](https://paperswithcode.com/sota/graph-classification-on-oasis?p=neuropath-a-neural-pathway-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neuropath-a-neural-pathway-transformer-for/graph-classification-on-uk-biobank-brain-mri)](https://paperswithcode.com/sota/graph-classification-on-uk-biobank-brain-mri?p=neuropath-a-neural-pathway-transformer-for)

# File explanition

`trainval.py` is the main function of the 5-fold cross validation classification experiments. 

`datasets.py` contains a general dataset class and the dataloader generator.

`visual_pathway.py` can visualize attention map and calculate weights of neural pathways.

`ttest_hcpa.py` can reproduce ttest results in the paper.

`models/*.py` contains all deep models tested in the experiments. `model/neuro_detour.py` is the proposed model. `model/graphormer.py` is Graphormer implemented by us with calculating the shortest-path distance (SPD) as a pre-transform of dataset.

`exp_scripts/*.sh` contains all shell commands for the experiments.

# For future works

## Available datasets (BOLD and SC mat)

 * HCP-Aging (4-task, 2-direction) ~4800 fMRI 716 subjects
 * HCP-YA (7-Task, 2-direction) ~3200 fMRI 248 subjects
 * UKB (2-task, 2-session) ~9000 fMRI ~4200 subjects
 * ADNI (AD/CN) 135 rfMRI&DWI
 * OASIS (AD/CN) 250 rfMRI&DWI
 * ADNI-DOD (Label unknown) 95 DWI 319 rfMRI
 * PPMI (Autism, not done)

## Available methods

### Model for brain connectome

 * BrainNetTransformer (BNT, 2023): take FC as input, permutation variant
 * FBnetgen (2022): Similar to BNT
 * BolT (2023): BOLD as input, permutation variant
 * BrainGNN (2021): Adj and node feature as input, node aggregation twice.
 * HUNet

### General graph transformer

 * Graphormer (2021)
 * NAGphormer (2022)

### General GNN

 * GCN
 * GIN
 * SAGE
 * SGC

### Cite
````bibtex
@inproceedings{wei2024neuropath,
  dimensions = {true},
  title = {NeuroPath: A Neural Pathway Transformer for Joining the Dots of Human Connectomes},
  author = {Wei, Ziquan and Dan, Tingting and Ding, Jiaqi and Wu, Guorong},
  booktitle = {The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year = {2024},
  publisher = {NeurIPS},
  url = {https://openreview.net/forum?id=AvBuK8Ezrg}
}
````