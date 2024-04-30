## Exp datasets (BOLD and SC mat)

 * HCP-Aging (4-task, 2-direction) ~4800 fMRI 716 subjects
 * HCP-YA (7-Task, 2-direction) ~4650 fMRI 331 subjects
 * UKB (2-task, 2-session) ~9000 fMRI ~4200 subjects
 * ADNI (AD/CN) 135 rfMRI
 * OASIS (AD/CN) 250 rfMRI
 * ADNI-DOD (unknown) 95 sMRI 319 rfMRI
 * PPMI (Autism, not done)

## Compare methods

### Model for brain connectome

 * BrainNetTransformer (BNT, 2023): take FC as input, permutation variant
 * Fbnetgen (2022): similar as above
 * BolT (2023): BOLD as input, permutation variant
 * BrainTokenGT (2023): Dynamic FC as input, permutation invariant
 * GNNs (older than 2020) from https://github.com/basiralab/GNNs-in-Network-Neuroscience
 * ~~BrainGNN~~ (too old, specific for biomarker)

### General graph transformer

 * Graphormer (2021)
 * NAGphormer (2022)
 * GRIT (2023)

### General GNN

 * GCN
 * GIN
 * SAGE
 * SGC

