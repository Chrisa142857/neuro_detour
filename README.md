## Exp datasets (BOLD and SC mat)

 * HCP-Aging (4-task, 2-direction) ~4800 fMRI 716 subjects
 * HCP-YA (7-Task, 2-direction) ~3200 fMRI 248 subjects
 * UKB (2-task, 2-session) ~9000 fMRI ~4200 subjects
 * ADNI (AD/CN) 135 rfMRI&DWI
 * OASIS (AD/CN) 250 rfMRI&DWI
 * ADNI-DOD (Label unknown) 95 DWI 319 rfMRI
 * PPMI (Autism, not done)

## Compare methods

### Model for brain connectome

 * BrainNetTransformer (BNT, 2023): take FC as input, permutation variant
 * Fbnetgen (2022): similar as above
 * BolT (2023): BOLD as input, permutation variant
 * BrainTokenGT (2023): Dynamic FC as input, permutation invariant
 * BrainGNN (2021): Adj and node feature as input, node aggregation twice.
 * GNNs (older than 2020) from https://github.com/basiralab/GNNs-in-Network-Neuroscience

### General graph transformer

 * Graphormer (2021)
 * NAGphormer (2022)
 * GRIT (2023)

### General GNN

 * GCN
 * GIN
 * SAGE
 * SGC

### Results
