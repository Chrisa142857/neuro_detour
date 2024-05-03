## Exp datasets (BOLD and SC mat)

 * HCP-Aging (4-task, 2-direction) ~4800 fMRI 716 subjects
 * HCP-YA (7-Task, 2-direction) ~4650 fMRI 331 subjects
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

|   BNT |   ADNI    |   statFC
Mean Accuracy: 0.8355555555555554, Std Accuracy: 0.07470104098320007
Mean F1 Score: 0.7911541964509106, Std F1 Score: 0.07997008054784756
Mean prec Score: 0.789997362034399, Std prec Score: 0.10653295623335432
Mean rec Score: 0.8355555555555556, Std rec Score: 0.0747010502843183

|   BNT |   OASIS   |   statFC
Mean Accuracy: 0.8800000000000001, Std Accuracy: 0.037669043987989426
Mean F1 Score: 0.8526573669031545, Std F1 Score: 0.050339489346228956
Mean prec Score: 0.829305369910615, Std prec Score: 0.06898882309573053
Mean rec Score: 0.8800000000000001, Std rec Score: 0.03766905448389849

|   NeuroDetour |   ADNI    |   statFC
