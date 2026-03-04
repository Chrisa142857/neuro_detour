
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --lconsist_w 0.25 --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:0 --node_attr FC --nlayer 2 --nhead 4 --atlas Gordon_333 > nimg_logs/neurodetour4H2LLc025_gcn_hcpa_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --lconsist_w 0.25 --models neurodetour --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 4 --atlas Gordon_333 > nimg_logs/neurodetour4H2LLc025_gcn_ukb_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --lconsist_w 0.25 --models neurodetour --classifier gcn --dataname adni --bold_winsize 500 --device cuda:0 --node_attr FC --nlayer 2 --nhead 4 > nimg_logs/neurodetour4H2LLc025_gcn_adni_statfcFC_aal_${dt}.log &

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --lconsist_w 0.25 --models neurodetour --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 4 --atlas D_160 > nimg_logs/neurodetour4H2LLc025_gcn_oasis_statfcFC_${dt}.log


dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --lconsist_w 0.5 --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:0 --node_attr FC --nlayer 2 --nhead 4 --atlas Gordon_333 > nimg_logs/neurodetour4H2LLc05_gcn_hcpa_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --lconsist_w 0.5 --models neurodetour --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 4 --atlas Gordon_333 > nimg_logs/neurodetour4H2LLc05_gcn_ukb_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --lconsist_w 0.5 --models neurodetour --classifier gcn --dataname adni --bold_winsize 500 --device cuda:2 --node_attr FC --nlayer 2 --nhead 4 > nimg_logs/neurodetour4H2LLc05_gcn_adni_statfcFC_aal_${dt}.log &

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --lconsist_w 0.5 --models neurodetour --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:3 --node_attr FC --nlayer 2 --nhead 4 --atlas D_160 > nimg_logs/neurodetour4H2LLc05_gcn_oasis_statfcFC_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --lconsist_w 1 --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:0 --node_attr FC --nlayer 2 --nhead 4 --atlas Gordon_333 > nimg_logs/neurodetour4H2LLc2_gcn_hcpa_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --lconsist_w 1 --models neurodetour --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 4 --atlas Gordon_333 > nimg_logs/neurodetour4H2LLc2_gcn_ukb_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --lconsist_w 1 --models neurodetour --classifier gcn --dataname adni --bold_winsize 500 --device cuda:4 --node_attr FC --nlayer 2 --nhead 4 > nimg_logs/neurodetour4H2LLc1_gcn_adni_statfcFC_aal_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --lconsist_w 1 --models neurodetour --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:5 --node_attr FC --nlayer 2 --nhead 4 --atlas D_160 > nimg_logs/neurodetour4H2LLc1_gcn_oasis_statfcFC_${dt}.log



dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --lconsist_w 2 --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:0 --node_attr FC --nlayer 2 --nhead 4 --atlas Gordon_333 > nimg_logs/neurodetour4H2LLc2_gcn_hcpa_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --lconsist_w 2 --models neurodetour --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 4 --atlas Gordon_333 > nimg_logs/neurodetour4H2LLc2_gcn_ukb_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --lconsist_w 2 --models neurodetour --classifier gcn --dataname adni --bold_winsize 500 --device cuda:4 --node_attr FC --nlayer 2 --nhead 4 > nimg_logs/neurodetour4H2LLc2_gcn_adni_statfcFC_aal_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --lconsist_w 2 --models neurodetour --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:5 --node_attr FC --nlayer 2 --nhead 4 --atlas D_160 > nimg_logs/neurodetour4H2LLc2_gcn_oasis_statfcFC_${dt}.log

