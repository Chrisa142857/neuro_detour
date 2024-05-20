
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 1 --atlas Gordon_333 > logs/neurodetour1H2L_gcn_hcpa_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 1 --atlas Gordon_333 > logs/neurodetour1H2L_gcn_ukb_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname adni --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 1 > logs/neurodetour1H2L_gcn_adni_statfcFC_aal_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 1 --atlas D_160 > logs/neurodetour1H2L_gcn_oasis_statfcFC_${dt}.log


dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 3 --atlas Gordon_333 > logs/neurodetour3H2L_gcn_hcpa_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 3 --atlas Gordon_333 > logs/neurodetour3H2L_gcn_ukb_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname adni --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 3 > logs/neurodetour3H2L_gcn_adni_statfcFC_aal_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 3 --atlas D_160 > logs/neurodetour3H2L_gcn_oasis_statfcFC_${dt}.log


dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 4 --atlas Gordon_333 > logs/neurodetour4H2L_gcn_hcpa_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 4 --atlas Gordon_333 > logs/neurodetour4H2L_gcn_ukb_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname adni --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 4 > logs/neurodetour4H2L_gcn_adni_statfcFC_aal_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 4 --atlas D_160 > logs/neurodetour4H2L_gcn_oasis_statfcFC_${dt}.log


dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 5 --atlas Gordon_333 > logs/neurodetour5H2L_gcn_hcpa_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 5 --atlas Gordon_333 > logs/neurodetour5H2L_gcn_ukb_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname adni --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 5 > logs/neurodetour5H2L_gcn_adni_statfcFC_aal_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 5 --atlas D_160 > logs/neurodetour5H2L_gcn_oasis_statfcFC_${dt}.log


dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 6 --atlas Gordon_333 > logs/neurodetour6H2L_gcn_hcpa_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 6 --atlas Gordon_333 > logs/neurodetour6H2L_gcn_ukb_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname adni --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 6 > logs/neurodetour6H2L_gcn_adni_statfcFC_aal_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 6 --atlas D_160 > logs/neurodetour6H2L_gcn_oasis_statfcFC_${dt}.log


dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 7 --atlas Gordon_333 > logs/neurodetour7H2L_gcn_hcpa_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 7 --atlas Gordon_333 > logs/neurodetour7H2L_gcn_ukb_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname adni --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 7 > logs/neurodetour7H2L_gcn_adni_statfcFC_aal_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 7 --atlas D_160 > logs/neurodetour7H2L_gcn_oasis_statfcFC_${dt}.log


dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 8 --atlas Gordon_333 > logs/neurodetour8H2L_gcn_hcpa_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 8 --atlas Gordon_333 > logs/neurodetour8H2L_gcn_ukb_statfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname adni --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 8 > logs/neurodetour8H2L_gcn_adni_statfcFC_aal_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:1 --node_attr FC --nlayer 2 --nhead 8 --atlas D_160 > logs/neurodetour8H2L_gcn_oasis_statfcFC_${dt}.log
