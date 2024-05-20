
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetourSingleSC --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:3 --node_attr FC > logs/neurodetourSingleSC2H1L_gcn_hcpa_statfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetourSingleSC --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:3 --node_attr FC --atlas Gordon_333 > logs/neurodetourSingleSC2H1L_gcn_hcpa_statfcFC_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetourSingleSC --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:3 --node_attr FC > logs/neurodetourSingleSC2H1L_gcn_ukb_statfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetourSingleSC --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:3 --node_attr FC --atlas Gordon_333 > logs/neurodetourSingleSC2H1L_gcn_ukb_statfcFC_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetourSingleSC --classifier gcn --dataname adni --bold_winsize 500 --device cuda:3 --node_attr FC > logs/neurodetourSingleSC2H1L_gcn_adni_statfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetourSingleSC --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:3 --node_attr FC --atlas D_160 > logs/neurodetourSingleSC2H1L_gcn_oasis_statfcFC_${dt}.log


dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetourSingleFC --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:3 --node_attr FC > logs/neurodetourSingleFC2H1L_gcn_hcpa_statfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetourSingleFC --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:3 --node_attr FC --atlas Gordon_333 > logs/neurodetourSingleFC2H1L_gcn_hcpa_statfcFC_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetourSingleFC --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:3 --node_attr FC > logs/neurodetourSingleFC2H1L_gcn_ukb_statfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetourSingleFC --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:3 --node_attr FC --atlas Gordon_333 > logs/neurodetourSingleFC2H1L_gcn_ukb_statfcFC_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetourSingleFC --classifier gcn --dataname adni --bold_winsize 500 --device cuda:3 --node_attr FC > logs/neurodetourSingleFC2H1L_gcn_adni_statfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetourSingleFC --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:3 --node_attr FC --atlas D_160 > logs/neurodetourSingleFC2H1L_gcn_oasis_statfcFC_${dt}.log

