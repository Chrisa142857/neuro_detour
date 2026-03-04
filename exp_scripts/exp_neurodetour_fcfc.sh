
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:2 --node_attr FC > logs/neurodetour2H1L_gcn_hcpa_statfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:2 --node_attr FC > logs/neurodetour2H1L_gcn_hcpa_dynfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:2 --node_attr FC --atlas Gordon_333 > logs/neurodetour2H1L_gcn_hcpa_statfcFC_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:3 --node_attr FC --atlas Gordon_333 > logs/neurodetour2H1L_gcn_hcpa_dynfcFC_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models neurodetour --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:2 --node_attr FC > logs/neurodetour2H1L_gcn_ukb_statfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models neurodetour --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:2 --node_attr FC > logs/neurodetour2H1L_gcn_ukb_dynfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models neurodetour --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:1 --node_attr FC --atlas Gordon_333 > logs/neurodetour2H1L_gcn_ukb_statfcFC_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models neurodetour --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:1 --node_attr FC --atlas Gordon_333 > logs/neurodetour2H1L_gcn_ukb_dynfcFC_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models neurodetour --classifier gcn --dataname adni --bold_winsize 500 --device cuda:2 --node_attr FC > logs/neurodetour2H1L_gcn_adni_statfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models neurodetour --classifier gcn --dataname adni --bold_winsize 100 --device cuda:2 --node_attr FC > logs/neurodetour2H1L_gcn_adni_dynfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models neurodetour --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:2 --node_attr FC --atlas D_160 > logs/neurodetour2H1L_gcn_oasis_statfcFC_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models neurodetour --classifier gcn --dataname oasis --bold_winsize 100 --device cuda:2 --node_attr FC --atlas D_160 > logs/neurodetour2H1L_gcn_oasis_dynfcFC_${dt}.log

