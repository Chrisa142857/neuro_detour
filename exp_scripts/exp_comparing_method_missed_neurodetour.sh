dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:3 --node_attr FC --atlas Gordon_333 --nhead 5 --nlayer 2 > logs/neurodetour5H2L_gcn_hcpa_dynfcFC_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:3 --node_attr BOLD --atlas Gordon_333 --nhead 5 --nlayer 2 > logs/neurodetour5H2L_gcn_hcpa_statfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:3 --node_attr BOLD --atlas Gordon_333 --nhead 5 --nlayer 2 > logs/neurodetour5H2L_gcn_hcpa_dynfcBOLD_gordon_${dt}.log
