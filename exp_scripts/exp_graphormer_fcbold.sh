
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models graphormer --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:3 --node_attr BOLD > logs/graphormer_gcn_hcpa_statfcBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models graphormer --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:3 --node_attr BOLD > logs/graphormer_gcn_hcpa_dynfcBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models graphormer --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:3 --node_attr BOLD --atlas Gordon_333 > logs/graphormer_gcn_hcpa_statfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models graphormer --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:3 --node_attr BOLD --atlas Gordon_333 > logs/graphormer_gcn_hcpa_dynfcBOLD_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models graphormer --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:3 --node_attr BOLD > logs/graphormer_gcn_ukb_statfcBOLD_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models graphormer --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:3 --node_attr BOLD > logs/graphormer_gcn_ukb_dynfcBOLD_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models graphormer --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:3 --node_attr BOLD --atlas Gordon_333 > logs/graphormer_gcn_ukb_statfcBOLD_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models graphormer --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:3 --node_attr BOLD --atlas Gordon_333 > logs/graphormer_gcn_ukb_dynfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models graphormer --classifier gcn --dataname adni --bold_winsize 500 --device cuda:3 --node_attr BOLD > logs/graphormer_gcn_adni_statfcBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models graphormer --classifier gcn --dataname adni --bold_winsize 100 --device cuda:3 --node_attr BOLD > logs/graphormer_gcn_adni_dynfcBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models graphormer --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:3 --node_attr BOLD --atlas D_160 > logs/graphormer_gcn_oasis_statfcBOLD_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models graphormer --classifier gcn --dataname oasis --bold_winsize 100 --device cuda:3 --node_attr BOLD --atlas D_160 > logs/graphormer_gcn_oasis_dynfcBOLD_${dt}.log
