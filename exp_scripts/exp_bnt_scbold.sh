
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:2 --adj_type SC --node_attr BOLD > logs/bnt_gcn_hcpa_statscBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:2 --adj_type SC --node_attr BOLD > logs/bnt_gcn_hcpa_dynscBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:2 --adj_type SC --node_attr BOLD --atlas Gordon_333 > logs/bnt_gcn_hcpa_statscBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:2 --adj_type SC --node_attr BOLD --atlas Gordon_333 > logs/bnt_gcn_hcpa_dynscBOLD_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bnt --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:2 --adj_type SC --node_attr BOLD > logs/bnt_gcn_ukb_statscBOLD_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bnt --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:2 --adj_type SC --node_attr BOLD > logs/bnt_gcn_ukb_dynscBOLD_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bnt --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:2 --adj_type SC --node_attr BOLD --atlas Gordon_333 > logs/bnt_gcn_ukb_statscBOLD_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bnt --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:2 --adj_type SC --node_attr BOLD --atlas Gordon_333 > logs/bnt_gcn_ukb_dynscBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --dataname adni --bold_winsize 500 --device cuda:2 --adj_type SC --node_attr BOLD > logs/bnt_gcn_adni_statscBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --dataname adni --bold_winsize 100 --device cuda:2 --adj_type SC --node_attr BOLD > logs/bnt_gcn_adni_dynscBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:2 --adj_type SC --node_attr BOLD --atlas D_160 > logs/bnt_gcn_oasis_statscBOLD_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --dataname oasis --bold_winsize 100 --device cuda:2 --adj_type SC --node_attr BOLD --atlas D_160 > logs/bnt_gcn_oasis_dynscBOLD_${dt}.log

