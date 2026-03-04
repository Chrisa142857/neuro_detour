
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --dataname hcpa --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:3 --adj_type SC --node_attr BOLD > logs/bolt_gcn_hcpa_statfcBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --dataname hcpa --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:3 --adj_type SC --node_attr BOLD > logs/bolt_gcn_hcpa_dynfcBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --dataname hcpa --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:3 --adj_type SC --node_attr BOLD --atlas Gordon_333 > logs/bolt_gcn_hcpa_statfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --dataname hcpa --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:3 --adj_type SC --node_attr BOLD --atlas Gordon_333 > logs/bolt_gcn_hcpa_dynfcBOLD_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bolt --classifier gcn --dataname ukb --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:3 --adj_type SC --node_attr BOLD > logs/bolt_gcn_ukb_statfcBOLD_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bolt --classifier gcn --dataname ukb --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:3 --adj_type SC --node_attr BOLD > logs/bolt_gcn_ukb_dynfcBOLD_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bolt --classifier gcn --dataname ukb --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:3 --adj_type SC --node_attr BOLD --atlas Gordon_333 > logs/bolt_gcn_ukb_statfcBOLD_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bolt --classifier gcn --dataname ukb --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:3 --adj_type SC --node_attr BOLD --atlas Gordon_333 > logs/bolt_gcn_ukb_dynfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --dataname adni --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:3 --adj_type SC --node_attr BOLD > logs/bolt_gcn_adni_statfcBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --dataname adni --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:3 --adj_type SC --node_attr BOLD > logs/bolt_gcn_adni_dynfcBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --dataname oasis --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:3 --adj_type SC --node_attr BOLD --atlas D_160 > logs/bolt_gcn_oasis_statfcBOLD_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --dataname oasis --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:3 --adj_type SC --node_attr BOLD --atlas D_160 > logs/bolt_gcn_oasis_dynfcBOLD_${dt}.log
