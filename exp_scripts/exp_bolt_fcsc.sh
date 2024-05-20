
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --dataname hcpa --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:1 --node_attr SC > logs/bolt_gcn_hcpa_statfc_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --dataname hcpa --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:1 --node_attr SC > logs/bolt_gcn_hcpa_dynfc_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --dataname hcpa --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:1 --node_attr SC --atlas Gordon_333 > logs/bolt_gcn_hcpa_statfc_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --dataname hcpa --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:1 --node_attr SC --atlas Gordon_333 > logs/bolt_gcn_hcpa_dynfc_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bolt --classifier gcn --dataname ukb --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:1 --node_attr SC > logs/bolt_gcn_ukb_statfc_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bolt --classifier gcn --dataname ukb --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:1 --node_attr SC > logs/bolt_gcn_ukb_dynfc_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bolt --classifier gcn --dataname ukb --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:1 --node_attr SC --atlas Gordon_333 > logs/bolt_gcn_ukb_statfc_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bolt --classifier gcn --dataname ukb --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:1 --node_attr SC --atlas Gordon_333 > logs/bolt_gcn_ukb_dynfc_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --dataname adni --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:1 --node_attr SC > logs/bolt_gcn_adni_statfc_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --dataname adni --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:1 --node_attr SC > logs/bolt_gcn_adni_dynfc_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --dataname oasis --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:1 --node_attr SC --atlas D_160 > logs/bolt_gcn_oasis_statfc_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --dataname oasis --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:1 --node_attr SC --atlas D_160 > logs/bolt_gcn_oasis_dynfc_${dt}.log
