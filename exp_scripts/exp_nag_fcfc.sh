
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models nagphormer --classifier gcn --dataname hcpa --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:2 --node_attr FC > logs/nagphormer_gcn_hcpa_statfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models nagphormer --classifier gcn --dataname hcpa --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:2 --node_attr FC > logs/nagphormer_gcn_hcpa_dynfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models nagphormer --classifier gcn --dataname hcpa --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:2 --node_attr FC --atlas Gordon_333 > logs/nagphormer_gcn_hcpa_statfcFC_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models nagphormer --classifier gcn --dataname hcpa --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:2 --node_attr FC --atlas Gordon_333 > logs/nagphormer_gcn_hcpa_dynfcFC_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models nagphormer --classifier gcn --dataname ukb --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:2 --node_attr FC > logs/nagphormer_gcn_ukb_statfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models nagphormer --classifier gcn --dataname ukb --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:2 --node_attr FC > logs/nagphormer_gcn_ukb_dynfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models nagphormer --classifier gcn --dataname ukb --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:2 --node_attr FC --atlas Gordon_333 > logs/nagphormer_gcn_ukb_statfcFC_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models nagphormer --classifier gcn --dataname ukb --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:2 --node_attr FC --atlas Gordon_333 > logs/nagphormer_gcn_ukb_dynfcFC_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models nagphormer --classifier gcn --dataname adni --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:2 --node_attr FC > logs/nagphormer_gcn_adni_statfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models nagphormer --classifier gcn --dataname adni --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:2 --node_attr FC > logs/nagphormer_gcn_adni_dynfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models nagphormer --classifier gcn --dataname oasis --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:2 --node_attr FC --atlas D_160 > logs/nagphormer_gcn_oasis_statfcFC_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models nagphormer --classifier gcn --dataname oasis --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:2 --node_attr FC --atlas D_160 > logs/nagphormer_gcn_oasis_dynfcFC_${dt}.log
