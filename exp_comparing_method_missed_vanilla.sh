
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname ukb --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:2 --node_attr FC > logs/none_gcn_ukb_statfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname ukb --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:2 --node_attr FC > logs/none_gcn_ukb_dynfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname ukb --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:2 --node_attr FC --atlas Gordon_333 > logs/none_gcn_ukb_statfcFC_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname ukb --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:2 --node_attr FC --atlas Gordon_333 > logs/none_gcn_ukb_dynfcFC_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname ukb --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:2 --node_attr FC > logs/none_gcn_ukb_statfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname ukb --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:2 --node_attr FC > logs/none_gcn_ukb_dynfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname ukb --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:2 --node_attr FC --atlas Gordon_333 > logs/none_gcn_ukb_statfcFC_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname ukb --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:2 --node_attr FC --atlas Gordon_333 > logs/none_gcn_ukb_dynfcFC_gordon_${dt}.log

