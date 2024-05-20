
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models nagphormer --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:1 --epochs 30 --max_patience 10 --node_attr BOLD > logs/nagphormer_gcn_ukb_statfcBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models nagphormer --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:1 --epochs 30 --max_patience 10 --node_attr BOLD > logs/nagphormer_gcn_ukb_dynfcBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models nagphormer --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:1 --epochs 30 --max_patience 10 --node_attr BOLD --atlas Gordon_333 > logs/nagphormer_gcn_ukb_statfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models nagphormer --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:1 --epochs 30 --max_patience 10 --node_attr BOLD --atlas Gordon_333 > logs/nagphormer_gcn_ukb_dynfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models nagphormer --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:1 --epochs 30 --max_patience 10 --node_attr FC > logs/nagphormer_gcn_ukb_statfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models nagphormer --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:1 --epochs 30 --max_patience 10 --node_attr FC > logs/nagphormer_gcn_ukb_dynfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models nagphormer --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:1 --epochs 30 --max_patience 10 --node_attr FC --atlas Gordon_333 > logs/nagphormer_gcn_ukb_statfcFC_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models nagphormer --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:1 --epochs 30 --max_patience 10 --node_attr FC --atlas Gordon_333 > logs/nagphormer_gcn_ukb_dynfcFC_gordon_${dt}.log

