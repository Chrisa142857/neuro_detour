
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models braingnn --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:2 --epochs 30 --max_patience 10 --node_attr BOLD --atlas AAL_116 --testname ukb --classifier_aggr mean > logs/braingnn_hcpa-ukb_gcn_statfcBOLD_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models braingnn --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:2 --epochs 30 --max_patience 10 --node_attr BOLD --atlas AAL_116 --testname hcpa --classifier_aggr mean > logs/braingnn_ukb-hcpa_gcn_statfcBOLD_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models braingnn --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:2 --epochs 30 --max_patience 10 --node_attr BOLD --atlas AAL_116 --testname ukb --classifier_aggr mean > logs/braingnn_hcpa-ukb_gcn_dynfcBOLD_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models braingnn --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:2 --epochs 30 --max_patience 10 --node_attr BOLD --atlas AAL_116 --testname hcpa --classifier_aggr mean > logs/braingnn_ukb-hcpa_gcn_dynfcBOLD_${dt}.log


# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bnt --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:2 --epochs 30 --max_patience 10 --node_attr BOLD --atlas Gordon_333 --testname ukb --classifier_aggr mean > logs/bnt_hcpa-ukb_gcn_statfcBOLD_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bnt --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:2 --epochs 30 --max_patience 10 --node_attr BOLD --atlas AAL_116 --testname hcpa --classifier_aggr mean > logs/bnt_ukb-hcpa_gcn_statfcBOLD_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bnt --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:2 --epochs 30 --max_patience 10 --node_attr BOLD --atlas AAL_116 --testname ukb --classifier_aggr mean > logs/bnt_hcpa-ukb_gcn_dynfcBOLD_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bnt --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:2 --epochs 30 --max_patience 10 --node_attr BOLD --atlas AAL_116 --testname hcpa --classifier_aggr mean > logs/bnt_ukb-hcpa_gcn_dynfcBOLD_${dt}.log


dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:1 --epochs 30 --max_patience 10 --node_attr BOLD --atlas Gordon_333 --testname ukb --classifier_aggr mean > logs/bolt_hcpa-ukb_gcn_statfcBOLD_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:1 --epochs 30 --max_patience 10 --node_attr BOLD --atlas Gordon_333 --testname hcpa --classifier_aggr mean > logs/bolt_ukb-hcpa_gcn_statfcBOLD_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:1 --epochs 30 --max_patience 10 --node_attr BOLD --atlas Gordon_333 --testname ukb --classifier_aggr mean > logs/bolt_hcpa-ukb_gcn_dynfcBOLD_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:1 --epochs 30 --max_patience 10 --node_attr BOLD --atlas Gordon_333 --testname hcpa --classifier_aggr mean > logs/bolt_ukb-hcpa_gcn_dynfcBOLD_${dt}.log

