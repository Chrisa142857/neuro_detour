
headn=7
layern=16
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --nlayer ${layern} --nhead ${headn} --dataname hcpa --bold_winsize 500 --device cuda:2 --node_attr BOLD --atlas Gordon_333 > rebuttal_logs/neurodetour${headn}H${layern}L_gcn_hcpa_statfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --nlayer ${layern} --nhead ${headn} --dataname ukb --bold_winsize 500 --device cuda:2 --node_attr BOLD --atlas Gordon_333 > rebuttal_logs/neurodetour${headn}H${layern}L_gcn_ukb_statfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --nlayer ${layern} --nhead ${headn} --dataname adni --bold_winsize 500 --device cuda:2 --node_attr BOLD > rebuttal_logs/neurodetour${headn}H${layern}L_gcn_adni_statfcBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --nlayer ${layern} --nhead ${headn} --dataname oasis --bold_winsize 500 --device cuda:2 --node_attr BOLD --atlas D_160 > rebuttal_logs/neurodetour${headn}H${layern}L_gcn_oasis_statfcBOLD_${dt}.log

layern=32
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --nlayer ${layern} --nhead ${headn} --dataname hcpa --bold_winsize 500 --device cuda:2 --node_attr BOLD --atlas Gordon_333 > rebuttal_logs/neurodetour${headn}H${layern}L_gcn_hcpa_statfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --nlayer ${layern} --nhead ${headn} --dataname ukb --bold_winsize 500 --device cuda:2 --node_attr BOLD --atlas Gordon_333 > rebuttal_logs/neurodetour${headn}H${layern}L_gcn_ukb_statfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --nlayer ${layern} --nhead ${headn} --dataname adni --bold_winsize 500 --device cuda:2 --node_attr BOLD > rebuttal_logs/neurodetour${headn}H${layern}L_gcn_adni_statfcBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --nlayer ${layern} --nhead ${headn} --dataname oasis --bold_winsize 500 --device cuda:2 --node_attr BOLD --atlas D_160 > rebuttal_logs/neurodetour${headn}H${layern}L_gcn_oasis_statfcBOLD_${dt}.log