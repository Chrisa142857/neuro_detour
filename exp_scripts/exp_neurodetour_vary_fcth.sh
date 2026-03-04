layern=2
headn=7

sc_th=0.3
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --nlayer ${layern} --nhead ${headn} --sc_th ${sc_th} --dataname hcpa --bold_winsize 500 --device cuda:1 --node_attr BOLD --atlas Gordon_333 > mia_logs/neurodetour${headn}H${layern}L_SCth${sc_th}_gcn_hcpa_statfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --nlayer ${layern} --nhead ${headn} --sc_th ${sc_th} --dataname ukb --bold_winsize 500 --device cuda:2 --node_attr BOLD --atlas Gordon_333 > mia_logs/neurodetour${headn}H${layern}L_SCth${sc_th}_gcn_ukb_statfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --nlayer ${layern} --nhead ${headn} --sc_th ${sc_th} --dataname adni --bold_winsize 500 --device cuda:3 --node_attr BOLD > mia_logs/neurodetour${headn}H${layern}L_SCth${sc_th}_gcn_adni_statfcBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --nlayer ${layern} --nhead ${headn} --sc_th ${sc_th} --dataname oasis --bold_winsize 500 --device cuda:4 --node_attr BOLD --atlas D_160 > mia_logs/neurodetour${headn}H${layern}L_SCth${sc_th}_gcn_oasis_statfcBOLD_${dt}.log

sc_th=0.5
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --nlayer ${layern} --nhead ${headn} --sc_th ${sc_th} --dataname hcpa --bold_winsize 500 --device cuda:5 --node_attr BOLD --atlas Gordon_333 > mia_logs/neurodetour${headn}H${layern}L_SCth${sc_th}_gcn_hcpa_statfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --nlayer ${layern} --nhead ${headn} --sc_th ${sc_th} --dataname ukb --bold_winsize 500 --device cuda:6 --node_attr BOLD --atlas Gordon_333 > mia_logs/neurodetour${headn}H${layern}L_SCth${sc_th}_gcn_ukb_statfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --nlayer ${layern} --nhead ${headn} --sc_th ${sc_th} --dataname adni --bold_winsize 500 --device cuda:7 --node_attr BOLD > mia_logs/neurodetour${headn}H${layern}L_SCth${sc_th}_gcn_adni_statfcBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --nlayer ${layern} --nhead ${headn} --sc_th ${sc_th} --dataname oasis --bold_winsize 500 --device cuda:7 --node_attr BOLD --atlas D_160 > mia_logs/neurodetour${headn}H${layern}L_SCth${sc_th}_gcn_oasis_statfcBOLD_${dt}.log
