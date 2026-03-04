layern=2
headn=7

sc_th=0.3
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --nlayer ${layern} --nhead ${headn} --epochs 30 --max_patience 15 --sc_th ${sc_th} --dataname hcpa --bold_winsize 500 --device cuda:1 --node_attr BOLD --atlas Gordon_333 > mia_logs/bolt${headn}H${layern}L_SCth${sc_th}_gcn_hcpa_statfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --nlayer ${layern} --nhead ${headn} --epochs 30 --max_patience 15 --sc_th ${sc_th} --dataname ukb --bold_winsize 500 --device cuda:1 --node_attr BOLD --atlas Gordon_333 > mia_logs/bolt${headn}H${layern}L_SCth${sc_th}_gcn_ukb_statfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --nlayer ${layern} --nhead ${headn} --epochs 30 --max_patience 15 --sc_th ${sc_th} --dataname adni --bold_winsize 500 --device cuda:1 --node_attr BOLD > mia_logs/bolt${headn}H${layern}L_SCth${sc_th}_gcn_adni_statfcBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --nlayer ${layern} --nhead ${headn} --epochs 30 --max_patience 15 --sc_th ${sc_th} --dataname oasis --bold_winsize 500 --device cuda:1 --node_attr BOLD --atlas D_160 > mia_logs/bolt${headn}H${layern}L_SCth${sc_th}_gcn_oasis_statfcBOLD_${dt}.log

sc_th=0.5
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --nlayer ${layern} --nhead ${headn} --epochs 30 --max_patience 15 --sc_th ${sc_th} --dataname hcpa --bold_winsize 500 --device cuda:1 --node_attr BOLD --atlas Gordon_333 > mia_logs/bolt${headn}H${layern}L_SCth${sc_th}_gcn_hcpa_statfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --nlayer ${layern} --nhead ${headn} --epochs 30 --max_patience 15 --sc_th ${sc_th} --dataname ukb --bold_winsize 500 --device cuda:1 --node_attr BOLD --atlas Gordon_333 > mia_logs/bolt${headn}H${layern}L_SCth${sc_th}_gcn_ukb_statfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --nlayer ${layern} --nhead ${headn} --epochs 30 --max_patience 15 --sc_th ${sc_th} --dataname adni --bold_winsize 500 --device cuda:1 --node_attr BOLD > mia_logs/bolt${headn}H${layern}L_SCth${sc_th}_gcn_adni_statfcBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bolt --classifier gcn --nlayer ${layern} --nhead ${headn} --epochs 30 --max_patience 15 --sc_th ${sc_th} --dataname oasis --bold_winsize 500 --device cuda:1 --node_attr BOLD --atlas D_160 > mia_logs/bolt${headn}H${layern}L_SCth${sc_th}_gcn_oasis_statfcBOLD_${dt}.log
