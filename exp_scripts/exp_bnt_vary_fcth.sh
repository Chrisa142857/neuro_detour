layern=2
headn=7

fc_th=0.3
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --nlayer ${layern} --nhead ${headn} --epochs 30 --max_patience 15 --fc_th ${fc_th} --dataname hcpa --bold_winsize 500 --device cuda:2 --node_attr BOLD --atlas Gordon_333 > rebuttal_logs/bnt${headn}H${layern}L_FCth${fc_th}_gcn_hcpa_statfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --nlayer ${layern} --nhead ${headn} --epochs 30 --max_patience 15 --fc_th ${fc_th} --dataname ukb --bold_winsize 500 --device cuda:2 --node_attr BOLD --atlas Gordon_333 > rebuttal_logs/bnt${headn}H${layern}L_FCth${fc_th}_gcn_ukb_statfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --nlayer ${layern} --nhead ${headn} --epochs 30 --max_patience 15 --fc_th ${fc_th} --dataname adni --bold_winsize 500 --device cuda:2 --node_attr BOLD > rebuttal_logs/bnt${headn}H${layern}L_FCth${fc_th}_gcn_adni_statfcBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --nlayer ${layern} --nhead ${headn} --epochs 30 --max_patience 15 --fc_th ${fc_th} --dataname oasis --bold_winsize 500 --device cuda:2 --node_attr BOLD --atlas D_160 > rebuttal_logs/bnt${headn}H${layern}L_FCth${fc_th}_gcn_oasis_statfcBOLD_${dt}.log

fc_th=0.7
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --nlayer ${layern} --nhead ${headn} --epochs 30 --max_patience 15 --fc_th ${fc_th} --dataname hcpa --bold_winsize 500 --device cuda:2 --node_attr BOLD --atlas Gordon_333 > rebuttal_logs/bnt${headn}H${layern}L_FCth${fc_th}_gcn_hcpa_statfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --nlayer ${layern} --nhead ${headn} --epochs 30 --max_patience 15 --fc_th ${fc_th} --dataname ukb --bold_winsize 500 --device cuda:2 --node_attr BOLD --atlas Gordon_333 > rebuttal_logs/bnt${headn}H${layern}L_FCth${fc_th}_gcn_ukb_statfcBOLD_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --nlayer ${layern} --nhead ${headn} --epochs 30 --max_patience 15 --fc_th ${fc_th} --dataname adni --bold_winsize 500 --device cuda:2 --node_attr BOLD > rebuttal_logs/bnt${headn}H${layern}L_FCth${fc_th}_gcn_adni_statfcBOLD_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --nlayer ${layern} --nhead ${headn} --epochs 30 --max_patience 15 --fc_th ${fc_th} --dataname oasis --bold_winsize 500 --device cuda:2 --node_attr BOLD --atlas D_160 > rebuttal_logs/bnt${headn}H${layern}L_FCth${fc_th}_gcn_oasis_statfcBOLD_${dt}.log
