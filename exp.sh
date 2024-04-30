
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier mlp --dataname hcpa --bold_winsize 500 --device cuda:0 > logs/bnt_mlp_hcpa_statfc_${dt}.log
python trainval.py --models bnt --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:0 > logs/bnt_gcn_hcpa_statfc_${dt}.log

python trainval.py --models bnt --classifier mlp --dataname hcpa --bold_winsize 500 --device cuda:0 > logs/bnt_mlp_hcpa_statfc_${dt}.log
python trainval.py --models bnt --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:0 > logs/bnt_gcn_hcpa_statfc_${dt}.log

python trainval.py --models bnt --classifier gcn --dataname adni --bold_winsize 500 --device cuda:1
python trainval.py --models bnt --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:1

python trainval.py --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:1 
python trainval.py --models neurodetour --classifier gcn --dataname adni --bold_winsize 500 --device cuda:0 
