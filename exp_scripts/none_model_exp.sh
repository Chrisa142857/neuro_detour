
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:2 > logs/none_gcn_hcpa_statfc_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:2 > logs/none_gcn_hcpa_dynfc_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:2 --atlas Gordon_333 > logs/none_gcn_hcpa_statfc_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:2 --atlas Gordon_333 > logs/none_gcn_hcpa_dynfc_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:2 > logs/none_gcn_ukb_statfc_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:2 > logs/none_gcn_ukb_dynfc_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:2 --atlas Gordon_333 > logs/none_gcn_ukb_statfc_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:2 --atlas Gordon_333 > logs/none_gcn_ukb_dynfc_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname adni --bold_winsize 500 --device cuda:2 > logs/none_gcn_adni_statfc_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname adni --bold_winsize 100 --device cuda:2 > logs/none_gcn_adni_dynfc_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:2 --atlas D_160 > logs/none_gcn_oasis_statfc_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname oasis --bold_winsize 100 --device cuda:2 --atlas D_160 > logs/none_gcn_oasis_dynfc_${dt}.log


dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname hcpa --bold_winsize 500 --device cuda:2 > logs/none_mlp_hcpa_statfc_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname hcpa --bold_winsize 100 --device cuda:2 > logs/none_mlp_hcpa_dynfc_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname hcpa --bold_winsize 500 --device cuda:2 --atlas Gordon_333 > logs/none_mlp_hcpa_statfc_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname hcpa --bold_winsize 100 --device cuda:2 --atlas Gordon_333 > logs/none_mlp_hcpa_dynfc_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname ukb --bold_winsize 500 --device cuda:2 > logs/none_mlp_ukb_statfc_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname ukb --bold_winsize 100 --device cuda:2 > logs/none_mlp_ukb_dynfc_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname ukb --bold_winsize 500 --device cuda:2 --atlas Gordon_333 > logs/none_mlp_ukb_statfc_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname ukb --bold_winsize 100 --device cuda:2 --atlas Gordon_333 > logs/none_mlp_ukb_dynfc_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname adni --bold_winsize 500 --device cuda:2 > logs/none_mlp_adni_statfc_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname adni --bold_winsize 100 --device cuda:2 > logs/none_mlp_adni_dynfc_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname oasis --bold_winsize 500 --device cuda:2 --atlas D_160 > logs/none_mlp_oasis_statfc_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname oasis --bold_winsize 100 --device cuda:2 --atlas D_160 > logs/none_mlp_oasis_dynfc_${dt}.log

