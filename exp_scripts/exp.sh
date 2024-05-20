
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bnt --classifier mlp --dataname hcpa --bold_winsize 500 --device cuda:0 > logs/bnt_mlp_hcpa_statfc_${dt}.log
# python trainval.py --models bnt --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:0 > logs/bnt_gcn_hcpa_statfc_${dt}.log

# python trainval.py --models bnt --classifier mlp --dataname hcpa --bold_winsize 500 --device cuda:0 > logs/bnt_mlp_hcpa_statfc_${dt}.log
# python trainval.py --models bnt --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:0 > logs/bnt_gcn_hcpa_statfc_${dt}.log

# python trainval.py --models bnt --classifier gcn --dataname adni --bold_winsize 500 --device cuda:1
# python trainval.py --models bnt --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:1

# python trainval.py --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:1 
# python trainval.py --models neurodetour --classifier gcn --dataname adni --bold_winsize 500 --device cuda:0 
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:0 > logs/bnt_gcn_hcpa_statfc_aal_${dt}.log &
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:1 > logs/bnt_gcn_hcpa_dynfc_aal_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --dataname adni --bold_winsize 500 --device cuda:0 > logs/bnt_gcn_adni_statfc_${dt}.log &
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --dataname adni --bold_winsize 100 --device cuda:0 > logs/bnt_gcn_adni_dynfc_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:0 --atlas D_160 > logs/bnt_gcn_oasis_statfc_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --dataname oasis --bold_winsize 100 --device cuda:0 --atlas D_160 > logs/bnt_gcn_oasis_dynfc_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:1 --atlas Gordon_333 > logs/bnt_gcn_hcpa_statfc_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:2 --atlas Gordon_333 > logs/bnt_gcn_hcpa_dynfc_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname adni --bold_winsize 500 --device cuda:0 > logs/ndt_gcn_adni_statfc_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname adni --bold_winsize 100 --device cuda:1 > logs/ndt_gcn_adni_dynfc_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname oasis --bold_winsize 100 --device cuda:3 --atlas D_160 > logs/ndt_gcn_oasis_dynfc_${dt}.log


dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:1 --atlas AAL_116 > logs/ndtNode_gcn_hcpa_aal_statfc_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:1 --atlas Gordon_333 > logs/ndtNode_gcn_hcpa_gordon_statfc_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:0 --atlas Gordon_333 > logs/ndtEdge_gcn_hcpa_gordon_statfc_${dt}.log



dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models braingnn --classifier gcn --dataname adni --bold_winsize 500 --device cuda:2 > logs/braingnn_gcn_adni_statfc_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models braingnn --classifier gcn --dataname adni --bold_winsize 100 --device cuda:3 > logs/braingnn_gcn_adni_dynfc_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models braingnn --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:3 --atlas D_160 > logs/braingnn_gcn_oasis_statfc_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models braingnn --classifier gcn --dataname oasis --bold_winsize 100 --device cuda:3 --atlas D_160 > logs/braingnn_gcn_oasis_dynfc_${dt}.log


dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models braingnn --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:0 > logs/braingnn_gcn_hcpa_statfc_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models braingnn --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:2 > logs/braingnn_gcn_hcpa_dynfc_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models braingnn --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:3 --atlas Gordon_333 > logs/braingnn_gcn_hcpa_statfc_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models braingnn --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:1 --atlas Gordon_333 > logs/braingnn_gcn_hcpa_dynfc_gordon_${dt}.log



dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models braingnn --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:2 > logs/braingnn_gcn_ukb_statfc_aal_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models braingnn --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:2 > logs/braingnn_gcn_ukb_dynfc_aal_${dt}.log



dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models braingnn --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:1 --atlas Gordon_333 > logs/braingnn_gcn_ukb_statfc_gordon_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models braingnn --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:3 --atlas Gordon_333 > logs/braingnn_gcn_ukb_dynfc_gordon_${dt}.log


dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:0 > logs/bnt_gcn_ukb_statfc_aal_${dt}.log

dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models bnt --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:2 > logs/bnt_gcn_ukb_dynfc_aal_${dt}.log

