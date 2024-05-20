
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:1 --adj_type FC --node_attr FC > logs/none_gcn_hcpa_statfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:1 --adj_type FC --node_attr FC > logs/none_gcn_hcpa_dynfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:1 --adj_type FC --node_attr FC --atlas Gordon_333 > logs/none_gcn_hcpa_statfcFC_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:1 --adj_type FC --node_attr FC --atlas Gordon_333 > logs/none_gcn_hcpa_dynfcFC_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models none --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:1 --adj_type FC --node_attr FC > logs/none_gcn_ukb_statfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models none --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:1 --adj_type FC --node_attr FC > logs/none_gcn_ukb_dynfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models none --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:1 --adj_type FC --node_attr FC --atlas Gordon_333 > logs/none_gcn_ukb_statfcFC_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models none --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:1 --adj_type FC --node_attr FC --atlas Gordon_333 > logs/none_gcn_ukb_dynfcFC_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname adni --bold_winsize 500 --device cuda:1 --adj_type FC --node_attr FC > logs/none_gcn_adni_statfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname adni --bold_winsize 100 --device cuda:1 --adj_type FC --node_attr FC > logs/none_gcn_adni_dynfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:1 --adj_type FC --node_attr FC --atlas D_160 > logs/none_gcn_oasis_statfcFC_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier gcn --dataname oasis --bold_winsize 100 --device cuda:1 --adj_type FC --node_attr FC --atlas D_160 > logs/none_gcn_oasis_dynfcFC_${dt}.log


dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname hcpa --bold_winsize 500 --device cuda:1 --adj_type FC --node_attr FC > logs/none_mlp_hcpa_statfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname hcpa --bold_winsize 100 --device cuda:1 --adj_type FC --node_attr FC > logs/none_mlp_hcpa_dynfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname hcpa --bold_winsize 500 --device cuda:1 --adj_type FC --node_attr FC --atlas Gordon_333 > logs/none_mlp_hcpa_statfcFC_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname hcpa --bold_winsize 100 --device cuda:1 --adj_type FC --node_attr FC --atlas Gordon_333 > logs/none_mlp_hcpa_dynfcFC_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models none --classifier mlp --dataname ukb --bold_winsize 500 --device cuda:1 --adj_type FC --node_attr FC > logs/none_mlp_ukb_statfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models none --classifier mlp --dataname ukb --bold_winsize 100 --device cuda:1 --adj_type FC --node_attr FC > logs/none_mlp_ukb_dynfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models none --classifier mlp --dataname ukb --bold_winsize 500 --device cuda:1 --adj_type FC --node_attr FC --atlas Gordon_333 > logs/none_mlp_ukb_statfcFC_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models none --classifier mlp --dataname ukb --bold_winsize 100 --device cuda:1 --adj_type FC --node_attr FC --atlas Gordon_333 > logs/none_mlp_ukb_dynfcFC_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname adni --bold_winsize 500 --device cuda:1 --adj_type FC --node_attr FC > logs/none_mlp_adni_statfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname adni --bold_winsize 100 --device cuda:1 --adj_type FC --node_attr FC > logs/none_mlp_adni_dynfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname oasis --bold_winsize 500 --device cuda:1 --adj_type FC --node_attr FC --atlas D_160 > logs/none_mlp_oasis_statfcFC_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models none --classifier mlp --dataname oasis --bold_winsize 100 --device cuda:1 --adj_type FC --node_attr FC --atlas D_160 > logs/none_mlp_oasis_dynfcFC_${dt}.log

