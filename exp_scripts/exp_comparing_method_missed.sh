

# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bnt --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:0 --node_attr FC > logs/bnt_gcn_ukb_statfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bnt --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:0 --node_attr FC > logs/bnt_gcn_ukb_dynfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bnt --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:0 --node_attr FC --atlas Gordon_333 > logs/bnt_gcn_ukb_statfcFC_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bnt --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:0 --node_attr FC --atlas Gordon_333 > logs/bnt_gcn_ukb_dynfcFC_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bnt --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:0 --node_attr BOLD > logs/bnt_gcn_ukb_statfcBOLD_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bnt --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:0 --node_attr BOLD > logs/bnt_gcn_ukb_dynfcBOLD_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bnt --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:0 --node_attr BOLD --atlas Gordon_333 > logs/bnt_gcn_ukb_statfcBOLD_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models bnt --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:0 --node_attr BOLD --atlas Gordon_333 > logs/bnt_gcn_ukb_dynfcBOLD_gordon_${dt}.log



# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models braingnn --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:0 --node_attr BOLD --atlas D_160 > logs/braingnn_gcn_oasis_statfcBOLD_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models braingnn --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:0 --node_attr BOLD --atlas Gordon_333 > logs/braingnn_gcn_hcpa_dynfcBOLD_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models braingnn --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:0 --node_attr FC --atlas Gordon_333 > logs/braingnn_gcn_hcpa_dynfcFC_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models braingnn --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:0 --node_attr BOLD > logs/braingnn_gcn_ukb_statfcBOLD_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models braingnn --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:0 --node_attr BOLD > logs/braingnn_gcn_ukb_dynfcBOLD_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models braingnn --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:0 --node_attr BOLD --atlas Gordon_333 > logs/braingnn_gcn_ukb_statfcBOLD_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models braingnn --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:0 --node_attr BOLD --atlas Gordon_333 > logs/braingnn_gcn_ukb_dynfcBOLD_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models braingnn --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:0 --node_attr FC > logs/braingnn_gcn_ukb_statfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models braingnn --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:0 --node_attr FC > logs/braingnn_gcn_ukb_dynfcFC_aal_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models braingnn --classifier gcn --dataname ukb --bold_winsize 500 --epochs 30 --max_patience 10 --device cuda:0 --node_attr FC --atlas Gordon_333 > logs/braingnn_gcn_ukb_statfcFC_gordon_${dt}.log
dt=$(date '+%d-%m-%Y-%H-%M-%S');
python trainval.py --models braingnn --classifier gcn --dataname ukb --bold_winsize 100 --epochs 30 --max_patience 10 --device cuda:1 --node_attr FC --atlas Gordon_333 > logs/braingnn_gcn_ukb_dynfcFC_gordon_${dt}.log



# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models transformer --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:0 --node_attr BOLD --atlas Gordon_333 > logs/transformer_gcn_hcpa_statfcBOLD_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models transformer --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:0 --node_attr BOLD --atlas Gordon_333 > logs/transformer_gcn_hcpa_dynfcBOLD_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models transformer --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:0 --node_attr BOLD > logs/transformer_gcn_ukb_statfcBOLD_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models transformer --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:0 --node_attr BOLD > logs/transformer_gcn_ukb_dynfcBOLD_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models transformer --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:0 --node_attr BOLD --atlas Gordon_333 > logs/transformer_gcn_ukb_statfcBOLD_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models transformer --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:0 --node_attr BOLD --atlas Gordon_333 > logs/transformer_gcn_ukb_dynfcBOLD_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models transformer --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:0 --node_attr FC > logs/transformer_gcn_ukb_statfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models transformer --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:0 --node_attr FC > logs/transformer_gcn_ukb_dynfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models transformer --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:0 --node_attr FC --atlas Gordon_333 > logs/transformer_gcn_ukb_statfcFC_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models transformer --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:0 --node_attr FC --atlas Gordon_333 > logs/transformer_gcn_ukb_dynfcFC_gordon_${dt}.log



# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models graphormer --classifier gcn --dataname hcpa --bold_winsize 500 --device cuda:0 --node_attr FC --atlas Gordon_333 > logs/graphormer_gcn_hcpa_statfcFC_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models graphormer --classifier gcn --dataname hcpa --bold_winsize 100 --device cuda:0 --node_attr FC --atlas Gordon_333 > logs/graphormer_gcn_hcpa_dynfcFC_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models graphormer --classifier gcn --dataname oasis --bold_winsize 100 --device cuda:0 --node_attr BOLD --atlas D_160 > logs/graphormer_gcn_oasis_dynfcBOLD_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models graphormer --classifier gcn --dataname oasis --bold_winsize 100 --device cuda:0 --node_attr FC --atlas D_160 > logs/graphormer_gcn_oasis_dynfcFC_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models graphormer --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:0 --node_attr BOLD > logs/graphormer_gcn_ukb_statfcBOLD_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models graphormer --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:0 --node_attr BOLD > logs/graphormer_gcn_ukb_dynfcBOLD_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models graphormer --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:0 --node_attr BOLD --atlas Gordon_333 > logs/graphormer_gcn_ukb_statfcBOLD_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models graphormer --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:0 --node_attr BOLD --atlas Gordon_333 > logs/graphormer_gcn_ukb_dynfcBOLD_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models graphormer --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:0 --node_attr FC > logs/graphormer_gcn_ukb_statfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models graphormer --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:0 --node_attr FC > logs/graphormer_gcn_ukb_dynfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models graphormer --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:0 --node_attr FC --atlas Gordon_333 > logs/graphormer_gcn_ukb_statfcFC_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models graphormer --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:0 --node_attr FC --atlas Gordon_333 > logs/graphormer_gcn_ukb_dynfcFC_gordon_${dt}.log


# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models nagphormer --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:0 --epochs 30 --max_patience 10 --node_attr BOLD > logs/nagphormer_gcn_ukb_statfcBOLD_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models nagphormer --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:0 --epochs 30 --max_patience 10 --node_attr BOLD > logs/nagphormer_gcn_ukb_dynfcBOLD_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models nagphormer --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:0 --epochs 30 --max_patience 10 --node_attr BOLD --atlas Gordon_333 > logs/nagphormer_gcn_ukb_statfcBOLD_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models nagphormer --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:0 --epochs 30 --max_patience 10 --node_attr BOLD --atlas Gordon_333 > logs/nagphormer_gcn_ukb_dynfcBOLD_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models nagphormer --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:0 --epochs 30 --max_patience 10 --node_attr FC > logs/nagphormer_gcn_ukb_statfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models nagphormer --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:0 --epochs 30 --max_patience 10 --node_attr FC > logs/nagphormer_gcn_ukb_dynfcFC_aal_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models nagphormer --classifier gcn --dataname ukb --bold_winsize 500 --device cuda:0 --epochs 30 --max_patience 10 --node_attr FC --atlas Gordon_333 > logs/nagphormer_gcn_ukb_statfcFC_gordon_${dt}.log
# dt=$(date '+%d-%m-%Y-%H-%M-%S');
# python trainval.py --models nagphormer --classifier gcn --dataname ukb --bold_winsize 100 --device cuda:0 --epochs 30 --max_patience 10 --node_attr FC --atlas Gordon_333 > logs/nagphormer_gcn_ukb_dynfcFC_gordon_${dt}.log

