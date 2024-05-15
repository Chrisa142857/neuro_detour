

python trainval.py --models neurodetour --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:3 --epochs 30 --max_patience 10 --node_attr BOLD --atlas D_160 --testname adni --classifier_aggr mean > logs/neurodetour_oasis-adni_gcn_statfcBOLD_${dt}.log
python trainval.py --models neurodetour --classifier gcn --dataname adni --bold_winsize 500 --device cuda:3 --epochs 30 --max_patience 10 --node_attr BOLD --atlas AAL_116 --testname oasis --classifier_aggr mean > logs/neurodetour_adni-oasis_gcn_statfcBOLD_${dt}.log
python trainval.py --models neurodetour --classifier gcn --dataname oasis --bold_winsize 100 --device cuda:3 --epochs 30 --max_patience 10 --node_attr BOLD --atlas D_160 --testname adni --classifier_aggr mean > logs/neurodetour_oasis-adni_gcn_dynfcBOLD_${dt}.log
python trainval.py --models neurodetour --classifier gcn --dataname adni --bold_winsize 100 --device cuda:3 --epochs 30 --max_patience 10 --node_attr BOLD --atlas AAL_116 --testname oasis --classifier_aggr mean > logs/neurodetour_adni-oasis_gcn_dynfcBOLD_${dt}.log



python trainval.py --models none --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:3 --epochs 30 --max_patience 10 --node_attr BOLD --atlas D_160 --testname adni --classifier_aggr mean > logs/none_oasis-adni_gcn_statfcBOLD_${dt}.log
python trainval.py --models none --classifier gcn --dataname adni --bold_winsize 500 --device cuda:3 --epochs 30 --max_patience 10 --node_attr BOLD --atlas AAL_116 --testname oasis --classifier_aggr mean > logs/none_adni-oasis_gcn_statfcBOLD_${dt}.log
python trainval.py --models none --classifier gcn --dataname oasis --bold_winsize 100 --device cuda:3 --epochs 30 --max_patience 10 --node_attr BOLD --atlas D_160 --testname adni --classifier_aggr mean > logs/none_oasis-adni_gcn_dynfcBOLD_${dt}.log
python trainval.py --models none --classifier gcn --dataname adni --bold_winsize 100 --device cuda:3 --epochs 30 --max_patience 10 --node_attr BOLD --atlas AAL_116 --testname oasis --classifier_aggr mean > logs/none_adni-oasis_gcn_dynfcBOLD_${dt}.log


python trainval.py --models none --classifier mlp --dataname oasis --bold_winsize 500 --device cuda:3 --epochs 30 --max_patience 10 --node_attr BOLD --atlas D_160 --testname adni --classifier_aggr mean > logs/none_oasis-adni_mlp_statfcBOLD_${dt}.log
python trainval.py --models none --classifier mlp --dataname adni --bold_winsize 500 --device cuda:3 --epochs 30 --max_patience 10 --node_attr BOLD --atlas AAL_116 --testname oasis --classifier_aggr mean > logs/none_adni-oasis_mlp_statfcBOLD_${dt}.log
python trainval.py --models none --classifier mlp --dataname oasis --bold_winsize 100 --device cuda:3 --epochs 30 --max_patience 10 --node_attr BOLD --atlas D_160 --testname adni --classifier_aggr mean > logs/none_oasis-adni_mlp_dynfcBOLD_${dt}.log
python trainval.py --models none --classifier mlp --dataname adni --bold_winsize 100 --device cuda:3 --epochs 30 --max_patience 10 --node_attr BOLD --atlas AAL_116 --testname oasis --classifier_aggr mean > logs/none_adni-oasis_mlp_dynfcBOLD_${dt}.log


python trainval.py --models graphormer --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:3 --epochs 30 --max_patience 10 --node_attr BOLD --atlas D_160 --testname adni --classifier_aggr mean > logs/graphormer_oasis-adni_gcn_statfcBOLD_${dt}.log
python trainval.py --models graphormer --classifier gcn --dataname adni --bold_winsize 500 --device cuda:3 --epochs 30 --max_patience 10 --node_attr BOLD --atlas AAL_116 --testname oasis --classifier_aggr mean > logs/graphormer_adni-oasis_gcn_statfcBOLD_${dt}.log
python trainval.py --models graphormer --classifier gcn --dataname oasis --bold_winsize 100 --device cuda:3 --epochs 30 --max_patience 10 --node_attr BOLD --atlas D_160 --testname adni --classifier_aggr mean > logs/graphormer_oasis-adni_gcn_dynfcBOLD_${dt}.log
python trainval.py --models graphormer --classifier gcn --dataname adni --bold_winsize 100 --device cuda:3 --epochs 30 --max_patience 10 --node_attr BOLD --atlas AAL_116 --testname oasis --classifier_aggr mean > logs/graphormer_adni-oasis_gcn_dynfcBOLD_${dt}.log


python trainval.py --models nagphormer --classifier gcn --dataname oasis --bold_winsize 500 --device cuda:3 --epochs 30 --max_patience 10 --node_attr BOLD --atlas D_160 --testname adni --classifier_aggr mean > logs/nagphormer_oasis-adni_gcn_statfcBOLD_${dt}.log
python trainval.py --models nagphormer --classifier gcn --dataname adni --bold_winsize 500 --device cuda:3 --epochs 30 --max_patience 10 --node_attr BOLD --atlas AAL_116 --testname oasis --classifier_aggr mean > logs/nagphormer_adni-oasis_gcn_statfcBOLD_${dt}.log
python trainval.py --models nagphormer --classifier gcn --dataname oasis --bold_winsize 100 --device cuda:3 --epochs 30 --max_patience 10 --node_attr BOLD --atlas D_160 --testname adni --classifier_aggr mean > logs/nagphormer_oasis-adni_gcn_dynfcBOLD_${dt}.log
python trainval.py --models nagphormer --classifier gcn --dataname adni --bold_winsize 100 --device cuda:3 --epochs 30 --max_patience 10 --node_attr BOLD --atlas AAL_116 --testname oasis --classifier_aggr mean > logs/nagphormer_adni-oasis_gcn_dynfcBOLD_${dt}.log

