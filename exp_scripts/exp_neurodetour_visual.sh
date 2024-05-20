

python trainval.py --models neurodetour --classifier gcn --dataname hcpa --bold_winsize 500 --savemodel --device cuda:3 --node_attr FC --nhead 8 --atlas Gordon_333

python trainval.py --models neurodetour --classifier gcn --dataname ukb --bold_winsize 500 --savemodel --device cuda:3 --node_attr FC --nhead 8 --atlas Gordon_333

python trainval.py --models neurodetour --classifier gcn --dataname adni --bold_winsize 500 --savemodel --device cuda:3 --node_attr FC --nhead 8

python trainval.py --models neurodetour --classifier gcn --dataname oasis --bold_winsize 500 --savemodel --device cuda:3 --node_attr FC --nhead 8 --atlas D_160
