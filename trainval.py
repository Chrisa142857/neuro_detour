from datasets import dataloader_generator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from models import brain_net_transformer, neuro_detour, brain_gnn
from models.classifier import Classifier
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SGConv
from tqdm import trange, tqdm
import torch.optim as optim
import torch.nn as nn
import torch, math
import argparse
import numpy as np
from data_detour import NeuroDetourNode, NeuroDetourEdge

MODEL_BANK = {
    'neurodetour': neuro_detour.DetourTransformer,
    'bnt': brain_net_transformer.BrainNetworkTransformer,
    'braingnn': brain_gnn.Network
}
CLASSIFIER_BANK = {
    'mlp': nn.Linear,
    'gcn': GCNConv,
    'gat': GATConv,
    'sage': SAGEConv,
    'sgc': SGConv
}
ATLAS_ROI_N = {
    'AAL_116': 116,
    'Gordon_333': 333,
    'D_160': 160
}
DATA_CLASS_N = {
    'ukb': 2,
    'hcpa': 4,
    'adni': 2,
    'oasis': 2,
}

def main():
    parser = argparse.ArgumentParser(description='NeuroDetour')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default = 100)
    parser.add_argument('--models', type=str, default = 'braingnn')
    parser.add_argument('--classifier', type=str, default = 'gcn')
    parser.add_argument('--max_patience', type=int, default = 30)
    parser.add_argument('--hiddim', type=int, default = 768)
    parser.add_argument('--lr', type=float, default = 0.001)
    parser.add_argument('--atlas', type=str, default = 'AAL_116')
    parser.add_argument('--dataname', type=str, default = 'hcpa')
    parser.add_argument('--node_attr', type=str, default = 'SC')
    parser.add_argument('--adj_type', type=str, default = 'FC')
    parser.add_argument('--bold_winsize', type=int, default = 500)
    parser.add_argument('--decay', type=float, default=0,
                        help='Weight decay (default: 0)')
    parser.add_argument('--device', type=str, default = 'cuda:0')
    args = parser.parse_args()
    print(args)
    device = args.device
    hiddim = args.hiddim
    nclass = DATA_CLASS_N[args.dataname]
    dataset = None
    # Initialize lists to store evaluation metrics
    accuracies = []
    f1_scores = []
    prec_scores = []
    rec_scores = []
    node_sz = ATLAS_ROI_N[args.atlas]
    # if args.models != 'neurodetour':
    transform = None
    dek, pek = 0, 0
    if args.node_attr != 'BOLD':
        input_dim = node_sz
    else:
        input_dim = args.bold_winsize
    # else:
    if args.models == 'neurodetour':
        transform = NeuroDetourNode(k=5, node_num=node_sz)
        # transform = NeuroDetourEdge(k=4, node_num=node_sz)
        if isinstance(transform, NeuroDetourEdge):
            input_dim = input_dim*2
            dek = transform.k
            pek = transform.PEK*2
        else:
            # input_dim = node_sz
            pek = transform.PEK
            dek = transform.k * 4

    for i in range(5):
        train_loader, val_loader, dataset = dataloader_generator(batch_size=args.batch_size, nfold=i, dataset=dataset, 
                                                                 node_attr=args.node_attr, adj_type=args.adj_type, transform=transform, dname=args.dataname,
                                                                 fc_winsize=args.bold_winsize, atlas_name=args.atlas)
        model = MODEL_BANK[args.models](node_sz=node_sz, out_channel=hiddim, in_channel=input_dim, dek=dek, pek=pek).to(device)
        classifier = Classifier(CLASSIFIER_BANK[args.classifier], hiddim, nclass=nclass, node_sz=node_sz if args.models!='braingnn' else braingnn_nodesz(node_sz, model.ratio)).to(device)
        optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=args.decay) 
        # print(optimizer)
        best_f1 = 0
        patience = 0
        for epoch in (pbar := trange(1, args.epochs+1, desc='Epoch')):
            train(model, classifier, device, train_loader, optimizer)
            acc, prec, rec, f1 = eval(model, classifier, device, val_loader)
            pbar.set_description(f'Accuracy: {acc}, F1 Score: {f1}, Epoch')
            if f1 >= best_f1:
                if f1 > best_f1: 
                    patience = 0
                else:
                    patience += 1
                best_f1 = f1
                best_acc = acc
                best_prec = prec
                best_rec = rec
            else:
                patience += 1
            if patience > args.max_patience: break
        accuracies.append(best_acc)
        f1_scores.append(best_f1)
        prec_scores.append(best_prec)
        rec_scores.append(best_rec)
        print(f'Accuracy: {best_acc}, F1 Score: {best_f1}, Prec: {best_prec}, Rec: {best_rec}')

    # Calculate mean and standard deviation of evaluation metrics
    mean_accuracy = sum(accuracies) / len(accuracies)
    std_accuracy = torch.std(torch.tensor(accuracies))
    mean_f1_score = sum(f1_scores) / len(f1_scores)
    std_f1_score = torch.std(torch.tensor(f1_scores))
    mean_prec_score = sum(prec_scores) / len(prec_scores)
    std_prec_score = torch.std(torch.tensor(prec_scores))
    mean_rec_score = sum(rec_scores) / len(rec_scores)
    std_rec_score = torch.std(torch.tensor(rec_scores))

    print(f'Mean Accuracy: {mean_accuracy}, Std Accuracy: {std_accuracy}')
    print(f'Mean F1 Score: {mean_f1_score}, Std F1 Score: {std_f1_score}')
    print(f'Mean prec Score: {mean_prec_score}, Std prec Score: {std_prec_score}')
    print(f'Mean rec Score: {mean_rec_score}, Std rec Score: {std_rec_score}')

def train(model, classifier, device, loader, optimizer):
    model.train()
    losses = []
    loss_fn = nn.CrossEntropyLoss()
    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        feat = model(batch)
        edge_index = batch.edge_index
        batchid = batch.batch
        if len(feat) == 3:  # brainGnn Selected Topk nodes
            feat, edge_index, batchid = feat
        y = classifier(feat, edge_index, batchid)
        loss = loss_fn(y, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())
    # print('Train loss', np.mean(losses))

def eval(model, classifier, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            feat = model(batch)
            edge_index = batch.edge_index
            batchid = batch.batch
            if len(feat) == 3:  # brainGnn Selected Topk nodes
                feat, edge_index, batchid = feat
            pred = classifier(feat, edge_index, batchid)

        y_true.append(batch.y)
        y_scores.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).detach().cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).numpy().argmax(1)
    acc = accuracy_score(y_true, y_scores)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores, average='weighted')
    return acc, prec, rec, f1

def braingnn_nodesz(node_sz, ratio):
    if node_sz != 333:
        return math.ceil(node_sz*ratio*ratio)
    else:
        return 31

if __name__ == '__main__': main()