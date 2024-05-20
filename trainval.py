from datasets import dataloader_generator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from models import brain_net_transformer, neuro_detour, brain_gnn, brain_identity, bolt, graphormer, nagphormer, vanilla_model
from models.classifier import Classifier
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SGConv
from tqdm import trange, tqdm
import torch.optim as optim
import torch.nn as nn
import torch, math
import argparse, os
import numpy as np
from datetime import datetime

MODEL_BANK = {
    'neurodetour': neuro_detour.DetourTransformer,
    'neurodetourSingleFC': neuro_detour.DetourTransformerSingleFC,
    'neurodetourSingleSC': neuro_detour.DetourTransformerSingleSC,
    'bnt': brain_net_transformer.BrainNetworkTransformer,
    'braingnn': brain_gnn.Network,
    'bolt': bolt.get_BolT,
    'graphormer': graphormer.Graphormer,
    'nagphormer': nagphormer.TransformerModel,
    'transformer': vanilla_model.Transformer,
    'gcn': vanilla_model.GCN,
    'sage': vanilla_model.SAGE,
    'sgc': vanilla_model.SGC,
    'none': brain_identity.Identity
}
CLASSIFIER_BANK = {
    'mlp': nn.Linear,
    'gcn': GCNConv,
    'gat': GATConv,
    'sage': SAGEConv,
    'sgc': SGConv
}
DATA_TRANSFORM = {
    'neurodetour': None,
    'neurodetourSingleFC': None,
    'neurodetourSingleSC': None,
    'bnt': None,
    'braingnn': None,
    'bolt': None,
    'graphormer': graphormer.ShortestDistance(),
    'nagphormer': nagphormer.NAGdataTransform(),
    'transformer': None,
    'gcn': None,
    'sage': None,
    'sgc': None,
    'none': None
}
ATLAS_ROI_N = {
    'AAL_116': 116,
    'Gordon_333': 333,
    'Shaefer_100': 100,
    'Shaefer_200': 200,
    'Shaefer_400': 400,
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
    parser.add_argument('--models', type=str, default = 'neurodetour')
    parser.add_argument('--classifier', type=str, default = 'gcn')
    parser.add_argument('--max_patience', type=int, default = 30)
    parser.add_argument('--hiddim', type=int, default = 768)
    parser.add_argument('--lr', type=float, default = 0.001)
    parser.add_argument('--atlas', type=str, default = 'AAL_116')
    parser.add_argument('--dataname', type=str, default = 'hcpa')
    parser.add_argument('--testname', type=str, default = 'None')
    parser.add_argument('--node_attr', type=str, default = 'SC')
    parser.add_argument('--adj_type', type=str, default = 'FC')
    parser.add_argument('--bold_winsize', type=int, default = 500)
    parser.add_argument('--nlayer', type=int, default = 1)
    parser.add_argument('--nhead', type=int, default = 2)
    parser.add_argument('--classifier_aggr', type=str, default = 'learn')
    parser.add_argument('--savemodel', action='store_true')
    parser.add_argument('--decay', type=float, default=0,
                        help='Weight decay (default: 0)')
    parser.add_argument('--device', type=str, default = 'cuda:0')
    # parser.add_argument('--detour_type', type=str, default = 'node')
    # parser.add_argument('--detour_k', type=int, default = 4)
    args = parser.parse_args()
    print(args)
    expdate = str(datetime.now())
    expdate = expdate.replace(':','-').replace(' ', '-').replace('.', '-')
    device = args.device
    hiddim = args.hiddim
    nclass = DATA_CLASS_N[args.dataname]
    dataset = None
    # Initialize lists to store evaluation metrics
    accuracies = []
    f1_scores = []
    prec_scores = []
    rec_scores = []
    taccuracies = []
    tf1_scores = []
    tprec_scores = []
    trec_scores = []
    node_sz = ATLAS_ROI_N[args.atlas]
    # if args.models != 'neurodetour':
    transform = None
    dek, pek = 0, 0
    if args.node_attr != 'BOLD':
        input_dim = node_sz
    else:
        input_dim = args.bold_winsize
    transform = DATA_TRANSFORM[args.models]
    testset = args.testname
    if args.savemodel:
        mweight_fn = f'model_weights/{args.models}_{args.atlas}_boldwin{args.bold_winsize}_{args.adj_type}{args.node_attr}'
        os.makedirs(mweight_fn, exist_ok=True)
    for i in range(5):
        dataloaders = dataloader_generator(batch_size=args.batch_size, nfold=i, dataset=dataset, 
                                                                 node_attr=args.node_attr, adj_type=args.adj_type, transform=transform, dname=args.dataname, testset=testset,
                                                                 fc_winsize=args.bold_winsize, atlas_name=args.atlas)
        if len(dataloaders) == 3:
            train_loader, val_loader, dataset = dataloaders
        else:
            train_loader, val_loader, dataset, test_loader, testset = dataloaders
        model = MODEL_BANK[args.models](node_sz=node_sz, out_channel=hiddim, in_channel=input_dim, batch_size=args.batch_size, device=device, nlayer=args.nlayer, heads=args.nhead).to(device)
        print(sum([p.numel() for p in model.parameters()]))
        exit()
        classifier = Classifier(CLASSIFIER_BANK[args.classifier], hiddim, nclass=nclass, node_sz=node_sz if args.models!='braingnn' else braingnn_nodesz(node_sz, model.ratio), aggr=args.classifier_aggr).to(device)
        optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=args.decay) 
        # print(optimizer)
        best_f1 = 0
        patience = 0
        for epoch in (pbar := trange(1, args.epochs+1, desc='Epoch')):
            train(model, classifier, device, train_loader, optimizer)
            acc, prec, rec, f1 = eval(model, classifier, device, val_loader)
            pbar.set_description(f'Accuracy: {acc:.6f}, F1 Score: {f1:.6f}, Epoch')
            if f1 >= best_f1:
                if f1 > best_f1: 
                    patience = 0
                else:
                    patience += 1
                best_f1 = f1
                best_acc = acc
                best_prec = prec
                best_rec = rec
                best_state = model.state_dict()
                best_cls_state = classifier.state_dict()
                if args.savemodel:
                    torch.save(model.state_dict(), f'{mweight_fn}/fold{i}_{expdate}.pt')
            else:
                patience += 1
            if patience > args.max_patience: break
        
        accuracies.append(best_acc)
        f1_scores.append(best_f1)
        prec_scores.append(best_prec)
        rec_scores.append(best_rec)
        print(f'Accuracy: {best_acc}, F1 Score: {best_f1}, Prec: {best_prec}, Rec: {best_rec}')
        if args.testname != 'None':
            model.load_state_dict(best_state)
            classifier.load_state_dict(best_cls_state)
            tacc, tprec, trec, tf1 = eval(model, classifier, device, test_loader, hcpatoukb=args.testname in ['hcpa', 'ukb'])
            print(f'Testset: Accuracy: {tacc}, F1 Score: {tf1}, Prec: {tprec}, Rec: {trec}')
            taccuracies.append(tacc)
            tf1_scores.append(tprec)
            tprec_scores.append(trec)
            trec_scores.append(tf1)

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

    if args.testname != 'None':
        mean_accuracy = sum(taccuracies) / len(taccuracies)
        std_accuracy = torch.std(torch.tensor(taccuracies))
        mean_f1_score = sum(tf1_scores) / len(tf1_scores)
        std_f1_score = torch.std(torch.tensor(tf1_scores))
        mean_prec_score = sum(tprec_scores) / len(tprec_scores)
        std_prec_score = torch.std(torch.tensor(tprec_scores))
        mean_rec_score = sum(trec_scores) / len(trec_scores)
        std_rec_score = torch.std(torch.tensor(trec_scores))
        print(f'Test set: {args.testname}')
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
        optimizer.zero_grad()
        batch = batch.to(device)
        feat = model(batch)
        edge_index = batch.edge_index
        batchid = batch.batch
        if len(feat) == 3:  # brainGnn Selected Topk nodes
            feat, edge_index, batchid = feat
        y = classifier(feat, edge_index, batchid)
        loss = loss_fn(y, batch.y)
        if hasattr(model, 'loss'):
            loss = loss + model.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())
    # print('Train loss', np.mean(losses))

def eval(model, classifier, device, loader, hcpatoukb=False):
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
    if hcpatoukb:
        y_scores[y_scores>1] = 1
        y_true[y_true>1] = 1
    acc = accuracy_score(y_true, y_scores)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores, average='weighted')
    return acc, prec, rec, f1

def braingnn_nodesz(node_sz, ratio):
    if node_sz != 333:
        return math.ceil(node_sz*ratio*ratio)
    else:
        return 31

if __name__ == '__main__': main()