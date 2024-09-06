import os
import csv
import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import random
import matplotlib.pyplot as plt
import h5py
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW
from torch.nn import BCEWithLogitsLoss
from random import shuffle
from torch.utils.data import Sampler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, average_precision_score
from tqdm import tqdm
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='trains a neural network on ProtTrans embeddings')

    parser.add_argument('-train_embedding_file', metavar='--train_embedding_file', type=str, help='enter full path to directory containing train embeddings')
    parser.add_argument('-train_annotation_file', metavar='--train_annotation_file', type=str, help='enter path to annotation train file')
    parser.add_argument('-valid_embedding_file', metavar='--valid_embedding_file', type=str, help='enter full path to directory containing validation embeddings')
    parser.add_argument('-valid_annotation_file', metavar='--valid_annotation_file', type=str, help='enter path to annotation validation file')

    parser.add_argument('-batch_size', metavar='--batch_size', type=int, help='select batch size')
    parser.add_argument('-n_epochs', metavar='--n_epochs', type=int, help='max number epochs to train for')
    parser.add_argument('-lr', metavar='--lr', type=float, help='AdamW optimizer learning rate')
    parser.add_argument('-weight_decay', metavar='--weight_decay', type=float, help='AdamW optimizer weight_decay')
    parser.add_argument('-early_stopping', metavar='--early_stopping', type=int, help='number of epochs without improvement before stopping')

    parser.add_argument('-boundary_weight', metavar='--boundary_weight', type=float, help='upweighting of minority boundary class')

    parser.add_argument('-output_dir', metavar='--output_dir', type=str, help='where to save model checkpoints and training plots')

    return parser


def parse_domain(dom_str):
    dom_list = [[it[0], list(map(int, it[1].split(':')))] for it in (dom.split(';') for dom in dom_str.split('|'))]
    return dom_list

def dom_list_to_linker_tensor(dom_list_, prot_len):
    dom_list = [dom for dom in dom_list_ if not ('m' in dom[0])]
    labels = torch.zeros(prot_len).int()
    for dom in dom_list:
        labels[max(dom[1][0]-1-10,0):dom[1][0]-1+11] += 1
        labels[dom[1][1]-1-10:min(dom[1][1]-1+11, prot_len)] += 1
    return (labels>0.5).int()

class ProtDomDataset(Dataset):
    def __init__(self, h5_file, annot_file):
        self.emb_dict = h5_file
        self.df = pd.read_csv(annot_file, header=None)

        self.uniprot_ids = self.df[0]
        self.domain_annot = self.df[1]
        self.protein_lens = self.df[2]
        self.n_bounds = self.df[1].apply(lambda l: [dom for dom in parse_domain(l) if 'm' not in dom[0]][0][1][0])
        self.c_bounds = self.df[1].apply(lambda l: [dom for dom in parse_domain(l) if 'm' not in dom[0]][-1][1][1])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        length = self.df.iloc[idx, 2]
        emb = torch.tensor(self.emb_dict[self.uniprot_ids[idx]][:])
        labels = dom_list_to_linker_tensor(parse_domain(self.df.iloc[idx, 1]), length)
        n_bound = self.n_bounds.iloc[idx]
        c_bound = self.c_bounds.iloc[idx]
        return emb, labels, length, n_bound, c_bound

def collate_fn(batch):

    # batch is list of (embedding, label, length) tuples
    # embedding is (L,D)-dim tensor, label is (L)-dim tensor and length is scalar
    embs, lbls, lengths, n_bounds, c_bounds = zip(*batch)

    # sort sequences in descending length
    sorted_idx = sorted(range(len(lengths)), key=lambda k: -lengths[k])

    # pad sequences
    padded_seqs = pad_sequence(embs)[:,sorted_idx,:].permute((1,0,2))
    padded_lbls = pad_sequence(lbls)[:,sorted_idx].permute((1,0))
    lengths = torch.tensor(lengths)[sorted_idx]
    n_bounds = torch.tensor(n_bounds)[sorted_idx]
    c_bounds = torch.tensor(c_bounds)[sorted_idx]

    # prepare mask (mostly for CNN)
    padding_mask = torch.zeros_like(padded_lbls)
    for idx, item in enumerate(lengths):
        padding_mask[idx, :item] = 1

    # padded_seqs is (N,max_L,D)-dim tensor, padded_lbls is (N,max_L)-dim tensor, lengths is N-dim tensor, padding_mask is (N,max_L)-dim tensor
    return padded_seqs, padded_lbls, lengths, padding_mask, n_bounds, c_bounds


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 1)
        
    def forward(self, X, padding_mask):
        out = self.fc(X).squeeze(2)
        return out


def eval_fn(model, valid_loader, boundary_weight):

    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    for idx, batch in enumerate(valid_loader):
        embs = batch[0].to(device) #N,L,D
        lbls = batch[1] #N,L
        lens = batch[2] #N
        mask = batch[3].to(device) #N,L
        n_bounds = batch[4] #N
        c_bounds = batch[5] #N

        with torch.no_grad():
            outputs = model(embs, mask).to('cpu') #N,L
            mask = mask.to('cpu') #N,L

            if idx == 0:
                pred_list = outputs[mask.bool()]
                lbl_list = lbls[mask.bool()]
            else:
                pred_list = torch.cat((pred_list, outputs[mask.bool()]))
                lbl_list = torch.cat((lbl_list, lbls[mask.bool()]))

            for i in range(len(lens)):
                if idx == 0 and i == 0:
                    no_ext_pred_list = outputs[i][mask[i].bool()][n_bounds[i]-1+11:c_bounds[i]-1-10]
                    no_ext_lbl_list = lbls[i][mask[i].bool()][n_bounds[i]-1+11:c_bounds[i]-1-10]
                else:
                    no_ext_pred_list = torch.cat((no_ext_pred_list, outputs[i][mask[i].bool()][n_bounds[i]-1+11:c_bounds[i]-1-10]))
                    no_ext_lbl_list = torch.cat((no_ext_lbl_list, lbls[i][mask[i].bool()][n_bounds[i]-1+11:c_bounds[i]-1-10]))
    
    proba_list = nn.Sigmoid()(pred_list)
    no_ext_proba_list = nn.Sigmoid()(no_ext_pred_list)

    mets = dict()
    mets['accuracy'] = accuracy_score(lbl_list, pred_list>0)
    mets['precision'] = precision_score(lbl_list, pred_list>0)
    mets['no_ext_precision'] = precision_score(no_ext_lbl_list, no_ext_pred_list>0)
    mets['recall'] = recall_score(lbl_list, pred_list>0)
    mets['no_ext_recall'] = recall_score(no_ext_lbl_list, no_ext_pred_list>0)
    mets['f1'] = f1_score(lbl_list, pred_list>0)
    mets['no_ext_f1'] = f1_score(no_ext_lbl_list, no_ext_pred_list>0)
    mets['auroc'] = roc_auc_score(lbl_list, proba_list)
    mets['auprc'] = average_precision_score(lbl_list, proba_list)
    mets['no_ext_auprc'] = average_precision_score(no_ext_lbl_list, no_ext_proba_list)
    mets['mcc'] = matthews_corrcoef(lbl_list, pred_list>0)
    mets['no_ext_mcc'] = matthews_corrcoef(no_ext_lbl_list, no_ext_pred_list>0)
    mets['no_ext_precision'] = precision_score(no_ext_lbl_list, no_ext_pred_list>0)
    mets['bce_loss'] = BCEWithLogitsLoss(pos_weight=torch.tensor([boundary_weight]))(pred_list, lbl_list.float()).item()

    model.train()

    return mets


def train_fn(model, train_loader, valid_loader, N_epochs, l_rate, weight_decay ,early_stopping, output_dir, boundary_weight):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    losses = []
    valid_accuracy = []
    valid_precision = []
    valid_recall = []
    valid_f1 = []
    valid_auroc = []
    valid_mcc = []
    valid_auprc = []
    valid_bce = []
    valid_mcc_no_ext = []

    criterion = BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([boundary_weight]).to(device))

    bias_params = [p for name, p in model.named_parameters() if 'bias' in name]
    others = [p for name, p in model.named_parameters() if 'bias' not in name]
    opt = AdamW([{'params': others},
                {'params': bias_params, 'weight_decay': 0}], weight_decay=weight_decay, lr=l_rate)

    with open(f'{output_dir}/metrics.csv', 'w') as fOut:
        fOut.write('accuracy,precision,recall,f1,auroc,auprc,mcc,bce_loss\n')

    for idx, epoch in enumerate(range(N_epochs)):
        loop = tqdm(train_loader)
        for batch in loop:

            embs = batch[0].to(device)
            lbls = batch[1].to(device)
            lens = batch[2].to(device)
            mask = batch[3].to(device)

            opt.zero_grad()

            outputs = model(embs, mask).squeeze()

            loss = (criterion(outputs, lbls.float())*mask).sum() / (len(lens) * 322.65)
            loss.backward()

            opt.step()

            losses.append(loss.item())

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

        with torch.no_grad():
            mets = eval_fn(model, valid_loader, boundary_weight)
            valid_accuracy.append(mets['accuracy'])
            valid_precision.append(mets['precision'])
            valid_recall.append(mets['recall'])
            valid_f1.append(mets['f1'])
            valid_mcc.append(mets['mcc'])
            valid_auroc.append(mets['auroc'])
            valid_auprc.append(mets['auprc'])
            valid_bce.append(mets['bce_loss'])
            valid_mcc_no_ext.append(mets['no_ext_mcc'])
            
        model.train()

        
        torch.save({
            'optimizer': opt.state_dict(),
            'model': model.state_dict(),
        }, f'{output_dir}/model_epoch_{idx}')

        with open(f'{output_dir}/metrics.csv', 'a') as fOut:
            fOut.write(f'{mets["accuracy"]},{mets["precision"]},{mets["recall"]},{mets["f1"]},{mets["auroc"]},{mets["auprc"]},{mets["mcc"]},{mets["bce_loss"]}\n')
        print(f'\nPrecision:{mets["precision"]}, Recall:{mets["recall"]}, F1:{mets["f1"]}, AUPRC:{mets["auprc"]}, MCC:{mets["mcc"]}')
        print(f'\nNoExtPrecision:{mets["no_ext_precision"]}, NoExtRecall:{mets["no_ext_recall"]}, NoExtF1:{mets["no_ext_f1"]}, NoExtAUPRC:{mets["no_ext_auprc"]}, NoExtMCC:{mets["no_ext_mcc"]}')

        # Early stopping
        if idx > early_stopping:
            if max(valid_mcc_no_ext[-early_stopping-1:]) < valid_mcc_no_ext[-early_stopping-2]:
                break
    
    print(valid_accuracy)
    print(valid_precision)
    print(valid_recall)
    print(valid_f1)
    print(valid_auroc)
    print(valid_auprc)
    print(valid_mcc)
    print(valid_bce)

    plt.plot(losses)
    plt.savefig(f'{output_dir}/training_loss')


def run(args):
    train_embedding_file = args.train_embedding_file
    train_annotation_file = args.train_annotation_file
    valid_embedding_file = args.valid_embedding_file
    valid_annotation_file = args.valid_annotation_file

    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    n_epochs = args.n_epochs
    early_stopping = args.early_stopping

    boundary_weight = args.boundary_weight

    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with h5py.File(train_embedding_file) as train_embs:
        with h5py.File(valid_embedding_file) as valid_embs:

            train_set = ProtDomDataset(h5_file=train_embs,
                                       annot_file=train_annotation_file)
            valid_set = ProtDomDataset(h5_file=valid_embs,
                                       annot_file=valid_annotation_file)

            #train_sampler = BySequenceLengthSampler(train_set.protein_lens, batch_size)

            train_loader = DataLoader(train_set, batch_size=batch_size, 
                                shuffle=True,
                                num_workers=4, 
                                drop_last=False, pin_memory=False, collate_fn=collate_fn)
            valid_loader = DataLoader(valid_set, batch_size=batch_size, 
                                num_workers=4, 
                                drop_last=False, pin_memory=False, collate_fn=collate_fn)
    
            model = Network()

            train_fn(model=model,
                     train_loader=train_loader,
                     valid_loader=valid_loader,
                     N_epochs=n_epochs,
                     l_rate = lr,
                     weight_decay=weight_decay,
                     early_stopping = early_stopping,
                     output_dir = output_dir,
                     boundary_weight = boundary_weight)

def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()