from ndo_calculator import *
from train_ms_cnn_mask_termini_training import *
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import h5py
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='trains a neural network on ProtTrans embeddings')

    parser.add_argument('-test_embedding_file', metavar='--test_embedding_file', type=str, help='enter full path to directory containing test embeddings')
    parser.add_argument('-test_annotation_file', metavar='--test_annotation_file', type=str, help='enter path to annotation test file')

    parser.add_argument('-model_checkpoint', metavar='--model_checkpoint', type=str, help='path to model checkpoint')

    parser.add_argument('-thresh', metavar='--thresh', type=float, help='threshold for predicting boundary')

    return parser


def run(args):
    test_embedding_file = args.test_embedding_file
    test_annotation_file = args.test_annotation_file
    checkpoint_file = args.model_checkpoint
    thresh = args.thresh

    model = Network(kernel_size_1=5, kernel_size_2=17, num_layers=5, num_channels=128, hidden_dim=128, conv_dropout=0.2, fc_dropout=0.4)
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.training

    with h5py.File(test_embedding_file) as embed_file:
        with torch.no_grad():
            df = pd.read_csv(test_annotation_file, header=None)
            ndos = []
            for i in range(len(df)):
                prot_id, dom_str, prot_len = df.loc[i,:]
                prot_len = int(prot_len)
                dom_list = [dom for dom in parse_domain(dom_str) if 'm' not in dom[0]]
                emb = torch.tensor(embed_file[prot_id][:]).unsqueeze(0)
                labels = dom_list_to_linker_tensor(dom_list, prot_len)
                padding_mask = torch.ones((1, prot_len))

                probs = nn.Sigmoid()(model(emb, padding_mask)).squeeze()

                preds = probs > thresh
                #preds = zeros_like(probs)

                pred_list = preds_to_doms(preds)
                pred_list = add_link_to_doms(pred_list, prot_len)

                tar_list = [dom[1] for dom in dom_list]
                tar_list = add_link_to_doms(tar_list, prot_len)

                ndos.append(ndo(tar_list, pred_list))

    print(sum(ndos)/len(ndos))
    print(np.std(ndos))

def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()