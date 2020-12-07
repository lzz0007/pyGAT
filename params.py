import argparse
import random

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='gnn cnn')
    parser.add_argument('--cuda', type=int, default=-1,
                        help='Avaiable GPU ID')
    # parser.add_argument('--data_dir', type=str, default='data/graph_amazon_recent_uu_ii.pk',
    #                     help='The address of preprocessed graph.')
    parser.add_argument('--seed', type=int, default=12345,
                        help='seed number')
    # parser.add_argument('--batch_size', type=int, default=32,
    #                     help='Number of output nodes for training')
    parser.add_argument('--n_epoch', type=int, default=1000,
                        help='Number of epoch to run')
    # parser.add_argument('--n_batch', type=int, default=128,
    #                     help='Number of batch (sampled graphs) for each epoch')

    parser.add_argument('--n_hid', type=int, default=100,
                        help='Number of hidden dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention head')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=int, default=0.2,
                        help='Dropout ratio')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Maximum learning rate.')
    # parser.add_argument('--scheduler', type=str, default='cycle',
    #                     help='Name of learning rate scheduler.', choices=['cycle', 'cosine'])
    # parser.add_argument('--clip', type=float, default=0.5,
    #                     help='Gradient Norm Clipping')

    # parser.add_argument('--Ks', nargs='?', default='[5, 10, 20, 50, 200, 500]', help='topK recommendation')

    # params for deepwalk
    parser.add_argument('--number_walks', type=int, default=5,
                        help='Number of output nodes for training')
    parser.add_argument('--walk_length', type=int, default=1,
                        help='Number of output nodes for training')
    parser.add_argument('--pretrain', help='Whether to add pretrain feature', action='store_true')
    parser.add_argument('--best_model', type=str, default='best_model', help='best model file name')
    args = parser.parse_known_args()[0]
    return args


args = parse_args()

