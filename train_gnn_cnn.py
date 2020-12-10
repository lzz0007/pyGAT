import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from random import randint
import argparse
from datetime import datetime
import os
import numpy as np
from time import time
from collections import defaultdict

from gnn_cnn_model.deepwalks import *
from gnn_cnn_model.model import *
from gnn_cnn_model.utils import *

from params import args
from utils import load_data, accuracy, load_data_wiki

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

print(os.getcwd())


def subgraph_tensor(walks, indices):
    a = []
    # may have duplicated keys so use i as the key
    for i, target_node in enumerate(indices):
        paths = walks[int(target_node)]
        b = []
        for path in paths:
            p = [int(n) for n in path]
            b.append(p)
        b_tensor = torch.tensor(b, dtype=torch.long)
        a.append(b_tensor)

    out = torch.stack(a).to(device)
    return out


# train
def train():
    print('********************* start training *********************')
    best_value = 1000000
    stopping_step = 0
    idx_all = torch.tensor(range(n_nodes), dtype=torch.long).to(device)

    best_epoch = 0
    for epoch in np.arange(args.n_epoch) + 1:
        # st = time()
        seed1 = randint(0, 99999)
        seed2 = randint(0, 99999)
        walks_train_1 = deepwalks(edges, undirected=True, number_walks=args.number_walks,
                                walk_length=args.walk_length, seed=seed1)
        walks_train_2 = deepwalks(edges, undirected=True, number_walks=args.number_walks,
                                walk_length=args.walk_length, seed=seed2)
        path_tensor_1 = subgraph_tensor(walks_train_1, idx_all)
        path_tensor_2 = subgraph_tensor(walks_train_2, idx_all)
        # print('sample subgraph time:', time()-st)
        model.train()
        start_time = time()
        n_train_nodes = len(idx_train)
        shuffled_indices = torch.randperm(n_train_nodes)

        scores = []
        losses_train, losses_val = 0, 0

        for batch, count in enumerate(range(0, n_train_nodes, args.batch_size)):
            optimizer.zero_grad()
            # train for users nodes
            indices = shuffled_indices[count:min(count+args.batch_size, n_train_nodes)]
            indices = idx_train[indices]

            # target_emb = features[indices]

            # loss
            # item_empty = torch.tensor(item_empty, dtype=torch.long).to(device)
            tmp = path_tensor_1[indices]
            score, ssl = model(indices, path_tensor_1[indices], path_tensor_2[indices])
            # idx_train_shuffled = torch.zeros_like(idx_train)
            # for i, idx in enumerate(idx_train):
            #     idx_train_shuffled[i] = torch.nonzero(indices == idx)

            loss_train = criterion(score, labels[indices], ssl)
            loss_train.backward()
            optimizer.step()

            losses_train += loss_train
            scores.append(score)

        end_time = time()

        score_train = torch.cat(scores, dim=0)
        acc_train = accuracy(score_train, labels[idx_train][shuffled_indices])
        # loss_val = criterion(score[idx_val], labels[idx_val], ssl[idx_val])
        # acc_val = accuracy(score[idx_val], labels[idx_val])
        loss_val, acc_val = eval_on_test_data(idx_val, seed1)
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.4f}'.format(losses_train),
              'acc_train: {:.4f}'.format(acc_train),
              'loss_val: {:.4f}'.format(loss_val.data.item()),
              'acc_val: {:.4f}'.format(acc_val.data.item()),
              'time: {:.4f}s'.format(end_time-start_time))

        # print('********************* end evaluation *********************')
        loss_val = loss_val.detach().cpu().item()
        if loss_val < best_value:
            # print('update!')
            torch.save(model.state_dict(), 'output/{}.pkl'.format(args.best_model))
            best_epoch = epoch
        best_value, stopping_step, should_stop = early_stopping(loss_val, best_value, stopping_step, flag_step=150)
        if should_stop:
            break
    return best_epoch


def eval_on_test_data(test_data, seed):
    walks_1 = deepwalks(edges, undirected=True, number_walks=args.number_walks,
                      walk_length=args.walk_length, seed=seed)
    walks_2 = deepwalks(edges, undirected=True, number_walks=args.number_walks,
                      walk_length=args.walk_length, seed=seed+100)
    model.eval()
    with torch.no_grad():
        # indices = test_data
        # path_tensor = sample_subgraph(walks, indices)
        # # target_emb = features[indices]
        #
        # # loss
        # score = model(path_tensor, indices)
        # label = labels[indices]
        # loss = criterion(score, label)
        #
        # acc_val = accuracy(score, label)
        path_tensor_1 = subgraph_tensor(walks_1, idx_all)
        path_tensor_2 = subgraph_tensor(walks_2, idx_all)
        score, ssl = model(idx_all, path_tensor_1, path_tensor_2)
        loss = criterion(score[test_data], labels[test_data], ssl[test_data])
        acc = accuracy(score[test_data], labels[test_data])
    return loss, acc


def criterion(score, label, ssl):
    loss = F.cross_entropy(score, label)
    tmp = torch.sum(ssl/10)
    loss = loss + torch.sum(ssl/10)
    return loss


if __name__ == '__main__':
    adj, features, labels, idx_train, idx_val, idx_test, edges = load_data()
    # adj, features, labels, idx_train, idx_val, idx_test, edges = load_data_wiki(dataset='chameleon')

    n_nodes = adj.shape[0]
    # tmp = torch.unique(labels)
    n_labels = int(torch.unique(labels).size(0))
    fea_dim = features.shape[1]

    # deepwalk
    # print('********************* walk for train val test *********************')
    # walks_train = deepwalks(edges, undirected=True, number_walks=args.number_walks,
    #                         walk_length=args.walk_length, seed=args.seed)
    # walks_val = deepwalks(edges, idx_val, undirected=True, number_walks=args.number_walks,
    #                       walk_length=args.walk_length, seed=args.seed)
    # walks_test = deepwalks(edges, idx_test, undirected=True, number_walks=args.number_walks,
    #                        walk_length=args.walk_length, seed=args.seed)

    # check_walk_length(walks_train, args.walk_length)
    # check_walk_length(walks_val, args.walk_length)
    # check_walk_length(walks_test, args.walk_length)

    # some nodes dont have paths since their edges are not in train
    # idx_train = torch.tensor(list(walks_train.keys()), dtype=torch.long).to(device)
    # idx_val = torch.tensor(list(walks_val.keys()), dtype=torch.long).to(device)
    # idx_test = torch.tensor(list(walks_test.keys()), dtype=torch.long).to(device)
    idx_all = torch.tensor(range(n_nodes), dtype=torch.long).to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    features = features.to(device)
    labels = labels.to(device)
    tmp = labels[:int(n_nodes*0.5)].detach().cpu().numpy()
    unique, counts = np.unique(tmp, return_counts=True)
    print(np.asarray((unique, counts)).T)
    # conv model
    # n_nodes, n_paths, dim, feature, d_model, d_inner, d_k, d_v, n_head, n_layers
    model = Transformer(n_nodes, args.number_walks, args.walk_length, n_labels, fea_dim, features.to(device),
                        100, 100, 100, 100, 8, args.n_layers, True).to(device)

    # model = recomm(transformer, 2, fea_dim, n_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-3, lr=args.lr)
    # criterion = nn.CrossEntropyLoss()

    best_epoch = train()

    # start test
    model.load_state_dict(torch.load('output/{}.pkl'.format(args.best_model)))
    loss_test, acc_test = eval_on_test_data(idx_test, args.seed)
    print("best epoch:", best_epoch)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.detach().cpu().item()),
          "accuracy= {:.4f}".format(acc_test.detach().cpu().item()))

    loss_test, acc_test = eval_on_test_data(idx_test, args.seed+100)
    print("best epoch:", best_epoch)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.detach().cpu().item()),
          "accuracy= {:.4f}".format(acc_test.detach().cpu().item()))

