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
from utils import load_data, accuracy

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

print(os.getcwd())


def sample_subgraph(walks, indices):
    subgraph_walks = defaultdict(lambda: [])
    # may have duplicated keys so use i as the key
    for i, target_node in enumerate(indices):
        paths = walks[int(target_node)]
        for path in paths:
            subgraph_walks[i].append([int(n) for n in path]) # need to convert the node into int

    a = []
    for k, v in subgraph_walks.items():
        v_tensor = torch.tensor(v, dtype=torch.long)
        a.append(v_tensor)

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
        model.train()
        start_time = time()
        # n_train_nodes = len(idx_train)
        shuffled_indices = torch.randperm(n_nodes)

        # scores = []
        # losses_train, losses_val = 0, 0

        # for batch, count in enumerate(range(0, n_train_nodes, args.batch_size)):
        optimizer.zero_grad()
        # train for users nodes
        # indices = shuffled_indices[count:min(count+args.batch_size, n_train_nodes)]
        indices = idx_all[shuffled_indices]
        path_tensor = sample_subgraph(walks_train, indices)
        # target_emb = features[indices]

        # loss
        # item_empty = torch.tensor(item_empty, dtype=torch.long).to(device)
        score = model(indices, path_tensor)
        idx_train_shuffled = torch.zeros_like(idx_train)
        for i, idx in enumerate(idx_train):
            idx_train_shuffled[i] = torch.nonzero(indices == idx)

        score_train = score[idx_train_shuffled]
        label_train = labels[idx_train]
        loss_train = criterion(score_train, label_train)
        loss_train.backward()
        optimizer.step()

        # losses_train += loss_train
        # scores.append(score)

        end_time = time()

        # label = labels[shuffled_indices]
        # scores = torch.cat(scores, dim=0)
        acc_train = accuracy(score_train, label_train)
        # print('********************* start evaluation *********************')
        idx_val_shuffled = torch.zeros_like(idx_val)
        for i, idx in enumerate(idx_val):
            idx_val_shuffled[i] = torch.nonzero(indices == idx)
        loss_val = criterion(score[idx_val_shuffled], labels[idx_val])
        acc_val = accuracy(score[idx_val_shuffled], labels[idx_val])
        # loss_val, acc_val = eval_on_test_data(idx_val, walks_train)
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.4f}'.format(loss_train),
              'acc_train: {:.4f}'.format(acc_train),
              'loss_val: {:.4f}'.format(loss_val.data.item()),
              'acc_val: {:.4f}'.format(acc_val.data.item()),
              'time: {:.4f}s'.format(end_time-start_time))

        # print('********************* end evaluation *********************')
        # best_value, stopping_step, should_stop = early_stopping(loss_val, best_value, stopping_step, flag_step=3)
        if loss_val.detach().cpu().item() < best_value:
            # print('update!')
            torch.save(model.state_dict(), 'output/{}.pkl'.format('best_model'))
            best_value = loss_val.detach().cpu().item()
            best_epoch = epoch
        # if should_stop:
        #     break
    return best_epoch


def eval_on_test_data(test_data):
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
        idx_all = torch.tensor(range(n_nodes), dtype=torch.long).to(device)
        path_tensor = sample_subgraph(walks_train, idx_all)
        score = model(idx_all, path_tensor)
        loss = criterion(score[test_data], labels[test_data])
        acc = accuracy(score[test_data], labels[test_data])
    return loss, acc


if __name__ == '__main__':
    adj, features, labels, idx_train, idx_val, idx_test, edges = load_data()

    n_nodes = adj.shape[0]
    # tmp = torch.unique(labels)
    n_labels = int(torch.unique(labels).size(0))
    fea_dim = features.shape[1]

    # deepwalk
    print('********************* walk for train val test *********************')
    walks_train = deepwalks(edges, undirected=True, number_walks=args.number_walks,
                            walk_length=args.walk_length, seed=args.seed)
    # walks_val = deepwalks(edges, idx_val, undirected=True, number_walks=args.number_walks,
    #                       walk_length=args.walk_length, seed=args.seed)
    # walks_test = deepwalks(edges, idx_test, undirected=True, number_walks=args.number_walks,
    #                        walk_length=args.walk_length, seed=args.seed)

    check_walk_length(walks_train, args.walk_length)
    # check_walk_length(walks_val, args.walk_length)
    # check_walk_length(walks_test, args.walk_length)

    # some nodes dont have paths since their edges are not in train
    # idx_train = torch.tensor(list(walks_train.keys()), dtype=torch.long).to(device)
    # idx_val = torch.tensor(list(walks_val.keys()), dtype=torch.long).to(device)
    # idx_test = torch.tensor(list(walks_test.keys()), dtype=torch.long).to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    features = features.to(device)
    labels = labels.to(device)
    tmp = labels[:1000].detach().cpu().numpy()
    unique, counts = np.unique(tmp, return_counts=True)
    print(np.asarray((unique, counts)).T)
    # conv model
    # n_nodes, n_paths, dim, feature, d_model, d_inner, d_k, d_v, n_head, n_layers
    model = Transformer(n_nodes, args.number_walks, n_labels, fea_dim, features.to(device), 100, 100, 100, 100, 8, args.n_layers).to(device)

    # model = recomm(transformer, 2, fea_dim, n_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-3, lr=args.max_lr)
    criterion = nn.CrossEntropyLoss()

    best_epoch = train()

    # start test
    model.load_state_dict(torch.load('output/{}.pkl'.format('best_model')))
    loss_test, acc_test = eval_on_test_data(idx_test)
    print("best epoch:", best_epoch)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.detach().cpu().item()),
          "accuracy= {:.4f}".format(acc_test.detach().cpu().item()))

