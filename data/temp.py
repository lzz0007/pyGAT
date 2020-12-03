import pandas as pd
import numpy as np
import os

os.getcwd()
target = pd.read_csv('data/wikipedia/chameleon/musae_chameleon_target.csv')
max_target = max(target.target)
min_target = min(target.target)
interval = (max_target-min_target)/5


def f(row):
    if row['target'] <= 300:
        val = 0
    elif 300 < row['target'] <= 1000:
        val = 1
    elif 1000 < row['target'] <= 5000:
        val = 2
    elif 5000 < row['target'] <= 15000:
        val = 3
    else:
        val = 4
    return val


target['label'] = target.apply(f, axis=1)
target.groupby(['label']).agg(['count'])
target.to_csv('data/wikipedia/chameleon/musae_chameleon_target_label.csv', index=False)

import json
dataset = 'chameleon'
feature_path = './data/wikipedia/' + str(dataset) + '/musae_' + str(dataset) + '_features.json'
with open(feature_path) as f:
    features = json.load(f)
tmp = list(features.keys())
i = 0
features[str(i)]
edges = pd.read_csv('./data/wikipedia/' + str(dataset) + '/musae_' + str(dataset) + '_edges.csv')
edges.id1.nunique()
edges = edges.to_numpy()

targets = pd.read_csv('./data/wikipedia/' + str(dataset) + '/musae_' + str(dataset) + '_target_label.csv')
targets = targets.sort_values(by=['id'])
labels = targets.label.to_numpy()

n_nodes = 2277
idx_train = range(int(n_nodes * 0.7))
max(idx_train)
idx_val = range(int(n_nodes * 0.7), int(n_nodes * 0.8))
max(idx_val)
idx_test = range(int(n_nodes * 0.8), int(n_nodes))
max(idx_test)

import torch
import torch.nn.functional as F
a = torch.randint(0, 10, (10, 5, 2))
b = a[0]
c = F.one_hot(a, num_classes=10)
torch.randint(0, 10, ())