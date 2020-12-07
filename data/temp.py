import pandas as pd
import numpy as np
import os

os.getcwd()
target = pd.read_csv('data/wikipedia/crocodile/musae_crocodile_target.csv')
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
target.to_csv('data/wikipedia/crocodile/musae_crocodile_target_label.csv', index=False)

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
b = F.one_hot(torch.arange(0, 10) % 3)
b.shape
input = torch.tensor([[1,2,4,5],[4,3,2,9]])
embedding_matrix = torch.rand(10, 3)
embedding_matrix
F.embedding(input, embedding_matrix)
c = F.one_hot(a, num_classes=10)

a = torch.Tensor([[1, 2, 3], [1, 2, 3]]).view(-1, 2)
b = torch.Tensor([[2, 1]]).view(2, -1)
print(a)
print(a.size())

print(b)
print(b.size())
a.t()*b
a.t()
b

torch.sum(a, dim=1)
class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

import matplotlib.pyplot as plt
import numpy as np
opts = [NoamOpt(100, 1, 400, None),
        NoamOpt(100, 1, 4000, None),
        NoamOpt(100, 1, 8000, None)]
plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
plt.legend(["512:4000", "512:8000", "256:4000"])


import torch.nn as nn
input1 = torch.randn(1, 128)
input2 = torch.randn(100, 128)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(input1, input2)