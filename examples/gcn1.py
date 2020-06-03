import os.path as osp
import argparse

import numpy as np
from scipy import sparse as spsp
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Reddit
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_sparse import SparseTensor
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

class RandomGraph(InMemoryDataset):
    def __init__(self, n):
        n_edges = m = n * 10
        in_feats = 100
        n_classes = 10

        row = np.random.choice(n, m)
        col = np.random.choice(n, m)
        spm = spsp.coo_matrix((np.ones(len(row)), (row, col)), shape=(n, n))

        features = torch.ones((n, in_feats))
        labels = torch.LongTensor(np.random.choice(n_classes, n))
        train_mask = np.ones(shape=(n))
        val_mask = np.ones(shape=(n))
        test_mask = np.ones(shape=(n))
        if hasattr(torch, 'BoolTensor'):
            train_mask = torch.BoolTensor(train_mask)
            val_mask = torch.BoolTensor(val_mask)
            test_mask = torch.BoolTensor(test_mask)
        else:
            train_mask = torch.ByteTensor(train_mask)
            val_mask = torch.ByteTensor(val_mask)
            test_mask = torch.ByteTensor(test_mask)

        self.edge_attr = torch.FloatTensor(spm.data)
        indices = np.vstack((spm.row, spm.col))
        indices = torch.LongTensor(indices)
        self.edge_index = indices
        self.x = features
        self.y = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.in_feats = in_feats
        self.n_classes = n_classes
        self.adj = SparseTensor(row=self.edge_index[0], col=self.edge_index[1], value=self.edge_attr)

    def to(self, device):
        self.x = self.x.cuda()
        self.y = self.y.cuda()
        self.train_mask = self.train_mask.cuda()
        self.val_mask = self.val_mask.cuda()
        self.test_mask = self.test_mask.cuda()
        self.edge_index = self.edge_index.cuda()
        self.edge_attr = self.edge_attr.cuda()
        self.adj = self.adj.cuda()
        return self

data = RandomGraph(10000)
torch.cuda.set_device(0)

class GCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.root_weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.root_weight)
        zeros(self.bias)

    def forward(self, x, adj):
        out = adj.matmul(x, reduce="mean") @ self.weight
        out = out + self.bias
        return out

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.in_feats, 16)
        self.conv2 = GCNConv(16, data.n_classes)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self):
        x, adj = data.x, data.adj
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
train_time = 0
for epoch in range(1, 201):
    start = time.time()
    train()
    train_time += time.time() - start
    #train_acc, val_acc, tmp_test_acc = test()
    #if val_acc > best_val_acc:
    #    best_val_acc = val_acc
    #    test_acc = tmp_test_acc
    #log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    #print(log.format(epoch, train_acc, best_val_acc, test_acc))
print('train time per epoch:', train_time / 200)
