import torch
from torch_geometric.nn import GCNConv
from math import ceil
from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool

'''
codes from 'https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial16/Tutorial16.ipynb'
'''

import os.path as osp
from math import ceil

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import GCNConv, dense_diff_pool

max_nodes = 150

class DiffPool(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_out_nodes):
        super(DiffPool, self).__init__()

        self.num_out_nodes = num_out_nodes
        self.gnn1_pool = GCNConv(in_channels, hidden_channels, num_out_nodes)
        self.gnn1_embed = GCNConv(in_channels, hidden_channels, out_channels)

    def forward(self, x, adj, index = None, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        #turn index tensor into index matrix
        s_matrix = torch.zeros((adj.size(-1), self.num_out_nodes))
        row_idx = torch.arange(0,3)
        col_idx = index
        s_matrix[row_idx, col_idx] = 1
        s = torch.mul(s, s_matrix)
        #x = self.gnn1_embed(x, adj, mask)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        x = x.mean(dim=1)
        return x
