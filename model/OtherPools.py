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
from torch_geometric.nn import GCNConv, dense_diff_pool
from torch_sparse import SparseTensor

max_nodes = 150

class DiffPool(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_out_nodes):
        super(DiffPool, self).__init__()

        self.num_out_nodes = num_out_nodes
        self.gnn1_pool = GCNConv(in_channels, num_out_nodes)
        #self.gnn1_embed = GCNConv(in_channels, hidden_channels, out_channels)

    def forward(self, x, adj, index = None):
        adj = SparseTensor(row=adj[1], col=adj[0], value=None, is_sorted=False)
        s = self.gnn1_pool(x, adj) #[n_individual, n_group]
        #turn index tensor into index matrix
        s_matrix = torch.zeros((x.size(0), self.num_out_nodes))
        row_idx = torch.arange(0,x.size(0))
        s_matrix[row_idx, index] = 1 #only real affiliation relation
        s = torch.mul(s, s_matrix) #[n_individual, n_group]
        #substitute 0 to -9e10
        s = torch.where(s != 0, s, -9e10*torch.ones_like(s))
        #softmax, 列和为1
        s = torch.softmax(s, dim=-1)
        x = torch.matmul(s.t(), x) #[ n_group, n_individual] * [n_individual, dim] = [n_group, dim]
        return x
