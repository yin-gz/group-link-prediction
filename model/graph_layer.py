from helper import *
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, RGCNConv, HGTConv
from torch_scatter import scatter_add,scatter
import math
from typing import Union, Tuple, Optional, Any
from torch import Tensor
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)


class GCN(torch.nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels, params=None):
        super(GCN, self).__init__()

        self.p = params
        self.layer = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize = False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize = False))

        self.dropout = self.p.dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, edge_index, x_init, edge_type = None):
        x = x_init
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels, params=None):
        super(GAT, self).__init__()

        self.p = params
        self.layer = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels))

        self.dropout = self.p.dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, edge_index, x_init, edge_type = None):
        x = x_init
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels, params=None):
        super(GraphSAGE, self).__init__()

        self.p = params
        self.layer = num_layers

        self.convs = torch.nn.ModuleList()

        self.convs.append(
            SAGEConv(in_channels, hidden_channels))
        if self.layer > 1:
            for _ in range(self.layer - 2):
                self.convs.append(
                    SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(
                SAGEConv(hidden_channels, out_channels))

        self.dropout = self.p.dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, edge_index, x_init, edge_type = None):
        x = x_init
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class HGT(nn.Module):
    def __init__(self, node_type_dict, edge_type_dict, params=None):
        super().__init__()

        self.p = params
        self.layer = self.p.gcn_layer
        self.convs = torch.nn.ModuleList()

        self.convs.append(
            HGTConv(self.p.init_dim, self.p.gcn_dim, metadata = (node_type_dict.keys(), edge_type_dict.keys())))
        if self.layer > 1:
            for _ in range(self.layer - 2):
                self.convs.append(
                    HGTConv(self.p.gcn_dim, self.p.gcn_dim, metadata = (node_type_dict.keys(), edge_type_dict.keys())))
            self.convs.append(
                HGTConv(self.p.gcn_dim, self.p.embed_dim, metadata = (node_type_dict.keys(), edge_type_dict.keys())))

    def forward(self, node_type_dict, edge_type_dict):
        """
        Args:
            node_type_dict: {'node_type':type_node_tensor[n_type_node, emb]}
            edge_type_dict: {’rel_type('type1','to','type2')‘:edge_tensor[2, type_node_num]}
        """
        for conv in self.convs:
            node_type_dict = conv(node_type_dict, edge_type_dict)

        return node_type_dict

class RGCN(torch.nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels, params=None):
        super(RGCN, self).__init__()

        self.p = params
        self.layer = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_channels, 4))
        for _ in range(num_layers - 2):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, 4))
        self.convs.append(RGCNConv(hidden_channels, out_channels, 4))

        self.dropout = self.p.dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, edge_index, x_init, edge_type):
        x = x_init
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_type)
        return x