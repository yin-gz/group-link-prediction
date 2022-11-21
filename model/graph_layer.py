from helper import *
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, RGCNConv, FastRGCNConv, HGTConv, HypergraphConv, HeteroConv, HANConv, GINConv, RGATConv
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

def constant(value: Any, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)
def zeros(value: Any):
    constant(value, 0.)


class GCN(torch.nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels, params=None):
        super(GCN, self).__init__()

        self.p = params
        self.layer = num_layers
        if self.p.gcn_node_concat:
            self.concat_liner = nn.Linear(in_channels + self.layer * hidden_channels, out_channels)

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize = False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        if self.p.gcn_node_concat:
            self.convs.append(GCNConv(hidden_channels, out_channels, normalize = False))
        else:
            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize = False))

        self.dropout = self.p.dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, edge_index, x_init, edge_type = None):
        '''
        x = x_init
        x_list = [x]
        for conv in self.convs[:-1]:
            last_layer_x = x
            x = conv(x, edge_index)
            if self.p.gcn_node_residual:
                x = x + last_layer_x
            x = F.relu(x)
            x_list.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.p.gcn_node_residual:
            x = self.convs[-1](x, edge_index) + x
        else:
            x = self.convs[-1](x, edge_index)
        x_list.append(x)

        if self.p.gcn_node_concat:
            x = torch.cat(x_list,dim = 1)
            x = self.concat_liner(x)  # (n_nodes,n_layer*emb)*(n_layer*emb,emb)
        '''
        x = x_init
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GAT(torch.nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels, params=None):
        super(GAT, self).__init__()

        self.p = params
        self.layer = num_layers
        if self.p.gcn_node_concat:
            self.concat_liner = nn.Linear(in_channels + self.layer * hidden_channels, out_channels)

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels))
        if self.p.gcn_node_concat:
            self.convs.append(GATConv(hidden_channels, out_channels))
        else:
            self.convs.append(GATConv(hidden_channels, hidden_channels))

        self.dropout = self.p.dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, edge_index, x_init, edge_type = None):
        x = x_init
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GIN(torch.nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels, params=None):
        super(GIN, self).__init__()

        self.p = params
        self.layer = num_layers
        if self.p.gcn_node_concat:
            self.concat_liner = nn.Linear(in_channels + self.layer * hidden_channels, out_channels)

        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GINConv(hidden_channels, hidden_channels))
        if self.p.gcn_node_concat:
            self.convs.append(GINConv(hidden_channels, out_channels))
        else:
            self.convs.append(GINConv(hidden_channels, hidden_channels))

        self.dropout = self.p.dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, edge_index, x_init, edge_type = None):
        x = x_init
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
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
        x_list = []
        for conv in self.convs[:-1]:
            last_layer_x = x
            x = conv(x, edge_index)
            if self.p.gcn_node_residual:
                x = x + last_layer_x
            x = F.relu(x)
            x_list.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.p.gcn_node_residual:
            x = self.convs[-1](x, edge_index) + x
        else:
            x = self.convs[-1](x, edge_index)
        x_list.append(x)
        if self.p.gcn_node_concat:
            x = torch.cat(x_list,dim = 1)
        return x

class HGT(nn.Module):
    def __init__(self, node_type_dict, edge_type_dict, params=None):
        super().__init__()

        self.p = params
        self.layer = self.p.gcn_layer
        self.convs = torch.nn.ModuleList()

        self.convs.append(
            #meda 为(([node_type1,node_type2,...],[edge_type1,edge_type2,...]))
            HGTConv(self.p.init_dim, self.p.gcn_dim, metadata = (node_type_dict.keys(), edge_type_dict.keys())))
        if self.layer > 1:
            for _ in range(self.layer - 2):
                self.convs.append(
                    HGTConv(self.p.gcn_dim, self.p.gcn_dim, metadata = (node_type_dict.keys(), edge_type_dict.keys())))
            self.convs.append(
                HGTConv(self.p.gcn_dim, self.p.embed_dim, metadata = (node_type_dict.keys(), edge_type_dict.keys())))

    def forward(self, node_type_dict, edge_type_dict):

        for conv in self.convs:
            #输入的x为dict,{'node_type':type_node_tensor(n_type_node,emb)}, 节点与边对应时的编号还是按总的顺序
            #输入的y为dict,{’rel_type('type1','to','type2')‘:edge(2,type_node_num)}
            node_type_dict = conv(node_type_dict, edge_type_dict)

        return node_type_dict

class MultiHetero(nn.Module):
    def __init__(self, node_type_dict, edge_type_dict, params=None):
        super().__init__()

        self.p = params
        self.layer = self.p.gcn_layer
        self.convs = torch.nn.ModuleList()
        edge_dict_layer = {}
        for key in edge_type_dict.keys():
            if key[0] != key[-1]:
                edge_dict_layer[key] = GATConv((-1,-1), self.p.embed_dim)
            else:
                edge_dict_layer[key] = GCNConv(-1, self.p.embed_dim)
        print('层信息 ',edge_dict_layer)
        hetero_conv = HeteroConv(edge_dict_layer, aggr='sum')

        for _ in range(self.layer):
            self.convs.append(hetero_conv)

    def forward(self, node_type_dict, edge_type_dict):

        for conv in self.convs:
            #输入的x为dict,{'node_type':type_node_tensor(n_type_node,emb)}, 节点与边对应时的编号还是按总的顺序
            #输入的y为dict,{’rel_type('type1','to','type2')‘:edge(2,type_node_num)}
            node_type_dict = conv(node_type_dict, edge_type_dict)

        return node_type_dict['author'], node_type_dict['group']


class RGCN(torch.nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels, params=None):
        super(RGCN, self).__init__()

        self.p = params
        self.layer = num_layers
        if self.p.gcn_node_concat:
            self.concat_liner = nn.Linear(in_channels + self.layer * hidden_channels, out_channels)
        self.convs = torch.nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_channels, 4))
        for _ in range(num_layers - 2):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, 4))
        if self.p.gcn_node_concat:
            self.convs.append(RGCNConv(hidden_channels, out_channels, 4))
        else:
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, 4))

        self.dropout = self.p.dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, edge_index, x_init, edge_type):
        x = x_init
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_type)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class RGAT(torch.nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels, params=None):
        super(RGAT, self).__init__()

        self.p = params
        self.layer = num_layers
        if self.p.gcn_node_concat:
            self.concat_liner = nn.Linear(in_channels + self.layer * hidden_channels, out_channels)
        self.convs = torch.nn.ModuleList()
        self.convs.append(RGATConv(in_channels, hidden_channels, 4))
        for _ in range(num_layers - 2):
            self.convs.append(
                RGATConv(hidden_channels, hidden_channels, 4))
        if self.p.gcn_node_concat:
            self.convs.append(RGATConv(hidden_channels, out_channels, 4))
        else:
            self.convs.append(RGATConv(hidden_channels, hidden_channels, 4))

        self.dropout = self.p.dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, edge_index, x_init, edge_type):
        x = x_init
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_type)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x