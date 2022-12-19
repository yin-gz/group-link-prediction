import torch
import torch.nn as nn
from helper import *

class AttributeEmb(nn.Module):
    """
    Embedding layers for discrete attribute.
    """
    def __init__(self, node_type_dict, embedding_dim, num, att_each_dim = 2):
        super(AttributeEmb, self).__init__()
        self.att_embeddings = nn.ModuleList()
        self.embedding_dim= embedding_dim
        self.node_type_dict= node_type_dict
        self.att_each_dim = att_each_dim
        self.num = num

        for type,content in node_type_dict.items():
            if 'emb' not in type:
                self.att_embeddings.append(nn.Embedding(len(set(content)), att_each_dim))
        for i, emb_layer in enumerate(self.att_embeddings):
            emb_layer.reset_parameters()

    def forward(self, attribute_dict):
        out_x = []
        key_list = list(attribute_dict.keys())
        j = 0
        for i, type in enumerate(key_list):
            if 'emb' not in type:
                index_tensor = torch.LongTensor(attribute_dict[type]).cuda()
                out_x.append(self.att_embeddings[j](index_tensor))
                j += 1
            else:
                emb_tensor = torch.Tensor(attribute_dict[type]).cuda()
                out_x.append(emb_tensor)
        return torch.cat(out_x, -1)