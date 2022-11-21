from helper import *
from torch.utils.data import Dataset
import numpy as np
from torch.utils import data
import torch.nn.functional as F
import random


class TrainDataset(Dataset):
    """
    Training Dataset class.

    Parameters
    ----------
    triples:	The triples used for training the model
    params:		Parameters for the experiments

    Returns
    -------
    A training Dataset class instance used by DataLoader
    """
    def __init__(self, triples, params, target_num = None):
        self.triples = triples
        self.p = params
        self.target_num = target_num

    def __len__(self):
        return len(self.triples)

    # 如何取batch中的一条数据
    def __getitem__(self, idx):
        source,object = list(self.triples.keys())[idx]
        label = self.triples[(source,object)] #label为[s,o]对应的所有object
        source, label = torch.LongTensor([source]), np.int32(list(label))
        trp_label = self.get_label(label) # 格式为所有实体的one_hot向量

        if self.p.lbl_smooth != 0.0:
            trp_label = (1.0 - self.p.lbl_smooth) * trp_label + (1.0 / self.target_num)

        return source, trp_label

    @staticmethod
    # 如何取一个batch的数据
    def collate_fn(data):
        #print(data)
        source = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        return source, trp_label

    def get_label(self, label):
        y = np.zeros([self.target_num], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)

class TrainNegativeDataset(Dataset):
    """
    Training Dataset class.

    Parameters
    ----------
    triples:	The triples used for training the model
    params:		Parameters for the experiments

    Returns
    -------
    A training Dataset class instance used by DataLoader
    """

    def __init__(self, triples, params, target_num = None):
        self.triples = triples
        self.p = params
        self.target_num = target_num

    def __len__(self):
        return len(self.triples)

    # 如何取batch中的一条数据
    def __getitem__(self, idx):
        #source 为[batch_size,1]
        #pos_target 为[batch_size,1]
        #neg_target 为[batch_size,K]
        source, object = list(self.triples.keys())[idx]
        pos_target = torch.LongTensor([object])
        neg_target = self.get_neg_ent(self.triples[(source,object)],self.target_num,self.p.K) # 取K个负例子
        source = torch.LongTensor([source])

        return source, pos_target, neg_target

    @staticmethod
    # 如何取一个batch的数据
    def collate_fn(data):
        source = torch.stack([_[0] for _ in data], dim=0)
        pos_target = torch.stack([_[1] for _ in data], dim=0)
        neg_target = torch.stack([_[2] for _ in data], dim=0)
        return source, pos_target, neg_target

    def get_neg_ent(self, all_pos_ent, target_num, K):
        sample_list = [i for i in range(target_num)]
        neg_entity = torch.LongTensor(np.random.choice(np.array(sample_list, dtype=int),K,replace = False))
        return neg_entity



class TestDataset(Dataset):
    """
    Evaluation Dataset class.

    Parameters
    ----------
    triples:	The triples used for evaluating the model
    params:		Parameters for the experiments

    Returns
    -------
    An evaluation Dataset class instance used by DataLoader for model evaluation
    """

    def __init__(self, triples, params, target_num):
        self.triples = triples
        self.p = params
        self.target_num = target_num

    def __len__(self):
        return len(self.triples)

    # 如何取batch中的一条数据
    def __getitem__(self, idx):
        source, object = list(self.triples.keys())[idx]
        label = self.triples[(source,object)]
        source, object, label = torch.LongTensor([source]), torch.LongTensor([object]), np.int32(list(label))
        trp_label = self.get_label(label) # 格式为所有实体的one_hot向量

        return source, trp_label, object  # 返回(s\r\o), 及(s,r)对应的所有o的label

    @staticmethod
    # 如何取一个batch的数据
    def collate_fn(data):
        #print(data)
        source = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        object = torch.stack([_[2] for _ in data], dim=0)
        return source, trp_label, object

    def get_label(self, label):
        y = np.zeros([self.target_num], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)
