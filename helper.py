import numpy as np, sys, os, random, pdb, json, uuid, time, argparse
from pprint import pprint
import logging, logging.config
from collections import defaultdict as ddict
from ordered_set import OrderedSet

# PyTorch related imports
import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from torch.utils.data import DataLoader
from torch.nn import Parameter
from torch_scatter import scatter_add

np.set_printoptions(precision=4)


def set_gpu(gpus):
    """
    Set the GPU to be used for the run.

    Args:
        gpus: List of GPUs to be used for the run

    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def get_logger(name, log_dir, config_dir):
    """
    Create a logger object.

    Args:
        name: Name of the logger file
        log_dir: Directory where logger file needs to be stored
        config_dir: Directory from where log_config.json needs to be read

    Returns:
        A logger object which writes to both file and stdout.

    """
    config_dict = json.load(open(config_dir + 'log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '_').replace(':', '') + '.log'
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)
    return logger

def get_param(shape):
    """
    Initialize the params in the neural networks using xavier_normal.

    Args:
        shape: Shape of the params

    Returns:
        Params after initializing

    """
    param = Parameter(torch.Tensor(*shape).cuda());
    xavier_normal_(param.data)
    return param

def get_metrics(ranking_list, n):
    """
    Calculate the metrics from pos ranking list.

    Args:
        ranking_list: List of the postive instances ranks
        n: Length of the ranking_list

    Returns:
        A dict including results of different metrics

    """
    hits1_list = (ranking_list <= 1).to(torch.float)
    hits5_list = (ranking_list <= 5).to(torch.float)
    hits10_list = (ranking_list <= 10).to(torch.float)
    mrr_list = 1. / ranking_list.to(torch.float)
    ndcg1_list = torch.log(torch.tensor(2)) / torch.log(ranking_list[(ranking_list <= 1)] + 1)
    ndcg5_list = torch.log(torch.tensor(2)) / torch.log(ranking_list[(ranking_list <= 5)] + 1)
    ndcg10_list = torch.log(torch.tensor(2)) / torch.log(ranking_list[(ranking_list <= 10)] + 1)

    return {'hits@1': float(torch.mean(hits1_list).cpu()),
            'hits@5': float(torch.mean(hits5_list).cpu()),
            'hits@10': float(torch.mean(hits10_list).cpu()),
            'mrr': float(torch.mean(mrr_list).cpu()),
            'ndcg@1': float((torch.sum(ndcg1_list) / n).cpu()),
            'ndcg@5': float((torch.sum(ndcg5_list) / n).cpu()),
            'ndcg@10': float((torch.sum(ndcg10_list) / n).cpu())}


def eval_pos_neg(y_pred_pos, y_pred_neg):
    """
    Calculate the pos ranking based on the positive scores and negative scores.

    Args:
        y_pred_pos: [batch*step]
        y_pred_neg: [batch*step,N_neg]

    Returns:
        A dict including results of different metrics

    """
    n = y_pred_pos.size(0)
    #print('pos ', y_pred_pos)
    #print('neg ', y_pred_neg)
    y_pred = torch.cat([y_pred_neg, y_pred_pos.view(-1, 1)], dim=1)
    argsort = torch.argsort(y_pred, dim=1, descending=True)
    ranking_list = torch.nonzero(argsort == y_pred.size(-1) - 1, as_tuple=False)

    ranking_list = ranking_list[:, 1] + 1
    #print('pos rank', ranking_list)
    results = get_metrics(ranking_list, n)
    return results