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
    Sets the GPU to be used for the run

    Parameters
    ----------
    gpus:           List of GPUs to be used for the run

    Returns
    -------

    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def get_logger(name, log_dir, config_dir):
    """
    Creates a logger object

    Parameters
    ----------
    name:           Name of the logger file
    log_dir:        Directory where logger file needs to be stored
    config_dir:     Directory from where log_config.json needs to be read

    Returns
    -------
    A logger object which writes to both file and stdout

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


def get_combined_results(left_results, right_results):
    results = {}
    count = float(left_results['count'])#mrr/count是在这除的

    results['left_mr'] = round(left_results['mr'] / count, 5)
    results['left_mrr'] = round(left_results['mrr'] / count, 5)
    results['right_mr'] = round(right_results['mr'] / count, 5)
    results['right_mrr'] = round(right_results['mrr'] / count, 5)
    results['mr'] = round((left_results['mr'] + right_results['mr']) / (2 * count), 5)
    results['mrr'] = round((left_results['mrr'] + right_results['mrr']) / (2 * count), 5)

    for k in [9, 49, 99]:
        results['left_hits@{}'.format(k + 1)] = round(left_results['hits@{}'.format(k + 1)] / count, 5)
        results['right_hits@{}'.format(k + 1)] = round(right_results['hits@{}'.format(k + 1)] / count, 5)
        results['hits@{}'.format(k + 1)] = round(
            (left_results['hits@{}'.format(k + 1)] + right_results['hits@{}'.format(k + 1)]) / (2 * count), 5)
    return results


def get_param(shape):
    param = Parameter(torch.Tensor(*shape).cuda());
    xavier_normal_(param.data)
    return param


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def cconv(a, b):
    return torch.irfft(com_mult(torch.rfft(a, 1), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


def ccorr(a, b):
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


def get_metrics(ranking_list, n):
    hits1_list = (ranking_list <= 1).to(torch.float)
    hits3_list = (ranking_list <= 3).to(torch.float)
    hits5_list = (ranking_list <= 5).to(torch.float)
    hits10_list = (ranking_list <= 10).to(torch.float)
    hits20_list = (ranking_list <= 20).to(torch.float)
    mrr_list = 1. / ranking_list.to(torch.float)
    ndcg1_list = torch.log(torch.tensor(2)) / torch.log(ranking_list[(ranking_list <= 1)] + 1)
    ndcg3_list = torch.log(torch.tensor(2)) / torch.log(ranking_list[(ranking_list <= 3)] + 1)
    ndcg5_list = torch.log(torch.tensor(2)) / torch.log(ranking_list[(ranking_list <= 5)] + 1)
    ndcg10_list = torch.log(torch.tensor(2)) / torch.log(ranking_list[(ranking_list <= 10)] + 1)
    ndcg20_list = torch.log(torch.tensor(2)) / torch.log(ranking_list[ranking_list <= 20] + 1)

    return {'hits@1': float(torch.mean(hits1_list).cpu()),
            'hits@3': float(torch.mean(hits3_list).cpu()),
            'hits@5': float(torch.mean(hits5_list).cpu()),
            'hits@10': float(torch.mean(hits10_list).cpu()),
            'hits@20': float(torch.mean(hits20_list).cpu()),
            'mrr': float(torch.mean(mrr_list).cpu()),
            'ndcg@1': float((torch.sum(ndcg1_list) / n).cpu()),
            'ndcg@3': float((torch.sum(ndcg3_list) / n).cpu()),
            'ndcg@5': float((torch.sum(ndcg5_list) / n).cpu()),
            'ndcg@10': float((torch.sum(ndcg10_list) / n).cpu()),
            'ndcg@20': float((torch.sum(ndcg20_list) / n).cpu())}


def eval_pos_neg(y_pred_pos, y_pred_neg):
    # y_pred_pos:[batch*step]
    # y_pred_neg:[batch*step,n_group/100]
    n = y_pred_pos.size(0)
    print('pos ', y_pred_pos)
    print('neg ', y_pred_neg)
    y_pred = torch.cat([y_pred_neg, y_pred_pos.view(-1, 1)], dim=1)
    argsort = torch.argsort(y_pred, dim=1, descending=True)
    ranking_list = torch.nonzero(argsort == y_pred.size(-1) - 1, as_tuple=False)

    ranking_list = ranking_list[:, 1] + 1
    print('pos rank', ranking_list)
    results = get_metrics(ranking_list, n)
    return results