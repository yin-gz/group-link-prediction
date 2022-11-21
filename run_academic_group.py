import json
import random

import numpy as np
import torch

from helper import *
from data_loader import *
import yaml
from model.models import G2GModel, LinkPredictor
from model.AttributeEmb import AttributeEmb
from torch_sparse import SparseTensor
from itertools import combinations
from logger import Logger
from scipy.sparse import csr_matrix
from model.node2vec import Node2Vec
from torch_scatter import scatter
from torch_geometric.data import HeteroData
import wandb
import pickle
import scipy.sparse as ssp

CUDA_LAUNCH_BLOCKING = 1

class Runner(object):
    def __init__(self, params, metrics):
        """
        Constructor of the runner class. Create computational graph and optimizer.

        Args:
            params: List of hyper-parameters of the model
        """
        self.p = params
        self.metrics = metrics
        self.logger = get_logger(self.p.store_name, self.p.log_dir, self.p.config_dir)
        self.logger.info(vars(self.p))
        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data()
        self.model = G2GModel(self.p, num_author=self.author_num, num_group=self.group_num).to(self.device)

        # param to optimize
        param_optimizer = list(self.model.named_parameters())
        if self.p.train_group:
            self.predictor_group = LinkPredictor(self.p.embed_dim, self.p.embed_dim, 1, self.p.predict_layer,
                                                 self.p.dropout, self.p.score_method, self.p.i2g_method,
                                                 self.p.view_num).to(self.device)
            param_optimizer += list(self.predictor_group.named_parameters())
        if self.p.train_author:
            self.predictor_author = LinkPredictor(self.p.embed_dim, self.p.embed_dim, 1, self.p.predict_layer,
                                                self.p.dropout, 'MLP', None, self.p.view_num).to(self.device)
            param_optimizer += list(self.predictor_author.named_parameters())
        if self.p.use_g_attribute:
            self.group_attemb_layer = AttributeEmb(self.group_attribute,
                                                   embedding_dim=self.p.gcn_dim, num=self.group_num,
                                                   att_each_dim=self.p.attr_dim).to(self.device)
            param_optimizer += list(self.group_attemb_layer.named_parameters())
        if self.p.use_a_attribute:
            self.author_attemb_layer = AttributeEmb(self.author_attribute, embedding_dim=self.p.gcn_dim,
                                                    num=self.author_num, att_each_dim=self.p.attr_dim).to(self.device)
            param_optimizer += list(self.author_attemb_layer.named_parameters())

        # optimizer
        if self.p.graph_based == 'HGT' or self.p.optimizer == 'AdamW':
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.p.lr, eps=1e-06)
        else:
            self.optimizer = torch.optim.Adam([p for n, p in param_optimizer], lr=self.p.lr, weight_decay=self.p.l2)

        # scheduler
        if self.p.scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1,
                                                                        patience=1, min_lr=0.000001, verbose=False)
        elif self.p.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                        T_max=self.p.max_epochs * self.p.batch_size,
                                                                        verbose=False)
        elif self.p.scheduler == 'onecycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, pct_start=0.3,
                                                                 anneal_strategy='linear', div_factor=5,
                                                                 final_div_factor=25, max_lr=1e-2,
                                                                 total_steps=self.p.max_epochs * self.p.batch_size + 1)
        else:
            self.scheduler = None

    def load_data(self):
        # 0. pre:fitler group according to self.group_info，generate group_train_set\group_valid_set\group_test_set
        # load from file
        author_set, group_set = OrderedSet(), OrderedSet()
        ag_edge = []
        self.g2a = ddict(set)
        self.a2g = {}
        org_info = {}
        
        if self.p.use_g_attribute:
            f_org = open(self.p.data_dir+self.p.dataset+'aminer_affiliations.txt', 'r', encoding='utf-8')
            for line in f_org.readlines():
                dict = json.loads(line)
                org_info[dict['id']] = dict
        group_co_train_all = json.load(open(self.p.data_dir + self.p.dataset + '/PAG_train.json', 'r', encoding='utf-8'))
        group_co_valid_all = json.load(open(self.p.data_dir + self.p.dataset + '/PAG_valid.json', 'r', encoding='utf-8'))
        group_co_vt = json.load(open(self.p.data_dir + self.p.dataset + '/PAG_test.json', 'r', encoding='utf-8'))
        self.group_info = json.load(open(self.p.data_dir + self.p.dataset + '/group_info.json', 'r', encoding='utf-8'))

        # calculate group size
        group_per_author = [len(set(v['member'])) for k, v in self.group_info.items()]
        print('Avg. group size', np.mean(group_per_author))
        group_list = list(self.group_info.keys())
        random.shuffle(group_list)

        # select partial group (option)
        group_filter_list = set(group_list[:int(len(self.group_info) * self.p.data_size)])
        group_co_train, group_co_valid, group_co_test = {}, {}, {}

        # filter out papers including groups not in group_filter_list
        for (paper_id, coo_info) in group_co_train_all.items():
            flag = 1
            for each_group in coo_info['group_cooperate']:
                if each_group not in group_filter_list:
                    flag = 0
                    break
            if flag == 1:
                group_co_train[paper_id] = coo_info
        for (paper_id, coo_info) in group_co_valid_all.items():
            flag = 1
            for each_group in coo_info['group_cooperate']:
                if each_group not in group_filter_list:
                    flag = 0
                    break
            if flag == 1:
                group_co_valid[paper_id] = coo_info
        for (paper_id, coo_info) in group_co_vt.items():
            flag = 1
            for each_group in coo_info['group_cooperate']:
                if each_group not in group_filter_list:
                    flag = 0
                    break
            if flag == 1:
                group_co_test[paper_id] = coo_info

        # merge the original valid and test (Next, split them according to the group split results)
        group_co_vt = group_co_test.copy()
        group_co_vt.update(json.load(open(self.p.data_dir + self.p.dataset + '/PAG_valid.json', 'r', encoding='utf-8')))


        # 1. load all entities, generate g2a/a2g/group_train_set/group_valid_set/group_test_set
        # load all entities and generate self.g2a, ag_edge
        for (paper_id, coo_info) in group_co_train.items():
            if len(set([each_author[0] for each_author in coo_info['author_cooperate']])) > 1:
                for each_author in coo_info['author_cooperate']:
                    author_set.add(each_author[0])
                    group_set.add(each_author[1])
                    self.g2a[each_author[1]].add(each_author[0])
                    ag_edge.append([each_author[0], each_author[1]])
        print('training paper num', len(group_co_train))
        for (paper_id, coo_info) in group_co_vt.items():
            if len(set([each_author[0] for each_author in coo_info['author_cooperate']])) > 1:
                for each_author in coo_info['author_cooperate']:
                    author_set.add(each_author[0])
                    group_set.add(each_author[1])
                    self.g2a[each_author[1]].add(each_author[0])
                    ag_edge.append([each_author[0], each_author[1]])
        print('test paper num', len(group_co_vt))
        self.author2id = {ent: idx for idx, ent in enumerate(author_set)}
        self.group2id = {ent: idx for idx, ent in enumerate(group_set)}
        self.id2group = {idx: ent for ent, idx in self.group2id.items()}
        self.id2author = {idx: ent for ent, idx in self.author2id.items()}
        self.author_num = len(author_set)
        self.group_num = len(group_set)
        self.p.group_num = self.group_num
        print('author num', self.author_num)
        print('group num', self.group_num)

        # turn g2a/a2g/ag_edge to index format
        self.g2a_new = {}
        for group, author_list in self.g2a.items():
            self.g2a_new[self.group2id[group]] = [self.author2id[author] for author in author_list]
        self.g2a = self.g2a_new
        for group_id, author_list in self.g2a.items():
            for author_id in author_list:
                self.a2g[author_id] = group_id
        self.ag_graph = [[self.author2id[edge[0]], self.group2id[edge[1]]] for edge in ag_edge]

        # split all groups to train/valid/test
        group_set = list(group_set)
        random.shuffle(group_set)
        group_train_set = set(list(group_set)[:int(self.group_num * 0.6)])
        group_valid_set = set(list(group_set)[int(self.group_num * 0.6):int(self.group_num * 0.8)])
        group_test_set = set(list(group_set)[int(self.group_num * 0.8):])
        

        # 2. train: build a_graph\ag_graph\author_train_edge\group_train_edge
        self.author_split_edge = {'train': {}, 'valid': {}, 'test': {}}
        self.group_split_edge = {'train': {}, 'valid': {}, 'test': {}}
        train_group_coo = ddict(list)
        self.a_graph = []
        author_train_edge, group_train_edge = [], []
        in_coo_num, out_coo_num = 0, 0 # count cooperation times in/out between orgs

        for (paper_id, coo_info) in group_co_train.items():
            coo_author_info = coo_info['author_cooperate']
            if len(coo_author_info) > 20:
                continue
            coo_author_set = set([i[0] for i in coo_author_info])
            coo_group_set = set(coo_info['group_cooperate'])
            if len(coo_author_set) > 1:
                for (i, j) in combinations(coo_author_set, 2):
                    try:
                        self.a_graph.append([self.author2id[i], self.author2id[j]])
                        self.a_graph.append([self.author2id[j], self.author2id[i]])
                        author_train_edge.append([self.author2id[i], self.author2id[j]])
                    except:
                        pass
            if len(coo_group_set) > 1:
                for (i, j) in combinations(coo_group_set, 2):
                    org_i = i.split('-')[-1]
                    org_j = j.split('-')[-1]
                    if org_i == org_j:
                        in_coo_num += 1
                    else:
                        out_coo_num += 1
                    if i in group_train_set and j in group_train_set:
                        train_group_coo[i].append(j)
                        group_train_edge.append([self.group2id[i], self.group2id[j]])

        random.shuffle(author_train_edge)
        random.shuffle(group_train_edge)
        print('author train edge', len(author_train_edge))
        print('group train edge', len(group_train_edge))
        self.author_split_edge['train']['edge'] = torch.LongTensor(author_train_edge)  # 只有训练时有用，不对user推荐效果进行测试
        self.group_split_edge['train']['edge'] = torch.LongTensor(group_train_edge)

        # generate group neg edge for training
        '''
        group_train_neg_edge = []
        neg_target_all = np.random.choice(np.array(list(group_train_set)), len(group_train_edge), replace=True)
        for index, edge in enumerate(group_train_edge):
            group_source = edge[0]
            neg_group = self.group2id[neg_target_all[index]]
            group_train_neg_edge.append([group_source, neg_group])
        self.group_split_edge['train']['neg_edge'] = torch.LongTensor(group_train_neg_edge)
        '''

        # calculate degrees of authors
        temp = self.a_graph.copy()
        for i in author_set:
            temp.append([self.author2id[i], self.author2id[i]])
        temp = torch.LongTensor(temp).t()
        self.author_degree = scatter(torch.ones_like(temp[0]), temp[1], dim=0, reduce='sum').to(self.device)  # 计算每个节点的度
        print('avg author degree', torch.mean(self.author_degree.float()))
        self.a_graph = torch.LongTensor(self.a_graph)
        self.A = ssp.csr_matrix((torch.ones_like(self.a_graph[0]), (self.a_graph[0], self.a_graph[1])), shape=(self.author_num, self.author_num))
        self.a_graph = self.a_graph.to(self.device).t()
        self.ag_graph = torch.LongTensor(self.ag_graph).to(self.device).t()

        # 3. valid/build pos/neg edges in train/valid/test splits for groups and authors
        # 3.1 build edges for author cooperation
        author_edge_valid, author_edge_test = [], []
        for (paper_id, coo_info) in group_co_valid.items():
            coo_author_info = coo_info['author_cooperate']
            coo_author_set = set(i[0] for i in coo_author_info)
            if len(coo_author_set) > 1:
                for (i, j) in combinations(coo_author_set, 2):
                    try:
                        author_edge_valid.append([self.author2id[i], self.author2id[j]])
                    except:
                        pass
        for (paper_id, coo_info) in group_co_test.items():
            coo_author_info = coo_info['author_cooperate']
            coo_author_set = set(i[0] for i in coo_author_info)
            if len(coo_author_set) > 1:
                for (i, j) in combinations(coo_author_set, 2):
                    try:
                        author_edge_test.append([self.author2id[i], self.author2id[j]])
                    except:
                        pass
        self.author_split_edge['valid']['edge'] = torch.LongTensor(author_edge_valid)
        self.author_split_edge['valid']['edge_neg'] = torch.randint(0, len(self.author2id),
                                                                    self.author_split_edge['valid']['edge'].size(),
                                                                    dtype=torch.long)
        self.author_split_edge['test']['edge'] = torch.LongTensor(author_edge_test)
        self.author_split_edge['test']['edge_neg'] = torch.randint(0, len(self.author2id),
                                                                   self.author_split_edge['test']['edge'].size(),
                                                                   dtype=torch.long)
        print('train author edge size', self.author_split_edge['train']['edge'].size())
        print('valid author edge size', self.author_split_edge['valid']['edge'].size())
        print('test author edge size', self.author_split_edge['test']['edge'].size())

        # 3.2 build edges for group cooperation
        group_valid_edge, group_test_edge = [], []
        group_valid_neg_edge, group_test_neg_edge = [], []
        valid_group_coo = ddict(list)  # {'group_id':['group_id1','group_id2',...]}
        test_group_coo = ddict(list)  # {'group_id':['group_id1','group_id2',...]}

        for (paper_id, coo_info) in group_co_vt.items():
            coo_group_set = set(coo_info['group_cooperate'])
            if len(coo_group_set) > 1:
                for (i, j) in combinations(coo_group_set, 2):
                    org_i = i.split('-')[-1]
                    org_j = j.split('-')[-1]
                    try:
                        i_id = self.group2id[i]
                        j_id = self.group2id[j]
                        if org_i == org_j:
                            in_coo_num += 1
                        else:
                            out_coo_num += 1
                        if (self.p.only_external_link is False) or (org_i != org_j):
                            if i in group_valid_set:
                                valid_group_coo[i_id].append(j_id)
                            elif i in group_test_set:
                                test_group_coo[i_id].append(j_id)
                            if j in group_valid_set:
                                valid_group_coo[j_id].append(i_id)
                            elif j in group_test_set:
                                test_group_coo[j_id].append(i_id)
                    except:
                        pass
        print('train组织总数', len(group_train_set))
        print('有合作的train组织数', len(train_group_coo))
        print('valid组织总数', len(group_valid_set))
        print('有合作的valid组织数', len(valid_group_coo))
        print('test组织总数', len(group_test_set))
        print('有合作的test组织数', len(test_group_coo))
        print('组织内部合作比例', float(in_coo_num / (in_coo_num + out_coo_num)))
        print('组织外部合作比例', float(out_coo_num / (in_coo_num + out_coo_num)))

        # generate pos/neg edge for validating
        all_group_set = set([i for i in range(self.group_num)])
        for group_id, coo_list in valid_group_coo.items():
            coo_list = set(coo_list)
            sample_neg_list = list(all_group_set - coo_list)
            #neg_ent_list = list(np.random.randint(0, self.group_num, (100)))
            neg_ent_list = np.random.choice(np.array(sample_neg_list, dtype=int), 100, replace=True)
            for coo_group in coo_list:
                group_valid_edge.append([group_id, coo_group])
                group_valid_neg_edge.append([[group_id, neg_ent] for neg_ent in neg_ent_list])

        # generate pos/neg edge for testing
        for group_id, coo_list in test_group_coo.items():
            coo_list = set(coo_list)
            sample_neg_list = list(all_group_set - coo_list)
            #neg_ent_list = list(np.random.randint(0, self.group_num, (100)))
            neg_ent_list = np.random.choice(np.array(sample_neg_list, dtype=int), 100, replace=True)
            for coo_group in coo_list:
                group_test_edge.append([group_id, coo_group])
                group_test_neg_edge.append([[group_id, neg_ent] for neg_ent in neg_ent_list])

        self.group_split_edge['valid']['edge'] = torch.LongTensor(group_valid_edge)
        self.group_split_edge['valid']['neg_edge'] = torch.LongTensor(group_valid_neg_edge)
        self.group_split_edge['test']['edge'] = torch.LongTensor(group_test_edge)
        self.group_split_edge['test']['neg_edge'] = torch.LongTensor(group_test_neg_edge)
        print('valid group edge size', self.group_split_edge['valid']['edge'].size())
        print('test group edge size', self.group_split_edge['test']['edge'].size())
        print('group/group',  2 * (len(group_train_edge) + len(group_valid_edge) + len(group_test_edge)) / self.group_num)
        if self.p.use_g_attribute:
            self.group_attribute = self.load_group_attribute()
        else:
            self.group_attribute = None
        if self.p.use_a_attribute:
            self.author_attribute = self.load_author_attribute()
        else:
            self.author_attribute = None


    def load_group_attribute(self):
        # load org's type\country\city
        self.org_info = {}
        f_org = open(self.p.data_dir + self.p.dataset + '/aminer_affiliations.txt', 'r', encoding='utf-8')
        for line in f_org.readlines():
            dict = json.loads(line)
            self.org_info[dict['id']] = dict
        group_attribute = []  #[{'type':'','country':'','city':''},{},{}...]

        for id, ent in self.id2group.items():
            org_id = self.group_info[ent]["affiliate"]
            try:
                info_dict = self.org_info[org_id]
                del (info_dict['id'])
            except:
                info_dict = {}
            group_attribute.append(info_dict)
        # modify group_attribute to {'type':['company,'university','',...],'country':[]}
        group_attribute_new = {'type': [], 'country': [], 'city': []}
        for each in group_attribute:
            for key_name in group_attribute_new.keys():
                if key_name in each:
                    group_attribute_new[key_name].append(each[key_name])
                else:
                    group_attribute_new[key_name].append('')
        # turn to id
        for key_name, each_list in group_attribute_new.items():
            key_set = set(each_list)
            item2id = {ent: idx for idx, ent in enumerate(key_set)}
            group_attribute_new[key_name] = [item2id[item] for item in each_list]

        return group_attribute_new

    def load_author_attribute(self):
        with open(self.p.data_dir + self.p.dataset + '/author_attr.pkl', 'rb') as f:
            data_df = pickle.load(f)
        author_attr = {'title_emb': [], 'n_citation': [], 'n_pub': []}  # 逐原Id生成

        # title Dimension Reduction
        from sklearn.decomposition import PCA
        other_attr_num = len(author_attr) - 1
        title_dim = self.p.embed_dim - self.p.attr_dim * other_attr_num
        print('title dim',title_dim )
        dim_model = PCA(n_components = title_dim)
        emb_matrix = data_df['title_emb'].to_numpy()
        emb_matrix = np.stack(emb_matrix, axis=0)
        emb_matrix = dim_model.fit_transform(emb_matrix)  # 将X降维(默认二维)后保存到Y中

        data_df['title_emb'] = list(emb_matrix)
        no_att_author = 0
        for id, ent in self.id2author.items():
            if ent in data_df['n_pub']:
                author_attr['title_emb'].append(data_df['title_emb'][ent])
                author_attr['n_citation'].append(data_df['n_citation'][ent])
                author_attr['n_pub'].append(data_df['n_pub'][ent])
            else:
                no_att_author += 1
                author_attr['title_emb'].append(np.zeros([title_dim]))
                author_attr['n_citation'].append(0)
                author_attr['n_pub'].append(0)
        print('no_att_author number',no_att_author)
        # turn to id
        for key_name, each_list in author_attr.items():
            if key_name != 'title_emb':
                key_set = set(each_list)
                item2id = {ent: idx for idx, ent in enumerate(key_set)}
                author_attr[key_name] = [item2id[item] for item in each_list]
            else:
                author_attr[key_name] = [list(item) for item in each_list]

        return author_attr

    def fit(self, run_id):
        self.best_val_value, self.best_val, self.best_epoch = 0., {}, 0
        save_path = os.path.join('./checkpoints', self.p.store_name.replace(':', ''))
        if self.p.restore:  # load
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')


        self.x_group = None
        self.x_author = None
        self.select_dim = None
        with torch.no_grad():
            if self.p.use_g_attribute:
                self.x_group = self.group_attemb_layer(self.group_attribute)
            if self.p.use_a_attribute:
                self.x_author  = self.author_attemb_layer(self.author_attribute)

        if self.p.only_test:
            results = self.evaluate_Recommend()
            for key, result in results.items():
                self.metrics[key].add_result(run_id, result)
        elif self.p.att_visual:
            self.case_study(['group_12905-204983213'])
        else:
            kill_cnt = 0
            for epoch in range(self.p.max_epochs):
                #train
                train_loss = self.run_epoch()
                if self.p.scheduler is not None:
                    if self.p.scheduler == 'plateau':
                        self.scheduler.step(train_loss)
                    else:
                        self.scheduler.step()
                if self.p.use_wandb:
                    wandb.log({f"train/loss-runs": train_loss, f"train/lr": self.optimizer.param_groups[0]["lr"],
                               f"epoch": epoch})
                #eval
                if epoch % self.p.eval_step == 0:
                    results = self.evaluate_Recommend()
                    kill_cnt = 0
                    valid_results, test_results = results
                    self.logger.info(
                        f'Epoch: {epoch:02d}, 'f'Loss: {train_loss:.4f}, 'f'Valid: {valid_results}%, 'f'Test: {test_results}%')
                    for key in valid_results.keys():
                        self.metrics[key].add_result(0, (100 * valid_results[key], 100 * test_results[key]))
                        if self.p.use_wandb:
                            wandb.log({f"valid/{key}": 100 * valid_results[key], f"test/{key}": 100 * test_results[key], "epoch": epoch})
                    if valid_results['mrr'] > self.best_val_value:
                        self.best_val_value = valid_results['mrr']
                        self.best_val = results
                        self.best_epoch = epoch
                        self.save_model(save_path)
                        self.logger.info('Save Results!')
                        kill_cnt = 0

                kill_cnt += 1
                if kill_cnt > 25:
                    self.logger.info("Early Stopping!!")
                    break
                print('---')

        # print the final results of current run
        for key in self.metrics.keys():
            print(key)
            self.metrics[key].print_statistics(metrics=key, run_id=run_id, use_wandb = args.use_wandb)

    # 一个训练epoch
    def run_epoch(self):
        self.model.train()
        self.predictor_group.train()
        if self.p.train_author:
            self.predictor_author.train()

        gpos_train_edge = self.group_split_edge['train']['edge']
        apos_train_edge = self.author_split_edge['train']['edge']
        total_loss = total_examples = 0
        step = 0
        epoch_length = gpos_train_edge.size(0)

        for perm in DataLoader(range(epoch_length), self.p.batch_size, shuffle=True):
            step += 1
            self.optimizer.zero_grad()
            pos_edge = gpos_train_edge[perm].t()
            # just random sample negative instances
            neg_target = torch.randint(0, self.group_num, (pos_edge[0].size(0), 1), dtype=torch.long)

            pos_out, neg_out = self.run_batch(pos_edge, neg_target)
            pos_loss = -torch.log(pos_out + 1e-15).mean()
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
            group_loss = pos_loss + neg_loss

            #author loss
            if self.p.train_author:
                edge = apos_train_edge[perm].t()
                pos_out = self.predictor_author(self.h_author[edge[0]], self.h_author[edge[1]])
                pos_loss = -torch.log(pos_out + 1e-15).mean()
                # Just do some trivial random sampling.
                neg_edge = torch.randint(0, self.author_num, edge.size(), dtype=torch.long)
                neg_out = self.predictor_author(self.h_author[neg_edge[0]], self.h_author[neg_edge[1]])
                neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
                author_loss = pos_loss + neg_loss

            loss = 0.
            if self.p.train_author:
                loss += author_loss
            if self.p.train_group:
                loss += group_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪，大于1时则直接裁
            # torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            self.optimizer.step()

            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples
            del loss, pos_out, neg_out, pos_loss, neg_loss

        return total_loss / total_examples

    def run_batch(self, pos_edge, neg_target):
        #pos edge: [2,batch], neg_target: [batch,1] / [batch，k]
        if len(neg_target.size()) == 1:
            neg_target = neg_target.unsqueeze(1)

        if self.p.graph_based == 'MLP':
            h_author, h_group = self.model(self.x_author, self.x_group, self.a_graph, self.ag_graph, group_num=self.group_num)
        else:
            # generate group batch_set
            old_group_set = set()
            for i in pos_edge[0]:
                old_group_set.add(i.item())
            for i in pos_edge[1]:
                old_group_set.add(i.item())
            for batch_id in range(len(neg_target)):
                for i in neg_target[batch_id]:
                    old_group_set.add(i.item())
            # generate new idx according to the batch_set
            old2new = {ent: idx for idx, ent in enumerate(old_group_set)}
            # substitute old group idx to new_idx in pos_edge, neg_edge
            for index, source_id in enumerate(pos_edge[0]):
                pos_edge[0][index] = old2new[source_id.item()]
            for index, target_id in enumerate(pos_edge[1]):
                pos_edge[1][index] = old2new[target_id.item()]
            for batch_id in range(len(neg_target)):
                for index, neg_target_id in enumerate(neg_target[batch_id]):
                    neg_target[batch_id][index] = old2new[neg_target_id.item()]
            # substitute old group idx to new_idx in ag_graph
            ag_graph = []
            for old_group_id in old_group_set:
                for author_id in self.g2a[old_group_id]:
                    ag_graph.append([author_id, old2new[old_group_id]])
            ag_graph = torch.LongTensor(ag_graph).to(self.device).t()

            if self.p.use_sample:
                # generate author batch_set
                hop_author = {}
                hop_author[0] = set(ag_graph[0])
                for k in range(1, self.p.gcn_layer-1):
                    hop_author[k] = set(self.A[list(hop_author[k-1])].indices).union(set(self.A[:, list(hop_author[k-1])].indices))
                #merge k hop authors in batch
                author_nodes = set()
                for k in range(0, self.p.gcn_layer - 1):
                    author_nodes = author_nodes.union(hop_author[k])
                author_nodes = list(author_nodes)
                aa_graph = self.A[author_nodes, :][:, author_nodes]  #取出author_nodes间涉及的边, author会重新编号
                source_a, target_a, r = ssp.find(aa_graph)
                aa_graph = torch.stack([torch.LongTensor(source_a).to(self.device), torch.LongTensor(target_a).to(self.device)], 0)
                # generate new idx according to the batch_set
                author_nodes_old2new = {ent: idx for idx, ent in enumerate(author_nodes)}
                # substitute old group idx to new_idx in pos_edge, neg_edge
                for index, old_author in enumerate(ag_graph[0]):
                    ag_graph[0][index] = author_nodes_old2new[old_author]
                ag_graph = torch.LongTensor(ag_graph).to(self.device)
                author_nodes = torch.LongTensor(author_nodes).to(self.device)
            else:
                author_nodes = None
                aa_graph = self.a_graph

            if self.p.only_GE:
                h_author, h_group = self.model(self.x_author, self.x_group, aa_graph, ag_graph, group_num = len(old_group_set), author_index = author_nodes)
            else:
                h_author, h_group = self.model(self.x_author, self.x_group, aa_graph, ag_graph, degree=self.author_degree, group_num = len(old_group_set), author_index = author_nodes)

        pos_out = self.predictor_group(h_group[pos_edge[0]], h_group[pos_edge[1]])  # (B,1)
        if neg_target.size(1) == 1: #when training
            neg_out = self.predictor_group(h_group[pos_edge[0]], h_group[neg_target.squeeze(-1)])
        else: #when testing
            neg_out = []
            for i in range(neg_target.size(1)):
                score = self.predictor_group(h_group[pos_edge[0]], h_group[neg_target[:,i].squeeze(-1)]) #[batch]
                neg_out += [score.squeeze().cpu()]
            neg_out = torch.cat(neg_out, dim=0).view(neg_target.size(1), -1).permute(1, 0)  # (batch, self.p.neg_for_test)

        del h_group, ag_graph
        self.h_author = h_author
        return pos_out, neg_out


    @torch.no_grad()
    # 一次验证,只有组织使用recommend推荐
    def evaluate_Recommend(self):
        self.model.eval()
        self.predictor_group.eval()

        split_edge = self.group_split_edge
        pos_valid_edge = split_edge['valid']['edge']
        neg_valid_edge = split_edge['valid']['neg_edge']
        pos_test_edge = split_edge['test']['edge']
        neg_test_edge = split_edge['test']['neg_edge']
        pos_valid_preds = []
        neg_valid_preds = []
        neg_test_preds = []
        pos_test_preds = []

        # 计算valid数据集的分数
        # 每条正边都有100个负例
        for perm in DataLoader(range(pos_valid_edge.size(0)), self.p.batch_size, shuffle=True):
            pos_edge = pos_valid_edge[perm].t()
            # 对于每条正边，保留edge[0]不变，采样100条负例子
            neg_target = neg_valid_edge[perm][:, :, 1]
            pos_valid_out, neg_valid_out = self.run_batch(pos_edge, neg_target)#[B,1] [B,num_neg]
            pos_valid_preds += [pos_valid_out.squeeze().cpu()]
            neg_valid_preds += [neg_valid_out.squeeze().cpu()]  # [step, (batch,100)]
        pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)  # (step*batch,100)

        # 计算test数据集的分数
        for perm in DataLoader(range(pos_test_edge.size(0)), self.p.batch_size):
            pos_edge = pos_test_edge[perm].t()
            # 对于每条正边，保留edge[0]不变，采样100条负例子
            neg_target = neg_test_edge[perm][:, :, 1]
            pos_test_out, neg_test_out = self.run_batch(pos_edge, neg_target)  # [B,1] [B,num_neg]
            pos_test_preds += [pos_test_out.squeeze().cpu()]
            neg_test_preds += [neg_test_out.squeeze().cpu()]  # [step, (batch,100)]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)
        neg_test_pred = torch.cat(neg_test_preds, dim=0)  # (step*batch,100)

        valid_results = eval_pos_neg(pos_valid_pred, neg_valid_pred)
        test_results = eval_pos_neg(pos_test_pred, neg_test_pred)
        return valid_results, test_results

    @torch.no_grad()
    #输入group_id集合，输出该小组每个成员的不同视图的注意力分布，以及每个group对应的推荐TOP10小组
    def case_study(self, group_id_list):
        self.model.eval()
        # 构建ag_graph,graph_id为新group
        ag_graph = []
        #att: (n_edge, set_seeds, heads)  或者 (n_edge,heads)
        for group_id in group_id_list:
            for author_id in self.g2a[self.group2id[group_id]]:
                ag_graph.append([author_id, 0])
            ag_graph = torch.LongTensor(ag_graph).to(self.device).t()
            self.h_author, h_group, att = self.model(self.x_author, self.x_group, self.a_graph, ag_graph, degree=self.author_degree, group_num=1)
            print('group_id', group_id)
            print('member list', self.group_info[group_id]['member_name'])
            #可视化该组的成员及每个成员对应的注意力权重
            #不采样，全量group版本
            '''
            index = torch.eq(self.ag_graph[1],self.group2id[group_id])
            group_att = att[index].squeeze(-1) #[n_group, set_seeds]
            print('member att', group_att)
            author_id_list = self.ag_graph[0][index].tolist()
            author_list = [self.id2author[author_id] for author_id in author_id_list]
            print('member id list', author_list)
            '''
            # 采样版
            group_att = att.squeeze(-1) #[n_member, set_seeed, 1]
            print('member att', group_att)
            author_id_list = ag_graph[0].tolist()
            author_list = [self.id2author[author_id] for author_id in author_id_list]
            print('member id list', author_list)
            degree_dis = self.author_degree[ag_graph[0]]
            print('degree 分布', degree_dis)

            #逐个输出每个人的注意力
            # i 是相对于组织总长度来索引的
            group_info = self.group_info[group_id]
            member_id2name = {group_info['member'][i]:group_info['member_name'][i] for i in range(len(group_info['member']))}
            for i, author_id in enumerate(author_list):
                print('member_id:',author_id, 'degree',degree_dis[i], 'member_name:',member_id2name[author_id] , 'att:', group_att[i])
            '''
            #构造所有的edge对， 共num_group个
            pred_results = []
            for idx in DataLoader(range(self.group_num), self.p.batch_size):
                group_id_batch = torch.LongTensor([self.group2id[group_id] for i in range (self.p.batch_size)])
                pred_results += [self.predictor_group(self.h_group[group_id_batch], self.h_group[idx]).squeeze().cpu()]
            pred_result = torch.cat(pred_results, dim=0) #一维张量，source与group_num个小组的评分
            argsort = torch.argsort(pred_result, dim=0, descending=True)  # (batch*step, 1+n_group), torch.argsort返回的值表示按序排列后，每个位置对应原数组的idx
            top_index = argsort[:10]
            print('推荐的top10:')
            for index in top_index:
                group = self.id2group[index]
                print(group, self.group_info[group]['member_name'],self.org_info[self.group_info[group]['affiliate']])
            '''

    def load_model(self, load_path):
        state = torch.load(load_path)
        state_dict = state['state_dict']
        self.best_val = state['best_val']

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])
        if self.p.scheduler:
            self.scheduler.load_state_dict(state_dict["scheduler"])

    def load_predictor(self, load_path):
        state = torch.load(load_path)
        state_dict = state['state_dict']
        self.predictor_author.load_state_dict(state_dict)

    def save_model(self, save_path):
        #存model
        state = {
            'state_dict': self.model.state_dict(),
            'best_val': self.best_val,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'args': vars(self.p)
        }
        torch.save(state, save_path)
        # 存predictor
        if self.p.train_author:
            state = {'state_dict': self.predictor_author.state_dict()}
            torch.save(state, save_path+'predictor_author')
        elif self.p.train_group:
            state = {'state_dict': self.predictor_group.state_dict()}
            torch.save(state, save_path+'predictor_group')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-args', help="use args", type=bool, default=False) #if true, args priority is higher than yaml
    parser.add_argument('-config_file', help="configuration file Aminer-G/MAG-G/Weeplaces-G.yml", type=str, required=False, default='Aminer-G.yml')

    # basic
    parser.add_argument('-use_wandb', default=True, help='use wandb or not')
    parser.add_argument('-data_dir', default='../data/', help='Log directory')
    parser.add_argument('-log_dir', default='./log/', help='Log directory')
    parser.add_argument('-config_dir', default='./config/', help='Config directory')
    parser.add_argument('-store_name', default='testrun', help='Set run name for saving/restoring models')
    parser.add_argument('-dataset', default='org-community', help='Dataset to use, default: OrgPaper')
    parser.add_argument('-runs', default=1, help='Number of runs')
    parser.add_argument('-use_a_attribute', default=False, help='use author attribute')
    parser.add_argument('-use_g_attribute', default=False, help='use group attribute')
    parser.add_argument('-restore', default=False, help='Restore from the previously saved model')  # 是否载入原模型，默认为False
    parser.add_argument('-use_sample', default = False, help='Use sampling or not') #only Aminer: GAT/HGT, use_sample 时train_author必须为False
    parser.add_argument('-data_size', default=1,type=float, help='Partial datasets')

    # train and test
    parser.add_argument('-train_author', default=True, help='use  author collabration to train')
    parser.add_argument('-train_group', default=True, help='use group collabration to train')
    parser.add_argument('-attr_dim', default=2, help='att attribute')
    parser.add_argument('-batch', dest='batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('-optimizer', dest='optimizer', default='AdamW', type=str, help='Adam/AdamW')
    parser.add_argument('-scheduler', dest='scheduler', default='plateau', type=str, help='cosine/plateau/onecycle/None')
    parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-epoch', dest='max_epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('-eval_step', dest='eval_step', type=int, default=1, help='Number of epochs between two evaluating test')
    parser.add_argument('-l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('-lr', type=float, default=0.001, help='Starting Learning Rate')
    parser.add_argument('-bias', action='store_true', help='Whether to use bias in the model')
    parser.add_argument('-num_workers', type=int, default=0, help='Number of processes to construct batches')
    parser.add_argument('-seed', dest='seed', default=5555, type=int, help='Seed for randomization')
    parser.add_argument('-only_test', default=False, help='only test')
    parser.add_argument('-only_external_link', default=True, help='only test links between orgs')
    parser.add_argument('-att_visual', default=False, help='visualize the attention distribution')

    # gcn设置部分
    parser.add_argument('-only_GE', default=True, help='evaluate on group or author')
    parser.add_argument('-graph_based', default='GCN', help="use MLP and raw features")  # GCN/GraphSage/GAT/HGT/RGCN
    parser.add_argument('-gcn_layer', default=3, type=int, help='Number of GCN Layers')  # KG 隐层
    parser.add_argument('-init_dim', default=128, type=int, help='Initial dimension size for entities and relations')  # kg 初始dim
    parser.add_argument('-gcn_dim', default=128, type=int, help='Number of hidden units in GCN')  # KG 隐层
    parser.add_argument('-embed_dim', default=128, type=int,help='Embedding dimension to give as input to score function')  # output_dim，即KG最后一层（embed_dim）
    parser.add_argument('-dropout', default=0.1, type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop', default=0.1, type=float, help='Dropout after GCN')
    parser.add_argument('-gcn_node_residual', default=False, help='Use kg nodes residual or not')
    parser.add_argument('-gcn_node_concat', default=False, help='Concat kg_nodes or not')
    parser.add_argument('-att_head', default=1, help='att head') #MMAN head 为1， att为 2

    # MMAN
    parser.add_argument('-view_num', default=3, help='Use sampling or not')  # MMAN: set seeds(m)
    parser.add_argument('-i2g_method', default = None, help='how to get group embedding according to the author')  # 设置a到g聚合方式, degree/att/average/set2set/MMAN/DiffPool/GMPool, None表示不从作者聚合信息到组织

    # predict layer
    parser.add_argument('-score_method', default='MLP', help='Use sampling or not')  # MLP/mv_score
    parser.add_argument('-predict_layer', default=3, type=int, help='Number of GCN Layers')

    # load yaml
    args = parser.parse_args()
    if args.args:
        # args priority is higher than yaml
        opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(vars(args))
        args = opt
    else:
        # yaml priority is higher than args
        opt = vars(args)
        args = yaml.load(open(args.config_dir + args.config_file), Loader=yaml.FullLoader)
        opt.update(args)
        args = opt

    args = argparse.Namespace(**args)

    if not args.restore:
        args.store_name = args.store_name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime(
            '%H:%M:%S')  # log文件的命名，在这里进行了更改
    else:
        args.store_name = 'testrun_09_03_2022_104840'  # 自定义的文件名

    if args.use_wandb:
        wandb.login(key='8014511a0dd48cc3ef7fa06c2d54d541b0ad4373')
        wandb.init(project='group-link-'+args.dataset+'(new)')
        wandb.config.update(args)
        if args.only_GE:
            if args.use_a_attribute:
                wandb.run.name = f"{args.graph_based},{args.gcn_layer}layer,X+A,author={args.train_author}"
            else:
                wandb.run.name = f"{args.graph_based},{args.gcn_layer}layer,X,author={args.train_author}"
        else:
            if args.use_a_attribute:
                wandb.run.name = f"{args.i2g_method}({args.view_num}),{args.gcn_layer}layer,X+A,{args.graph_based},{args.score_method},{args.combine_opt},author={args.train_author},head={args.att_head}"
            else:
                wandb.run.name = f"{args.i2g_method}({args.view_num}),{args.gcn_layer}layer,X,{args.graph_based},{args.score_method},{args.combine_opt},author={args.train_author},head={args.att_head}"

    #set
    metrics = {
        'hits@1': Logger(args.runs),
        'hits@3': Logger(args.runs),
        'hits@5': Logger(args.runs),
        'hits@10': Logger(args.runs),
        'hits@20': Logger(args.runs),
        'mrr': Logger(args.runs),
        'ndcg@1': Logger(args.runs),
        'ndcg@3': Logger(args.runs),
        'ndcg@5': Logger(args.runs),
        'ndcg@10': Logger(args.runs),
        'ndcg@20': Logger(args.runs)
    }
    set_gpu(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #######

    model = Runner(args, metrics)
    for run_id in range(args.runs):
        model.fit(run_id)

    for key in metrics.keys():
        print(key)
        metrics[key].print_statistics(metrics=key, use_wandb = args.use_wandb)