import json

import numpy as np
import torch
import yaml
from helper import *
from data_loader import *
from torch_scatter import scatter
from model.models import B2GModel, LinkPredictor
from torch_geometric.data import DataLoader
from logger import Logger
import wandb

def eval_mrr_ndcg(y_pred_pos, y_pred_neg):
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
    hits1_list = (ranking_list <= 1).to(torch.float)
    hits3_list = (ranking_list <= 3).to(torch.float)
    hits5_list = (ranking_list <= 5).to(torch.float)
    hits10_list = (ranking_list <= 10).to(torch.float)
    hits20_list = (ranking_list <= 20).to(torch.float)
    mrr_list = 1. / ranking_list.to(torch.float)
    ndcg1_list= torch.log(torch.tensor(2)) / torch.log(ranking_list[(ranking_list <= 1)] + 1)
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
        self.model = B2GModel(self.p, num_user=self.user_num, num_item=self.item_num, num_group=self.group_num,
                               num_cate=self.cate_num).to(self.device)

        # param to optimize
        param_optimizer = list(self.model.named_parameters())
        if self.p.train_group:
            self.predictor_group = LinkPredictor(self.p.embed_dim, self.p.embed_dim, 1, self.p.predict_layer,
                                                 self.p.dropout, self.p.score_method, self.p.i2g_method,
                                                 self.p.view_num).to(self.device)
            param_optimizer += list(self.predictor_group.named_parameters())

        if self.p.train_user:
            self.predictor_user = LinkPredictor(self.p.embed_dim, self.p.embed_dim, 1, self.p.predict_layer,
                                                self.p.dropout, 'MLP', None, self.p.view_num).to(self.device)
            param_optimizer += list(self.predictor_user.named_parameters())

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
                                                                        patience=10, min_lr=0.000001, verbose=False)
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
        """
        Build graph and generate train\valid\test datasets.
        """

        self.ui_graph = []  # user-item graph, for gcn encoding
        self.ug_graph = []  # user-group graph, for aggregating
        self.ic_graph = []  # item-cate graph, for aggregating
        self.g2u = ddict(list)
        self.c2i = ddict(list)
        self.u2i = ddict(list)
        self.i2u = ddict(list)


        user_set = OrderedSet()
        group_set = OrderedSet()
        item_set = OrderedSet()
        cate_set = OrderedSet()

        # 1. load all entities, generate g2u/c2i/u2i/i2u
        item_cate = json.load(open(self.p.data_dir+self.p.dataset+'/item_cate.json', 'r', encoding='utf-8'))
        for item,cate in item_cate.items():
            item_set.add(item)
            cate_set.add(cate)

        user_item = json.load(open(self.p.data_dir+self.p.dataset+'/user_item.json', 'r', encoding='utf-8'))
        for user,item_list in user_item.items():
            user_set.add(user)

        item_in_group = 0
        group_item = json.load(open(self.p.data_dir+self.p.dataset+'/group_item.json', 'r', encoding='utf-8'))
        for group,item_list in group_item.items():
            group_set.add(group)
            item_in_group += len(set(item_list))

        user_in_group = 0
        user_group = json.load(open(self.p.data_dir+self.p.dataset+'/group_info.json', 'r', encoding='utf-8'))
        for group, user_list in user_group.items():
            group_set.add(group)
            for user in set(user_list):
                user_in_group += 1
                user_set.add(user)

        user2id = {ent: idx for idx, ent in enumerate(user_set)}
        group2id = {ent: idx for idx, ent in enumerate(group_set)}
        item2id = {ent: idx for idx, ent in enumerate(item_set)}
        cate2id = {ent: idx for idx, ent in enumerate(cate_set)}
        self.user_num = len(user_set)
        self.group_num = len(group_set)
        self.item_num = len(item_set)
        self.cate_num = len(cate_set)

        print('user number', self.user_num)
        print('group number', self.group_num)
        print('item number', self.item_num)
        print('cate number', self.cate_num)
        print('Avg. group size', user_in_group/self.group_num)
        print('Avg. group/item', item_in_group/self.group_num)


        # 2. build ui_graph, ic_graph, ug_graph
        ui_edge_all = []
        for user, item_list in user_item.items():
            user_id = user2id[user]
            for item in item_list:
                item_id = item2id[item]
                ui_edge_all.append([user_id, item_id])
                self.ui_graph.append([user_id, item_id])
                self.u2i[user_id].append(item_id)
                self.i2u[item_id].append(user_id)

        for item,cate in item_cate.items():
            self.ic_graph.append([item2id[item], cate2id[cate]])
            self.c2i[cate2id[cate]].append(item2id[item])

        for group, user_list in user_group.items():
            for user in user_list:
                self.ug_graph.append([user2id[user], group2id[group]])
                self.g2u[group2id[group]].append(user2id[user])

        # calculate degrees of users and items
        temp = self.ui_graph.copy()
        for i in user_set:
            try:
                temp.append([self.user2id[i], self.user2id[i]])
            except:
                pass
        temp = torch.LongTensor(temp).t()
        self.user_degree = scatter(torch.ones_like(temp[0]), temp[0], dim=0, reduce='sum').to(self.device)
        self.item_degree = scatter(torch.ones_like(temp[1]), temp[1], dim=0, reduce='sum').to(self.device)

        self.ui_graph = torch.LongTensor(self.ui_graph).to(self.device).t()
        self.ic_graph = torch.LongTensor(self.ic_graph).to(self.device).t()
        self.ug_graph = torch.LongTensor(self.ug_graph).to(self.device).t()


        # 3. build train\valid\test datasets
        # 3.1 user to item edges
        self.ui_split_edge = {'train': {}, 'valid': {}, 'test': {}}
        random.shuffle(ui_edge_all)
        ui_edge_train = ui_edge_all[:int(0.6*len(ui_edge_all))]
        ui_edge_valid = ui_edge_all[int(0.6*len(ui_edge_all)):int(0.8*len(ui_edge_all))]
        ui_edge_test = ui_edge_all[int(0.8 * len(ui_edge_all)):]
        ui_valid_neg = []
        ui_test_neg = []
        for user_id, item_id in ui_edge_valid:
            neg_ent_list = list(np.random.randint(0, self.item_num, (self.p.neg_for_test)))
            ui_valid_neg.append([[user_id, neg_ent] for neg_ent in neg_ent_list])
        for user_id, item_id in ui_edge_test:
            neg_ent_list = list(np.random.randint(0, self.item_num, (self.p.neg_for_test)))
            ui_test_neg.append([[user_id, neg_ent] for neg_ent in neg_ent_list])

        self.ui_split_edge['train']['edge'] = torch.LongTensor(ui_edge_train)
        self.ui_split_edge['valid']['edge'] = torch.LongTensor(ui_edge_valid)
        self.ui_split_edge['test']['edge'] = torch.LongTensor(ui_edge_test)
        self.ui_split_edge['valid']['neg_edge'] = torch.LongTensor(ui_valid_neg)
        self.ui_split_edge['test']['neg_edge'] = torch.LongTensor(ui_test_neg)

        print('train ui edge size', self.ui_split_edge['train']['edge'].size())
        print('valid ui edge size', self.ui_split_edge['valid']['edge'].size())
        print('test ui edge size', self.ui_split_edge['test']['edge'].size())
        print('Avg: user to item', 2*(len(ui_edge_train)+len(ui_edge_valid)+len(ui_edge_test))/self.user_num)

        # 3.2 group to cate edges
        self.gc_split_edge = {'train': {}, 'valid': {}, 'test': {}}
        gc_train_edge, gc_valid_edge, gc_test_edge = [], [], [] #[[source1,target1],[source2,target2]...]
        gc_train_neg_edge, gc_valid_neg_edge, gc_test_neg_edge = [], [], [] #[n_edge,self.p.neg_for_test,2]
        gc_group_valid, gc_group_test = {}, {} #{'group_id':[[pos_ent1, pos_ent2, ...],[neg_ent1, neg_ent2, ...]]}

        train_group_num = int(0.6 * self.group_num)
        valid_group_num = int(0.2 * self.group_num)
        count = 0

        for group, item_list in group_item.items():
            group_id = group2id[group]
            cate_list = [cate2id[item_cate[item]] for item in item_list]
            sample_neg_list = [cate for cate in range(self.cate_num) if cate not in cate_list] #neg target
            # sample one neg example for each pos example when training
            if count < train_group_num:
                count += 1
                for cate in cate_list:
                    gc_train_edge.append([group_id, cate])
                    neg_ent = np.random.choice(np.array(sample_neg_list, dtype=int), 1, replace=False)[0]
                    gc_train_neg_edge.append([group_id,neg_ent])
            # sample #neg_for_test examples for each pos example when testing
            elif count < train_group_num + valid_group_num:
                count += 1
                neg_ent_list = list(np.random.choice(np.array(sample_neg_list, dtype=int), self.p.neg_for_test, replace=True))
                gc_group_valid[group_id] = [cate_list, neg_ent_list]
                for cate in cate_list:
                    gc_valid_edge.append([group_id,cate])
                    gc_valid_neg_edge.append([[group_id, neg_ent] for neg_ent in neg_ent_list])
            else:
                count += 1
                neg_ent_list = list(np.random.choice(np.array(sample_neg_list, dtype=int), self.p.neg_for_test, replace=True))
                gc_group_test[group2id[group]] = [cate_list, neg_ent_list]
                for cate in cate_list:
                    gc_test_edge.append([group_id,cate])
                    gc_test_neg_edge.append([[group_id, neg_ent] for neg_ent in neg_ent_list])

        print('Avg. group/group', 2*(len(gc_train_edge)+len(gc_valid_edge)+len(gc_test_edge))/self.group_num)
        self.gc_split_edge['train']['edge'] = torch.LongTensor(gc_train_edge)
        self.gc_split_edge['train']['neg_edge'] = torch.LongTensor(gc_train_neg_edge)
        self.gc_split_edge['valid']['edge'] = torch.LongTensor(gc_valid_edge)
        self.gc_split_edge['valid']['neg_edge'] = torch.LongTensor(gc_valid_neg_edge)
        self.gc_split_edge['test']['edge'] = torch.LongTensor(gc_test_edge)
        self.gc_split_edge['test']['neg_edge'] = torch.LongTensor(gc_test_neg_edge)


    def fit(self, run_id):
        self.best_val_value, self.best_val, self.best_epoch = 0., {}, 0
        save_path = os.path.join('./checkpoints', self.p.store_name.replace(':', ''))
        self.x_user = None
        self.x_item = None

        if self.p.restore: # load
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        if self.p.only_test: # test
            valid_results, test_results = self.evaluate_sample()
            for key in valid_results.keys():
                self.metrics[key].add_result(0, (100 * valid_results[key], 100 * test_results[key]))

        else:
            kill_cnt = 0
            for epoch in range(self.p.max_epochs):
                #train
                train_loss = self.train_epoch()
                if self.p.scheduler is not None:
                    if self.p.scheduler == 'plateau':
                        self.scheduler.step(train_loss)
                    else:
                        self.scheduler.step()
                if self.p.use_wandb:
                    wandb.log({f"train/loss-runs": train_loss,
                               f"train/lr": self.optimizer.param_groups[0]["lr"], f"epoch": epoch})

                #eval
                if epoch % self.p.eval_step == 0:
                    results = self.evaluate_sample()
                    valid_results, test_results = results
                    self.logger.info(
                        f'Epoch: {epoch:02d}, 'f'Loss: {train_loss:.4f}, 'f'Valid: {valid_results}%, 'f'Test: {test_results}%')
                    for key in valid_results.keys():
                        self.metrics[key].add_result(run_id, (100 * valid_results[key], 100 * test_results[key]))
                        if self.p.use_wandb:
                            wandb.log({f"valid/{key}": 100 * valid_results[key], f"test/{key}": 100 * test_results[key],
                                       "epoch": epoch})
                    if valid_results['mrr'] > self.best_val_value:
                        self.best_val_value = valid_results['mrr']
                        self.best_val = results
                        self.best_epoch = epoch
                        self.save_model(save_path)
                        self.logger.info('Save Results!')
                        kill_cnt = 0

                kill_cnt += 1
                if kill_cnt > 20:
                    self.logger.info("Early Stopping!!")
                    break
                print('---------------------------------------------------')

        #print the final results of current run
        for key in self.metrics.keys():
            print(key)
            self.metrics[key].print_statistics(metrics=key, run_id=run_id, use_wandb = self.p.use_wandb)


    def train_epoch(self):
        self.model.train()
        if self.p.train_user:
            self.predictor_user.train()
        if self.p.train_group:
            self.predictor_group.train()

        upos_train_edge = self.ui_split_edge['train']['edge']
        gpos_train_edge = self.gc_split_edge['train']['edge']
        gneg_train_edge = self.gc_split_edge['train']['neg_edge']

        total_loss = total_examples = 0
        step = 0
        epoch_length = min(gpos_train_edge.size(0), upos_train_edge.size(0))

        print('total train step', epoch_length/self.p.batch_size)
        for perm in DataLoader(range(epoch_length), self.p.batch_size, shuffle=True):
            step += 1
            self.optimizer.zero_grad()
            # wandb.watch(self.model)

            # group loss
            if self.p.train_group:
                edge = gpos_train_edge[perm].t()
                neg_edge = gneg_train_edge[perm].t()
                pos_out, neg_out = self.run_batch(edge, neg_edge[1])
                pos_loss = -torch.log(pos_out + 1e-15).mean()
                neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
                group_loss = pos_loss + neg_loss

            # ui loss
            if self.p.train_user:
                edge = upos_train_edge[perm].t()
                pos_out = self.predictor_user(self.h_user[edge[0]], self.h_item[edge[1]])
                pos_loss = -torch.log(pos_out + 1e-15).mean()
                neg_edge =  torch.randint(0, self.user_num, edge.size(), dtype=torch.long, device= self.device)
                neg_out = self.predictor_user(self.h_user[neg_edge[0]], self.h_item[neg_edge[1]])
                neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
                author_loss = pos_loss + neg_loss

            loss = 0.
            if self.p.train_user:
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

            if step % 10 == 0:
                print(step, total_loss / total_examples)

        return total_loss / total_examples

    def run_batch(self, pos_edge, neg_target):
        #when trainning： pos edge[2,batch], neg_target[batch，1]
        #when testing： pos edge[2,batch], neg_target[batch，k]
        ic_graph, ug_graph = [],[]
        if len(neg_target.size()) == 1:
            neg_target = neg_target.unsqueeze(1)

        #generate group/cate batch_set
        old_group_set = set()
        old_cate_set = set()
        old_user_set = set()
        old_item_set = set()
        for i in pos_edge[0]:
            old_group_set.add(i.item())
        for i in pos_edge[1]:
            old_cate_set.add(i.item())
        for batch_id in range(len(neg_target)):
            for i in neg_target[batch_id]:
                old_cate_set.add(i.item())

        # generate new idx according to the batch_set
        old2new_group = {ent: idx for idx, ent in enumerate(old_group_set)}
        old2new_cate = {ent: idx for idx, ent in enumerate(old_cate_set)}
        node_num = {'group':len(old_group_set), 'cate':len(old_cate_set), 'user':len(old_user_set), 'item':len(old_item_set)}

        # substitute old group/cate idx to new_idx in pos_edge, neg_edge
        for index, source_id in enumerate(pos_edge[0]):
            pos_edge[0][index] = old2new_group[source_id.item()]
        for index, target_id in enumerate(pos_edge[1]):
            pos_edge[1][index] = old2new_cate[target_id.item()]
        for batch_id in range(len(neg_target)):
            for index, neg_target_id in enumerate(neg_target[batch_id]):
                neg_target[batch_id][index] = old2new_cate[neg_target_id.item()]

        # substitute old group/cate idx to new_idx in ic_graph, ug_graph
        for group_id in old_group_set:
            for user_id in self.g2u[group_id]:
                ug_graph.append([user_id, old2new_group[group_id]])
        for cate_id in old_cate_set:
            for item_id in self.c2i[cate_id]:
                ic_graph.append([item_id, old2new_cate[cate_id]])

        ic_graph = torch.LongTensor(ic_graph).to(self.device).t()
        ug_graph = torch.LongTensor(ug_graph).to(self.device).t()

        if self.p.only_GE:
            h_group, h_cate, self.h_user, self.h_item = self.model(self.x_user, self.x_item, self.ui_graph, ic_graph, ug_graph, node_num = node_num)
        else:
            h_group, h_cate, self.h_user, self.h_item = self.model(self.x_user, self.x_item, self.ui_graph, ic_graph, ug_graph, node_num = node_num, u_degree = self.user_degree, i_degree = self.item_degree)
        # wandb.watch(self.model)

        pos_out = self.predictor_group(h_group[pos_edge[0]], h_cate[pos_edge[1]])  # (B,1)
        if neg_target.size(1) == 1:
            neg_out = self.predictor_group(h_group[pos_edge[0]], h_cate[neg_target.squeeze(-1)])  # (B,1)
        else:
            neg_out = []
            for i in range(neg_target.size(1)):
                score = self.predictor_group(h_group[pos_edge[0]], h_cate[neg_target[:,i].squeeze(-1)]) #[batch]
                neg_out += [score.squeeze().cpu()]  # 每次得到一个batch维的张量，一共得到self.p.neg_for_test个batch维的张量
            neg_out = torch.cat(neg_out, dim=0).view(neg_target.size(1), -1).permute(1, 0)  # (batch,self.p.neg_for_test)

        del pos_edge,neg_target, h_group, h_cate
        return pos_out, neg_out


    @torch.no_grad()
    def evaluate_sample(self):
        self.model.eval()
        if self.p.train_user:
            self.predictor_user.eval()
        if self.p.train_group:
            self.predictor_group.eval()

        pos_valid_preds = []
        neg_valid_preds = []
        neg_test_preds = []
        pos_test_preds = []

        # calculate group-cate scores
        if self.p.train_group:
            split_edge = self.gc_split_edge
            pos_valid_edge = split_edge['valid']['edge']
            neg_valid_edge = split_edge['valid']['neg_edge']
            pos_test_edge = split_edge['test']['edge']
            neg_test_edge = split_edge['test']['neg_edge']

            #valid group-cate scores
            print('valid_edge',pos_valid_edge.size(0))
            for perm in DataLoader(range(pos_valid_edge.size(0)), self.p.batch_size, shuffle=True):
                edge = pos_valid_edge[perm].t()
                neg_target = neg_valid_edge[perm][:,:,1] #[batch,100,2] → [batch,100]
                pos_out, neg_out = self.run_batch(edge, neg_target)
                pos_valid_preds += [pos_out.squeeze().cpu()]
                neg_valid_preds += [neg_out.squeeze().cpu()]

            #test group-cate scores
            print('test_edge', pos_test_edge.size(0))
            for perm in DataLoader(range(pos_test_edge.size(0)), self.p.batch_size):
                edge = pos_test_edge[perm].t()
                neg_target = neg_test_edge[perm][:,:,1]
                pos_out, neg_out = self.run_batch(edge, neg_target)
                pos_test_preds += [pos_out.squeeze().cpu()]
                neg_test_preds += [neg_out.squeeze().cpu()]

        pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)  # (step*batch,100)
        pos_test_pred = torch.cat(pos_test_preds, dim=0)
        neg_test_pred = torch.cat(neg_test_preds, dim=0)  # (step*batch,100)
        valid_results = eval_mrr_ndcg(pos_valid_pred, neg_valid_pred)
        test_results = eval_mrr_ndcg(pos_test_pred, neg_test_pred)

        return valid_results, test_results

    @torch.no_grad()
    def evaluate_all(self):
        self.model.eval()


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
        self.predictor_user.load_state_dict(state_dict)

    #只存了model而没有存predictor
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
        if self.p.train_user:
            state = {'state_dict': self.predictor_user.state_dict()}
            torch.save(state, save_path+'predictor_user')
        elif self.p.train_group:
            state = {'state_dict': self.predictor_group.state_dict()}
            torch.save(state, save_path+'predictor_group')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-args', help="use args", type=bool, default=True) #if true, args priority is higher than yaml
    parser.add_argument('-config_file', help="configuration file Aminer-G/MAG-G/Weeplaces-G.yml", type=str, default='Weeplaces-G.yml')

    # basic
    parser.add_argument('-use_wandb', default=True, help='use wandb or not')
    parser.add_argument('-data_dir', default='../data/', help='Log directory')
    parser.add_argument('-log_dir', default='./log/', help='Log directory')
    parser.add_argument('-config_dir', default='./config/', help='Config directory')
    parser.add_argument('-store_name', default='testrun', help='Set run name for saving/restoring models')
    parser.add_argument('-dataset', default='weeplaces', help='Dataset to use, default: OrgPaper')
    parser.add_argument('-data_size', default=1, help='Use sampling or not')
    parser.add_argument('-runs', default=5, help='Number of runs')
    parser.add_argument('-restore', default = False, help='Restore from the previously saved model') # agg_user_result 时需要设置为True

    # train and test
    parser.add_argument('-train_user', default=True, help='use user item to train')
    parser.add_argument('-train_group', default=True, help='use group collabration to train')
    parser.add_argument('-use_attribute', default=False, help='use_attribute')
    parser.add_argument('-neg_for_test', default=100, help='sample neg entities for test')
    parser.add_argument('-batch', dest='batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('-optimizer', dest='optimizer', default='AdamW', type=str, help='Adam/AdamW')
    parser.add_argument('-scheduler', dest='scheduler', default='plateau', type=str, help='cosine/plateau/onecycle/None')
    parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-epoch', dest='max_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('-eval_step', dest='eval_step', type=int, default=1, help='Number of epochs between two evaluating test')
    parser.add_argument('-l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('-lr', type=float, default=0.001, help='Starting Learning Rate')
    parser.add_argument('-bias', action='store_true', help='Whether to use bias in the model')
    parser.add_argument('-num_workers', type=int, default=0, help='Number of processes to construct batches')
    parser.add_argument('-seed', dest='seed', default=2345, type=int, help='Seed for randomization')
    parser.add_argument('-only_test', default=False, help='only test')

    # gcn
    parser.add_argument('-only_GE', default=False, help='evaluate on group or author') #only_GE
    parser.add_argument('-graph_based', default='GCN', help='Only use graph when encode author and group')  # /GCN/GAT/GraphSage/RGCN/RGAT/HGT
    parser.add_argument('-init_dim', default=128, type=int, help='Initial dimension size for entities and relations')  # kg 初始dim
    parser.add_argument('-gcn_dim', default=128, type=int, help='Number of hidden units in GCN')  # KG 隐层
    parser.add_argument('-gcn_layer', default=4, type=int, help='Number of GCN Layers')  # KG 隐层
    parser.add_argument('-embed_dim', default=128, type=int,
                        help='Embedding dimension to give as input to score function')  # output_dim，即KG最后一层（embed_dim）
    parser.add_argument('-dropout', default=0.1, type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop', default=0.1, type=float, help='Dropout after GCN')
    parser.add_argument('-gcn_node_residual', default=False, help='Use kg nodes residual or not')
    parser.add_argument('-gcn_node_concat', default=False, help='Concat kg_nodes or not')
    parser.add_argument('-att_head', default=1, help='att head')

    #MMAN
    parser.add_argument('-view_num', default=3, help='Use sampling or not')  # MMAN: set seeds(m)
    parser.add_argument('-i2g_method', default ='GMPool',help='how to get  group embedding according to the author')  # 设置a到g聚合方式, degree/att/average/set2set/MMAN, None表示不从作者聚合信息到组织

    # predict layer
    parser.add_argument('-score_method', default='mv_score', help='Use sampling or not')  # MLP/mv_score
    parser.add_argument('-predict_layer', default=3, type=int, help='Number of GCN Layers')

    # load yaml
    args = parser.parse_args()
    if args.args:  # args priority is higher than yaml
        opt = yaml.load(open(args.config_dir + args.config_file), Loader=yaml.FullLoader)
        opt.update(vars(args))
        args = opt
    else:  # yaml priority is higher than args
        opt = vars(args)
        args = yaml.load(open(args.config_dir + args.config_file), Loader=yaml.FullLoader)
        opt.update(args)
        args = opt
    args = argparse.Namespace(**args)

    if not args.restore:
        args.store_name = args.store_name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')  # log文件的命名，在这里进行了更改
    else:
        args.store_name = 'testrun_16_07_2022_161501'  # 自定义的文件名

    if args.use_wandb:
        wandb.login(key='8014511a0dd48cc3ef7fa06c2d54d541b0ad4373')
        wandb.init(project='group-link-weeplaces(new)')
        wandb.config.update(args)

        if args.only_GE:
            wandb.run.name = f"{args.graph_based},{args.gcn_layer}layer,user={args.train_user}"
        else:
            wandb.run.name = f"{args.i2g_method}({args.view_num}),{args.gcn_layer}layer,{args.graph_based},user={args.train_user},{args.score_method},head={args.att_head}"

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

    print('----------------------------------------------')
    for key in metrics.keys():
        print(key)
        metrics[key].print_statistics(metrics=key, use_wandb = args.use_wandb)