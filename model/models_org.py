from helper import *
from torch_geometric.nn import Set2Set
from model.graph_layer import *


class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p		= params
		self.act	= torch.tanh
		self.bceloss	= torch.nn.BCELoss()

	def cal_bceloss(self, pred, true_label):
		return self.bceloss(pred, true_label)

	def cal_BPRloss(self, pos_score, neg_score):
		#pos score:[batch]
		#neg score:[batch,K]
		batch = neg_score.size()[0]
		K = neg_score.size()[1]
		pos_score = pos_score.unsqueeze(-1).expand([batch,K])#[batch,k]
		result = torch.sum(-torch.log(torch.sigmoid(pos_score - neg_score)), dim = -1)#[batch]
		return torch.mean(result, dim = -1)

	def cal_triplet_Loss(self, pos_score, neg_score):
		#max(0,(pos_score-neg_score-1))
		batch = neg_score.size()[0]
		K = neg_score.size()[1]
		pos_score = pos_score.unsqueeze(-1).expand([batch,K])#[batch,k]
		total_score = pos_score - neg_score - 1
		result,_ = torch.max(torch.cat([total_score,torch.zeros([batch,1])],dim = -1), dim = 1)
		return torch.mean(result, dim = -1)


# 与最普通链接预测处理方式一致
# 输出author_emb, h_emb
class MainGraph(BaseModel):
    def __init__(self, params, num_author=0, num_group=0, node_type_dict=None, edge_type_dict=None):
        self.p = params
        super(MainGraph, self).__init__(params)

        self.num_author = num_author
        self.num_group = num_group

        self.drop = torch.nn.Dropout(self.p.hid_drop)

        # base model
        if self.p.graph_based == 'GCN':
            self.GraphBase = GCN(self.p.gcn_layer, self.p.init_dim, self.p.gcn_dim, self.p.embed_dim, params=self.p)
        elif self.p.graph_based == 'GAT':
            self.GraphBase = GAT(self.p.gcn_layer, self.p.init_dim, self.p.gcn_dim, self.p.embed_dim, params=self.p)
        elif self.p.graph_based == 'GraphSage':
            self.GraphBase = GraphSAGE(self.p.gcn_layer, self.p.init_dim, self.p.gcn_dim, self.p.embed_dim, params=self.p)
        elif self.p.graph_based == 'HGT':
            node_type_dict = {'author':[], 'group':[]}
            edge_type_dict = {('author', 'to', 'author'):[],('author', 'to', 'group'):[],('group', 'to', 'author'):[]}
            self.GraphBase = HGT(node_type_dict, edge_type_dict, params=self.p)
        elif self.p.graph_based == 'RGCN':
            self.GraphBase = RGCN(self.p.gcn_layer, self.p.init_dim, self.p.gcn_dim, self.p.embed_dim, params=self.p)

        # group model
        if self.p.i2g_method == 'MMAN':
            self.GraphGroup = GCN(3, self.p.view_num * self.p.gcn_dim, self.p.view_num * self.p.gcn_dim,
                                  self.p.view_num * self.p.embed_dim, params=self.p)
        else:
            self.GraphGroup = GCN(3, self.p.gcn_dim, self.p.gcn_dim, self.p.embed_dim, params=self.p)

        # init embedding
        if self.p.use_a_attribute is False:
            self.author_embedding_layer = nn.Embedding(self.num_author, self.p.init_dim)
            self.author_embedding_layer.reset_parameters()

        self.group_embedding_layer = nn.Embedding(self.num_group, self.p.init_dim)
        self.group_embedding_layer.reset_parameters()


        # For:　concat group
        self.group_out_lin = Linear(2 * self.p.embed_dim, self.p.embed_dim)

        # For:　author and item liner layers
        self.individual_lin = Linear(self.p.embed_dim, self.p.embed_dim)
        self.group_lin = Linear(self.p.embed_dim, self.p.embed_dim)

        # i2g model
        if self.p.i2g_method is not None:
            if self.p.i2g_method == 'att':
                self.heads = self.p.att_head
                self.att_src = Parameter(torch.Tensor(1, self.heads, self.p.gcn_dim))
                self.att_dst = Parameter(torch.Tensor(1, self.heads, self.p.gcn_dim))
                self.lin_src = Linear(self.p.embed_dim, self.heads * self.p.gcn_dim, False, weight_initializer='glorot')
                self.lin_dst = Linear(self.p.embed_dim, self.heads * self.p.gcn_dim, False, weight_initializer='glorot')
                self.lin_src.reset_parameters()
                self.lin_dst.reset_parameters()
                glorot(self.att_src)
                glorot(self.att_dst)

            elif self.p.i2g_method == 'MMAN':
                self.heads = 1
                self.lin_src_K = Linear(self.p.embed_dim, self.heads * self.p.gcn_dim, False,
                                        weight_initializer='glorot')
                self.lin_src_V = Linear(self.p.embed_dim, self.heads * self.p.gcn_dim, False,
                                        weight_initializer='glorot')
                self.Q = Parameter(torch.Tensor(1, self.p.view_num, self.heads, self.p.gcn_dim))
                self.W_r = Parameter(torch.Tensor(self.p.view_num, self.heads, self.p.gcn_dim, self.p.gcn_dim))
                self.ln = nn.LayerNorm(self.p.gcn_dim)
                self.ln1 = nn.LayerNorm(self.p.gcn_dim)
                self.lin_src_K.reset_parameters()
                self.lin_src_V.reset_parameters()
                self.ln.reset_parameters()
                self.ln1.reset_parameters()
                glorot(self.Q)
                glorot(self.W_r)

            elif self.p.i2g_method == 'set2set':
                self.pool = Set2Set(self.p.embed_dim,
                                    processing_steps=1)  # global_add_pool/global_mean_pool/global_max_pool/GlobalAttention
                self.group_lin = Linear(2 * self.p.embed_dim, self.p.embed_dim)


        self.group_out_lin.reset_parameters()
        self.individual_lin.reset_parameters()
        self.group_lin.reset_parameters()
        self.h_author = None

    def i2g_degree(self, each_i, agg_graph, a_degree):
        each_i = self.individual_lin(each_i)  # 线性变换
        # 针对每条ag边，计算对应的alpha
        alpha = torch.index_select(a_degree, 0, agg_graph[0]).float()  # 0是作者，1是组织
        # 计算softmax 后的注意力向量
        alpha = softmax(alpha, agg_graph[1], num_nodes=int(agg_graph[0].size(0)))
        x_i = torch.mul(alpha.unsqueeze(1), each_i)
        h_group = scatter(x_i, agg_graph[1], dim=0, reduce='sum')
        h_group = self.group_lin(h_group)
        return h_group

    def i2g_att(self, each_i, agg_graph, h_group=None):
        if h_group is None:
            h_group = scatter(each_i, agg_graph[1], dim=0,
                              reduce='mean')  # average individual emb to get initial h_group
        x_j = torch.index_select(h_group, 0, agg_graph[1])
        x_i = self.lin_src(each_i).view(-1, self.heads, self.p.gcn_dim)
        x_j = self.lin_dst(x_j).view(-1, self.heads, self.p.gcn_dim)
        alpha_src = (x_j * self.att_src).sum(-1)  # (n_edge,self.heads)
        alpha_dst = (x_i * self.att_dst).sum(-1)
        alpha = alpha_src + alpha_dst
        alpha = F.leaky_relu(alpha, 0.2)  # relu(W*source +W*target): (n_edge,self.heads)
        alpha = softmax(alpha, agg_graph[1], num_nodes=int(agg_graph[0].size(0)))
        x_j = x_j * alpha.unsqueeze(-1)  # (n_edge, self.heads, dim) * (n_edge, self.heads) : (n_edge, self.heads, dim)
        x_j = x_j.mean(dim=1)  # (n_edge, dim)
        h_group = scatter(x_j, agg_graph[1], dim=0, reduce='sum')
        h_group = self.group_lin(h_group)
        return h_group

    def i2g_MMAN(self, each_i, agg_graph, num_group):
        x_i = self.individual_lin(each_i)  # 线性变换
        x_i_K = self.lin_src_K(x_i).view(-1, self.heads, self.p.gcn_dim).unsqueeze(1).expand(-1, self.p.view_num, -1,
                                                                                             -1)  # (n_edge, view_num, heads, dim)
        x_i_V = self.lin_src_V(x_i).view(-1, self.heads, self.p.gcn_dim).unsqueeze(1).expand(-1, self.p.view_num, -1,
                                                                                             -1)  # (n_edge, view_num, heads, dim)
        alpha = (self.Q * x_i_K).sum(dim=-1)  # (n_edge, view_num, heads)
        alpha = alpha / math.sqrt(x_i_K.size(-1))
        alpha = softmax(alpha, agg_graph[1], num_nodes=int(agg_graph[0].size(0)),
                        dim=0)  # softmax for source nodes to the same target, (n_edge, view_num, heads)
        out = (x_i_V * (alpha.unsqueeze(-1))).mean(
            dim=2)  # (n_edge, view_num, heads, dim)  * (n_edge, view_num, heads,1 ) → (n_edge, view_num, heads, dim) → (n_edge, view_num, dim)
        h_group = scatter(out, agg_graph[1], dim=0, reduce='sum', dim_size=num_group)  # (n_group, view_num, dim)
        h_group = self.ln(h_group)
        return h_group

    def forward(self, x_author, x_group, aa_graph, ag_graph, degree, group_num = None, author_index = None, att_author = None):
        '''
        if x_author is None:
            x_author = self.author_embedding_layer(torch.LongTensor([idx for idx in range(self.num_author)]).cuda())
            self.h_author = self.GraphBase.forward(aa_graph, x_author)
        else:
            self.h_author = x_author
        '''
        agg_graph = ag_graph  # 聚合用

        if group_num is None:
            group_num =self.num_group

        if x_author is None:
            x_author = self.author_embedding_layer(torch.LongTensor([idx for idx in range(self.num_author)]).to(ag_graph.device))
        if x_group is None:
            x_group = torch.zeros(group_num, self.p.init_dim).to(ag_graph.device)  # 初始化为全0张量

        if self.p.att_top_combine == 'init_concat':
            x_author = torch.cat((x_author, att_author), dim=-1)

        if author_index is not None:
            x_author = torch.index_select(x_author, 0, author_index)


        if self.p.graph_based == 'HGT':
            edge_type_dict = {}
            node_type_dict = {}
            # 构造edge_type_dict
            '''
            aa_graph = SparseTensor(
                row=aa_graph[1], col=aa_graph[0],
                value=None, sparse_sizes=(self.num_author, self.num_author),
                is_sorted=False)
            ag_graph = SparseTensor(
                row=ag_graph[1], col=ag_graph[0],
                value=None, sparse_sizes=(self.num_author+group_num, self.num_author+group_num),
                is_sorted=False)
            '''
            edge_type_dict[('author', 'to', 'author')] = aa_graph
            edge_type_dict[('author', 'to', 'group')] = ag_graph
            edge_type_dict[('group', 'to', 'author')] = torch.stack((ag_graph[1], ag_graph[0]),0)
            node_type_dict['author'] = x_author
            node_type_dict['group'] = x_group
            node_type_dict = self.GraphBase.forward(node_type_dict, edge_type_dict)
            h_author = node_type_dict['author']
            h_group_gcf = node_type_dict['group']
        else:
            # 合并成同一个图，使用graph模型传递
            # 顺序为: group\cate\user\item
            x_union = torch.cat([x_author, x_group], dim=0)
            author_num = x_author.size(0)
            ag_graph = torch.stack((ag_graph[0], ag_graph[1] + author_num))
            ga_graph = torch.stack((ag_graph[1], ag_graph[0]))
            edge_uninon = torch.cat([aa_graph, ag_graph, ga_graph], dim=-1)
            # edge_uninon = torch.cat([a_graph, ag_graph], dim=-1)

            edge_type = None
            if self.p.graph_based == 'RGCN' or self.p.graph_based == 'RGAT':
                edge_type = [0 for i in range(aa_graph.size(-1))]
                edge_type.extend([1 for i in range(ag_graph.size(-1))])
                edge_type.extend([2 for i in range(ga_graph.size(-1))])
                edge_type = torch.LongTensor(edge_type).to(ag_graph.device)
                graph_union = edge_uninon
            else:
                graph_union = SparseTensor(
                    row=edge_uninon[1], col=edge_uninon[0],
                    value=None,
                    is_sorted=False)
            del edge_uninon
            x_union = self.GraphBase.forward(graph_union, x_union, edge_type)
            h_author = x_union[:author_num]
            h_group_gcf = x_union[author_num:]
            #self.h_group = x_union[author_num:]


        # author2group聚合
        each_i = torch.index_select(h_author, 0, agg_graph[0])  # 选出source node 对应的表征
        # 依据节点度设置节点权重
        if self.p.i2g_method is not None:
            if self.p.i2g_method == 'degree':
                h_group = self.a2g_degree(each_i, agg_graph, degree, group_num)
            elif self.p.i2g_method == 'att':
                if self.p.att_visual:
                    h_group, att = self.a2g_att(each_i, agg_graph, group_num,h_group_gcf)
                else:
                    h_group = self.a2g_att(each_i, agg_graph, group_num, h_group_gcf)
            elif self.p.i2g_method == 'att_set':  # transset中使用的att
                if self.p.att_visual:
                    h_group, att = self.a2g_att_set(each_i, agg_graph, group_num)
                else:
                    h_group = self.a2g_att_set(each_i, agg_graph, group_num)
            elif self.p.i2g_method == 'transformer_cls':  # transformer中使用的att,类似取CLS表征
                h_group = self.a2g_transformer_cls(each_i, agg_graph, group_num)
            elif self.p.i2g_method == 'MMAN':  # MMAN
                if self.p.att_visual:
                    h_group, att = self.a2g_MMAN(each_i, agg_graph, group_num)
                else:
                    h_group = self.a2g_MMAN(each_i, agg_graph, group_num)
            elif self.p.i2g_method == 'set2set':  # set2set
                h_group = self.pool(each_i, agg_graph[1])  # 前一个是emb,后面是该emb对应的组序号,取出序号相同的节点embedding运用pool函数聚合到一起，最后返回[n_index,embedding]
                h_group = self.group_lin(h_group)
            else:  # average(deepset)
                #each_i = self.individual_lin(each_i)  # 线性变换
                h_group = scatter(each_i, agg_graph[1], dim=0, reduce='mean', dim_size = group_num)
                #h_group = self.group_lin(h_group)
        else:
            h_group = None

        if self.p.att_visual:
            return h_author, h_group, h_group_gcf, att
        else:
            return h_author, h_group, h_group_gcf

    def HIN_foward(self, x_author, x_group, node_type_dict, edge_type_dict):
        if x_author is None:
            x_author = self.author_embedding_layer(torch.LongTensor([idx for idx in range(self.num_author)]).cuda())
        if x_group is None:
            x_group = self.group_embedding_layer(torch.LongTensor([idx for idx in range(self.num_group)]).cuda())
        node_type_dict['author'] = x_author
        node_type_dict['group'] = x_group
        node_type_dict = self.GraphBase.forward(node_type_dict, edge_type_dict)
        return node_type_dict['author'], node_type_dict['group']

    def ag_joint_foward(self, x_author, x_group, ag_joint_graph, edge_type = None):
        if x_author is None:
            x_author = self.author_embedding_layer(torch.LongTensor([idx for idx in range(self.num_author)]).cuda())
            x_author = self.individual_lin(x_author)
        if x_group is None:
            x_group = self.group_embedding_layer(torch.LongTensor([idx for idx in range(self.num_group)]).cuda())
            #x_group = self.group_lin(x_group)
        x_all = torch.cat((x_group, x_author), 0)
        h = self.GraphBase.forward(ag_joint_graph, x_all, edge_type)
        return h[self.num_group:], h[:self.num_group]

    #for inductive training
    def graph_forward(self, x_author, x_group, a_graph, ag_graph, group_num, author_index = None, att_author = None):
        if x_author is None:
            x_author = self.author_embedding_layer(torch.LongTensor([idx for idx in range(self.num_author)]).to(ag_graph.device))

        if x_group is None:
            x_group = torch.zeros(group_num, self.p.init_dim).to(ag_graph.device)  # 初始化为全0张量

        if author_index is not None:
            x_author = torch.index_select(x_author, 0, author_index)

        if self.p.att_top_combine == 'init_concat':
            x_author = torch.cat((x_author, att_author), dim=-1)

        if self.p.graph_based == 'HGT':
            edge_type_dict = {}
            node_type_dict = {}
            # 构造edge_type_dict
            '''
            a_graph = SparseTensor(
                row=a_graph[1], col=a_graph[0],
                value=None, sparse_sizes=(self.num_author, self.num_author),
                is_sorted=False)
            ag_graph = SparseTensor(
                row=ag_graph[1], col=ag_graph[0],
                value=None, sparse_sizes=(self.num_author+group_num, self.num_author+group_num),
                is_sorted=False)
            '''
            edge_type_dict[('author', 'to', 'author')] = a_graph
            edge_type_dict[('author', 'to', 'group')] = ag_graph
            edge_type_dict[('group', 'to', 'author')] = torch.stack((ag_graph[1], ag_graph[0]))
            node_type_dict['author'] = x_author
            node_type_dict['group'] = x_group
            node_type_dict = self.GraphBase.forward(node_type_dict, edge_type_dict)
            return node_type_dict['author'], node_type_dict['group']
        elif self.p.graph_based == 'MLP':
            each_i = torch.index_select(x_author, 0, ag_graph[0])  # 选出source node 对应的表征
            h_group = scatter(each_i, ag_graph[1], dim=0, reduce='mean', dim_size = group_num)
            h_group = self.group_lin(h_group)
            return x_author, h_group
        else:
            # 合并成同一个图，使用graph模型传递
            # 顺序为: group\cate\user\item
            x_union = torch.cat([x_author, x_group], dim=0)
            author_num = x_author.size(0)

            ag_graph = torch.stack((ag_graph[0], ag_graph[1] + author_num))
            ga_graph = torch.stack((ag_graph[1], ag_graph[0]))
            edge_uninon = torch.cat([a_graph, ag_graph, ga_graph], dim=-1)
            #edge_uninon = torch.cat([a_graph, ag_graph], dim=-1)

            edge_type = None
            if self.p.graph_based == 'RGCN' or self.p.graph_based == 'RGAT':
                edge_type = [0 for i in range(a_graph.size(-1))]
                edge_type.extend([1 for i in range(ag_graph.size(-1))])
                edge_type.extend([2 for i in range(ga_graph.size(-1))])
                edge_type = torch.LongTensor(edge_type).to(ag_graph.device)
                graph_union= edge_uninon
            else:
                graph_union = SparseTensor(
                    row=edge_uninon[1], col=edge_uninon[0],
                    value=None,
                    is_sorted=False)
            del edge_uninon
            x_union = self.GraphBase.forward(graph_union, x_union, edge_type)
            return x_union[:author_num], x_union[author_num:]


class LinkPredictor(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, score_method, i2g_method, view_num):
		super(LinkPredictor, self).__init__()

		self.score_method = score_method
		if i2g_method == 'MMAN' and self.score_method != 'mv_score':
			l_in_channels = view_num * in_channels
		else:
			l_in_channels = in_channels

		self.lins = torch.nn.ModuleList()
		self.lins.append(torch.nn.Linear(l_in_channels, hidden_channels))
		for _ in range(num_layers - 2):
			self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
		self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
		self.reset_parameters()
		self.dropout = dropout


	def reset_parameters(self):
		for lin in self.lins:
			lin.reset_parameters()

	#原有的添加三层MLP预测出分数
	def forward(self, x_i, x_j):
		# (batch, dim)/(batch, view_num, dim)
		if self.score_method == 'MLP':
			if len(x_i.size()) != 2:
				x_i = x_i.view(x_i.size(0),-1)
				x_j = x_j.view(x_i.size(0),-1)
			x = x_i * x_j
			for lin in self.lins[:-1]:
				x = lin(x)
				x = F.relu(x)
				x = F.dropout(x, p=self.dropout, training=self.training)
			x = self.lins[-1](x) #(batch, 1)
			return torch.sigmoid(x.squeeze(-1)) #(batch)

		elif self.score_method == 'mv_score':
			if len(x_i.size()) == 2:
				x_i = x_i.unsqueeze(1)
				x_j = x_j.unsqueeze(1)

			#view_list = x_j.chunk(x_j.size(1), 1)
			view_list = x_j
			multi_result = []
			# for view_s in view_list:
			for s in range(view_list.size(1)):
				view_s = view_list[:,s,:].unsqueeze(1)
				x = x_i * view_s.expand(-1, x_i.size(1), -1)  #(batch, s, dim)
				multi_result.append(x)
			x = torch.cat(multi_result, dim=1)  # s * (batch, s, dim) → (batch, s^2, dim)

			#MLP for each dot result
			x = x.view(-1, x_i.size(-1))  # (batch*s^2, dim)
			for lin in self.lins[:-1]:
				x = lin(x)
				x = F.relu(x)
				x = F.dropout(x, p=self.dropout, training=self.training)
			x = torch.sigmoid(self.lins[-1](x))  # (batch*s^2, 1)
			x, _ = torch.max(x.view(x_i.size(0), -1), -1)
			return x