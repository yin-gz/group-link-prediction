from helper import *
from torch_geometric.nn import Set2Set, GraphMultisetTransformer
from model.graph_layer import *
from torch_geometric.nn.dense.linear import Linear



class GE(torch.nn.Module):
	"""
	Graph-based encoders, implemented by pytorch geometric.
	"""
	def __init__(self, params, node_type_dict = None, edge_type_dict = None):
		super(GE, self).__init__()
		self.p = params
		#graph encoder
		if self.p.graph_based == 'GCN':
			self.GraphBase = GCN(self.p.gcn_layer, self.p.init_dim, self.p.gcn_dim, self.p.embed_dim, params=self.p)
		elif self.p.graph_based == 'GAT':
			self.GraphBase = GAT(self.p.gcn_layer, self.p.init_dim, self.p.gcn_dim, self.p.embed_dim, params=self.p)
		elif self.p.graph_based == 'GraphSage':
			self.GraphBase = GraphSAGE(self.p.gcn_layer, self.p.init_dim, self.p.gcn_dim, self.p.embed_dim, params=self.p)
		elif self.p.graph_based == 'RGCN':
			self.GraphBase = RGCN(self.p.gcn_layer, self.p.init_dim, self.p.gcn_dim, self.p.embed_dim, params=self.p)
		elif self.p.graph_based == 'HGT':
			self.GraphBase = HGT(node_type_dict, edge_type_dict, params=self.p)


class GroupAgg(torch.nn.Module):
	"""
	Aggregation functions to aggregate individuals' embeddings to group embeddings.
	"""
	def __init__(self, params):
		super(GroupAgg, self).__init__()
		self.p = params
		# For:　author and item liner layers
		self.individual_lin = Linear(self.p.embed_dim, self.p.embed_dim)
		self.group_lin = Linear(self.p.embed_dim, self.p.embed_dim)
		self.individual_lin.reset_parameters()
		self.group_lin.reset_parameters()

		#i2g model
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
			self.lin_src_K = Linear(self.p.embed_dim, self.heads * self.p.gcn_dim, False, weight_initializer='glorot')
			self.lin_src_V = Linear(self.p.embed_dim, self.heads * self.p.gcn_dim, False, weight_initializer='glorot')
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
			self.set_pool = Set2Set(self.p.embed_dim, processing_steps=1) #global_add_pool/global_mean_pool/global_max_pool/GlobalAttention
			self.group_lin = Linear(2*self.p.embed_dim, self.p.embed_dim)
		elif self.p.i2g_method == 'MAB':
			self.mab_pool = GraphMultisetTransformer(self.p.embed_dim, self.p.embed_dim, self.p.embed_dim, pool_sequences=['GMPool_I'], num_heads=self.p.att_head)


	def i2g_degree(self, each_i, agg_graph, degree):
		each_i = self.individual_lin(each_i)
		alpha = torch.index_select(degree, 0, agg_graph[0]).float()
		alpha = softmax(alpha, agg_graph[1], num_nodes=int(agg_graph[0].size(0)))
		x_i = torch.mul(alpha.unsqueeze(1), each_i)
		h_group = scatter(x_i, agg_graph[1], dim=0, reduce='sum')
		h_group = self.group_lin(h_group)
		return h_group, alpha

	def i2g_att(self, each_i, agg_graph, h_group=None):
		h_group = scatter(each_i, agg_graph[1], dim=0,reduce='mean')  # average individual emb to get initial h_group
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
		return h_group, alpha

	def i2g_MMAN(self, each_i, agg_graph, num_group):
		x_i = self.individual_lin(each_i)
		x_i_K = self.lin_src_K(x_i).view(-1, self.heads, self.p.gcn_dim).unsqueeze(1).expand(-1, self.p.view_num, -1,
																							 -1)  # (n_edge, view_num, heads, dim)
		x_i_V = self.lin_src_V(x_i).view(-1, self.heads, self.p.gcn_dim).unsqueeze(1).expand(-1, self.p.view_num, -1,
																							 -1)  # (n_edge, view_num, heads, dim)
		alpha = (self.Q * x_i_K).sum(dim=-1)  # (n_edge, view_num, heads)
		alpha = alpha / math.sqrt(x_i_K.size(-1))
		alpha = softmax(alpha, agg_graph[1], num_nodes=int(agg_graph[0].size(0)), dim=0)  # softmax for source nodes to the same target, (n_edge, view_num, heads)
		out = (x_i_V * (alpha.unsqueeze(-1))).mean(dim=2)  # (n_edge, view_num, heads, dim)  * (n_edge, view_num, heads,1 ) → (n_edge, view_num, heads, dim) → (n_edge, view_num, dim)
		h_group = scatter(out, agg_graph[1], dim=0, reduce='sum', dim_size=num_group)  # (n_group, view_num, dim)
		h_group = self.ln(h_group)
		return h_group, alpha

	def forward(self, h_i, i2g_graph, i_degree, group_num = 0 , h_group = None):
		"""
		Args:
			h_i: [n_individuals, emb_dim]
			i2g_graph: denote the individuals belong to which group, [2, n_pairs]
			i_degree: [n_individuals]
			h_group: [n_groups, emb_dim]

		Return:
			h_group: [n_groups, emb_dim]
			alpha: the weights distribution of each group, [n_individuals]

		"""
		each_i = torch.index_select(h_i, 0, i2g_graph[0])
		if self.p.i2g_method == 'degree':
			return self.i2g_degree(each_i, i2g_graph, i_degree)
		elif self.p.i2g_method == 'avg':
			x_i = self.individual_lin(each_i)  # 线性变换
			h_group = scatter(x_i, i2g_graph[1], dim=0, reduce='mean')
			h_group = self.group_lin(h_group)
			return h_group, None
		elif self.p.i2g_method == 'att':
			self.heads = self.p.att_head
			return self.i2g_att(each_i, i2g_graph, h_group)
		elif self.p.i2g_method == 'set2set':
			h_group = self.set_pool(each_i, i2g_graph[1])
			h_group = self.group_lin(h_group)
			return h_group, None
		elif self.p.i2g_method == 'MAB':
			h_group = self.mab_pool(each_i, i2g_graph[1])
			h_group = self.group_lin(h_group)
			return h_group, None
		elif self.p.i2g_method == 'MMAN':
			return self.i2g_MMAN(each_i, i2g_graph, group_num)



class B2GModel(GE):
	"""
	Model framework for bundle-to-group recommendation.
	"""
	def __init__(self, params, num_user = 0 , num_item = 0, num_group = 0,num_cate = 0):
		self.p = params
		node_type_dict = {'user': [], 'item': [], 'cate': [], 'group': []}
		edge_type_dict = {('user', 'to', 'item'): [], ('item', 'to', 'user'): [], ('item', 'to', 'cate'): [],
						  ('cate', 'to', 'item'): [], ('user', 'to', 'group'): [], ('group', 'to', 'user'): []}
		super(B2GModel, self).__init__(params, node_type_dict, edge_type_dict)

		self.num_user = num_user
		self.num_item = num_item
		self.num_group = num_group
		self.num_cate = num_cate
		self.drop = torch.nn.Dropout(self.p.dropout)

		#init embedding
		self.user_embedding_layer = nn.Embedding(self.num_user, self.p.init_dim)
		self.item_embedding_layer = nn.Embedding(self.num_item, self.p.init_dim)
		#user to group
		self.u2g_agg = GroupAgg(self.p)
		#item to cate
		self.i2c_agg = GroupAgg(self.p)
		#out
		self.group_out_lin = Linear(2 * self.p.embed_dim, self.p.embed_dim, weight_initializer='glorot')
		#self.group_out_lin.reset_parameters()
		self.user_embedding_layer.reset_parameters()
		self.item_embedding_layer.reset_parameters()
		self.h_user = None
		self.h_item = None

	def forward(self, x_user, x_item, ui_graph, ic_graph, ug_graph, x_user_index = None, x_item_index = None, node_num = None, u_degree = None, i_degree = None):
		"""
		Args:
			x_user: initial user embeddings
			x_item: initial item embeddings
			ui_graph: user-to-item graph, [2, edge_ui]
			ic_graph: item-to-cate graph, [2, edge_ic]
			ug_graph: user-to-group graph, [2, edge_ug]
			x_user_index: user index in a mini-batch (optional)
			x_item_index: item index in a mini-batch (optional)
			node_num:  node count dict (optional)
			u_degree: [n_users]
			i_degree: [n_items]

		Return:
			h_group: [n_groups, emb_dim]
			h_cate: [n_cates, emb_dim]
			self.h_user: [n_users, emb_dim]
			self.h_item: [n_items, emb_dim]

		"""
		if x_user is None:
			x_user = self.user_embedding_layer(torch.LongTensor([idx for idx in range(self.num_user)]).to(ui_graph.device))
		if x_item is None:
			x_item = self.item_embedding_layer(torch.LongTensor([idx for idx in range(self.num_item)]).to(ui_graph.device))
		if x_user_index is not None:
			x_user = torch.index_select(x_user, 0, x_user_index)
			x_item = torch.index_select(x_item, 0, x_item_index)
		if node_num is not None:
			group_num = node_num['group']
			cate_num = node_num['cate']
		else:
			group_num = self.num_group
			cate_num = self.num_cate

		x_group = torch.zeros(group_num, self.p.init_dim).to(ui_graph.device)
		x_cate = torch.zeros(cate_num, self.p.init_dim).to(ui_graph.device)
		agg_graph_ug = ug_graph  # for aggregating
		agg_graph_ic = ic_graph  # for aggregating

		# graph encoder
		if self.p.graph_based == 'HGT':
			edge_type_dict = {}
			node_type_dict = {}
			edge_type_dict[('user', 'to', 'item')] = ui_graph
			edge_type_dict[('item', 'to', 'user')] = torch.stack((ui_graph[1], ui_graph[0]))
			edge_type_dict[('item', 'to', 'cate')] = ic_graph
			edge_type_dict[('cate', 'to', 'item')] = torch.stack((ic_graph[1], ic_graph[0]))
			edge_type_dict[('user', 'to', 'group')] = ug_graph
			edge_type_dict[('group', 'to', 'user')] = torch.stack((ug_graph[1], ug_graph[0]))
			node_type_dict['user'] = x_user
			node_type_dict['item'] = x_item
			node_type_dict['cate'] = x_cate
			node_type_dict['group'] = x_group
			node_type_dict = self.GraphBase(node_type_dict, edge_type_dict)
			h_group = node_type_dict['group']
			h_cate = node_type_dict['cate']
			self.h_user = node_type_dict['user']
			self.h_item = node_type_dict['item']
		else:
			# index sequence:　group\cate\user\item
			x_union = torch.cat([x_group, x_cate, x_user, x_item], dim=0)
			ui_graph = torch.stack((ui_graph[0] + group_num + cate_num, ui_graph[1] + group_num + cate_num + self.num_user))
			iu_graph = torch.stack((ui_graph[1], ui_graph[0]))
			ic_graph = torch.stack((ic_graph[0] + group_num + cate_num + self.num_user, ic_graph[1] + group_num))
			ci_graph = torch.stack((ic_graph[1], ic_graph[0]))
			ug_graph = torch.stack((ug_graph[0] + group_num + cate_num, ug_graph[1]))
			gu_graph = torch.stack((ug_graph[1], ug_graph[0]))
			edge_uninon = torch.cat([ui_graph, iu_graph, ic_graph, ug_graph, ci_graph, gu_graph], dim=-1)
			edge_type = None
			if self.p.graph_based == 'RGCN' or self.p.graph_based == 'RGAT':
				edge_type = [0 for i in range(ui_graph.size(-1))]
				edge_type.extend([1 for i in range(iu_graph.size(-1))])
				edge_type.extend([2 for i in range(ic_graph.size(-1))])
				edge_type.extend([3 for i in range(ug_graph.size(-1))])
				edge_type.extend([4 for i in range(ci_graph.size(-1))])
				edge_type.extend([5 for i in range(gu_graph.size(-1))])
				edge_type = torch.LongTensor(edge_type).to(ui_graph.device)
			x_union = self.GraphBase(edge_uninon, x_union, edge_type)
			h_group = x_union[:group_num]
			h_cate = x_union[group_num:group_num+cate_num]
			self.h_user = x_union[group_num+cate_num:group_num+cate_num+self.num_user]
			self.h_item = x_union[-self.num_item:]

		if self.p.only_GE:
			return h_group, h_cate, self.h_user, self.h_item

		#user2group aggregate/item2cate aggregate
		if self.p.i2g_method is not None:
			h_group, att_user = self.u2g_agg(self.h_user, agg_graph_ug, u_degree, group_num = group_num , h_group = h_group)
			h_cate, att_item = self.i2c_agg(self.h_item, agg_graph_ic, i_degree, group_num = cate_num , h_group = h_cate)
		else:
			h_group = None
			h_cate = None

		return h_group, h_cate, self.h_user, self.h_item


class G2GModel(GE):
	"""
	Model framework for acdemic group link prediction.
	"""
	def __init__(self, params, num_author=0, num_group=0):
		self.p = params
		node_type_dict = {'author': [], 'group': []}
		edge_type_dict = {('author', 'to', 'author'): [], ('author', 'to', 'group'): [],
						  ('group', 'to', 'author'): []}
		super(G2GModel, self).__init__(params, node_type_dict, edge_type_dict)

		self.num_author = num_author
		self.num_group = num_group
		self.drop = torch.nn.Dropout(self.p.dropout)

		# init embedding
		if self.p.use_a_attribute is False:
			self.author_embedding_layer = nn.Embedding(self.num_author, self.p.init_dim)
			self.author_embedding_layer.reset_parameters()

		self.group_embedding_layer = nn.Embedding(self.num_group, self.p.init_dim)
		self.group_embedding_layer.reset_parameters()

		# For:　concat group
		self.group_out_lin = Linear(2 * self.p.embed_dim, self.p.embed_dim)
		# i2g model
		self.i2g_agg = GroupAgg(self.p)
		self.group_out_lin.reset_parameters()
		self.h_author = None

	def forward(self, x_author, x_group, aa_graph, ag_graph, degree=None, group_num=None, author_index=None):
		"""
		Args:
			x_author: [n_individuals, emb_dim]
			x_group: [n_groups, emb_dim]
			aa_graph: author collaboration graph, [2, n_author_collaborate]
			ag_graph: denote the individuals belong to which group, [2, n_pairs]
			degree: [n_authors]
			group_num: number of groups in the computing graph
			author_index: author index in a mini-batch (optional)

		Return:
			h_author: [n_authors, emb_dim]
			h_group: [n_groups, emb_dim]

		"""
		agg_graph = ag_graph
		if group_num is None:
			group_num = self.num_group
		if x_author is None:
			x_author = self.author_embedding_layer(
				torch.LongTensor([idx for idx in range(self.num_author)]).to(ag_graph.device))
		if x_group is None:
			x_group = torch.zeros(group_num, self.p.init_dim).to(ag_graph.device)
		if author_index is not None:
			x_author = torch.index_select(x_author, 0, author_index)

		if self.p.only_attribute:
			h_author = x_author
			h_group = x_group
		elif self.p.graph_based == 'HGT':
			edge_type_dict = {}
			node_type_dict = {}
			edge_type_dict[('author', 'to', 'author')] = aa_graph
			edge_type_dict[('author', 'to', 'group')] = ag_graph
			edge_type_dict[('group', 'to', 'author')] = torch.stack((ag_graph[1], ag_graph[0]), 0)
			node_type_dict['author'] = x_author
			node_type_dict['group'] = x_group
			node_type_dict = self.GraphBase(node_type_dict, edge_type_dict)
			h_author = node_type_dict['author']
			h_group = node_type_dict['group']
		else:
			x_union = torch.cat([x_author, x_group], dim=0)
			author_num = x_author.size(0)
			ag_graph = torch.stack((ag_graph[0], ag_graph[1] + author_num))
			ga_graph = torch.stack((ag_graph[1], ag_graph[0]))
			edge_uninon = torch.cat([aa_graph, ag_graph, ga_graph], dim=-1)

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
			x_union = self.GraphBase(graph_union, x_union, edge_type)
			h_author = x_union[:author_num]
			h_group = x_union[author_num:]

		# author2group aggregate
		if self.p.only_GE:
			return h_author, h_group
		else:
			h_group, att = self.i2g_agg(h_author, agg_graph, degree, group_num, h_group)
			return h_author, h_group



class LinkPredictor(torch.nn.Module):
	"""
	Link predicting layer, including MLP and the proposed multi-view scoring module.
	"""
	def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, score_method, i2g_method, view_num, param = None):
		super(LinkPredictor, self).__init__()
		self.p = param
		self.score_method = score_method
		if i2g_method == 'MMAN' and self.score_method != 'mv_score' and self.p.only_GE is False:
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

	def forward(self, x_i, x_j):
		"""
		Args:
			x_i: [batch, view_num, dim] (In MMAN) / [batch, dim] (Other Aggregators)
			x_j: [batch, view_num, dim] (In MMAN) / [batch, dim] (Other Aggregators)

		Return:
			x: link prediction scores of groups, [batch]

		"""
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