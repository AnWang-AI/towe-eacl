import torch
from torch import nn
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing

import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

import math

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]

class ARGCN_dep_distance_conv(MessagePassing):
    """
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.

        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, edge_feature_dim=2,
                 root_weight=True, bias=True, **kwargs):
        super(ARGCN_dep_distance_conv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_feature_dim = edge_feature_dim

        self.negative_slope = 0.2
        self.dropout = 0.5

        self.neighbor_weight = Param(torch.Tensor(in_channels, out_channels))

        self.dep_emb_dim = 10
        self.dep_embedding = torch.nn.Embedding(num_embeddings=50, embedding_dim=self.dep_emb_dim)
        self.edge_trans1 = Param(torch.Tensor(self.dep_emb_dim, 1))

        self.distance_emb_dim = 100

        self.distance_embedding = PositionalEmbedding(demb=self.distance_emb_dim)
        # self.distance_embedding = torch.nn.Embedding(num_embeddings=30, embedding_dim=self.distance_emb_dim)

        self.edge_trans2 = Param(torch.Tensor(self.distance_emb_dim, 1))

        self.att_dim = 5
        self.att_weight = Param(torch.Tensor(out_channels*2 + self.distance_emb_dim, self.att_dim))
        self.att_bias = Param(torch.Tensor(self.att_dim))

        if root_weight:
            self.root_weight = Param(torch.Tensor(in_channels, out_channels))

        else:
            self.register_parameter('root', None)

        self.Qusetion_weight = Param(torch.Tensor(in_channels, out_channels))
        self.Key_weight = Param(torch.Tensor(in_channels, out_channels))

        self.sum_weight = Param(torch.Tensor(self.dep_emb_dim+self.att_dim, 1))

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.neighbor_weight)
        uniform(size, self.root_weight)

        uniform(size, self.Qusetion_weight)
        uniform(size, self.Key_weight)

        uniform(size, self.edge_trans1)

        uniform(size, self.att_weight)
        uniform(size, self.att_bias)

        uniform(size, self.sum_weight)

        uniform(size, self.bias)

        torch.nn.init.xavier_normal_(self.dep_embedding.weight)

        # torch.nn.init.xavier_normal_(self.distance_embedding.weight)


    def forward(self, x, edge_index, edge_type, edge_distance, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_distance=edge_distance, edge_norm=edge_norm)

    def message(self, x_i, x_j, edge_index_j, size_i, edge_type, edge_distance, edge_norm, ptr):

        alpha = self.dep_embedding(edge_type)
        # alpha = torch.matmul(alpha, self.edge_trans1)
        # alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = alpha.reshape(-1, self.dep_emb_dim)

        edge_distance = edge_distance.reshape(-1).float()
        gamma = self.distance_embedding(edge_distance)
        # gamma = torch.matmul(gamma, self.edge_trans2)
        # gamma = F.leaky_relu(gamma, self.negative_slope)
        gamma = gamma.reshape(-1, self.distance_emb_dim)
        # alpha = alpha + gamma

        trans_x_i = torch.matmul(x_i, self.Qusetion_weight)
        trans_x_j = torch.matmul(x_j, self.Qusetion_weight)

        beta = torch.matmul(torch.cat([trans_x_i, trans_x_j, gamma], dim=-1), self.att_weight)
        # beta = ((trans_x_i * trans_x_j).sum(dim=1)/ (trans_x_i.sum(dim=1) * trans_x_j.sum(dim=1) + 1e-4)).reshape(-1,1)
        beta = F.leaky_relu(beta, self.negative_slope)
        beta = softmax(beta, edge_index_j, ptr, size_i)
        beta = F.dropout(beta, p=self.dropout, training=self.training)

        edge_weight = torch.matmul(torch.cat([alpha, beta], dim=-1), self.sum_weight)
        edge_weight = F.leaky_relu(edge_weight, self.negative_slope)

        out = torch.matmul(x_j, self.neighbor_weight) * edge_weight

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root_weight is not None:
            if x is None:
                aggr_out = aggr_out + self.root_weight
            else:
                aggr_out = aggr_out + torch.matmul(x, self.root_weight)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,)

class ARGCN_dep_distance_conv_v2(MessagePassing):
    """
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.

        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, edge_feature_dim=2,
                 root_weight=True, bias=True, **kwargs):
        super(ARGCN_dep_distance_conv_v2, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_feature_dim = edge_feature_dim

        self.negative_slope = 0.1
        self.dropout = 0.2

        self.neighbor_weight = Param(torch.Tensor(in_channels, out_channels))

        self.dep_emb_dim = 1
        self.dep_embedding = torch.nn.Embedding(num_embeddings=50, embedding_dim=self.dep_emb_dim)
        self.edge_trans1 = Param(torch.Tensor(self.dep_emb_dim, 1))

        self.distance_emb_dim = 50
        self.distance_embedding = torch.nn.Embedding(num_embeddings=20, embedding_dim=self.distance_emb_dim)
        self.edge_trans2 = Param(torch.Tensor(self.distance_emb_dim, 1))

        self.att_weight = Param(torch.Tensor(out_channels*2 + self.distance_emb_dim, 1))
        self.att_bias = Param(torch.Tensor(1))

        if root_weight:
            self.root_weight = Param(torch.Tensor(in_channels, out_channels))

        else:
            self.register_parameter('root', None)

        self.Qusetion_weight = Param(torch.Tensor(in_channels, out_channels))
        self.Key_weight = Param(torch.Tensor(in_channels, out_channels))

        self.sum_weight = Param(torch.Tensor(self.dep_emb_dim+1, 1))
        self.sum_bias = Param(torch.Tensor(1))

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.neighbor_weight)
        uniform(size, self.root_weight)

        uniform(size, self.Qusetion_weight)
        uniform(size, self.Key_weight)

        uniform(size, self.edge_trans1)

        uniform(size, self.att_weight)
        uniform(size, self.att_bias)

        uniform(size, self.sum_weight)
        uniform(size, self.sum_bias)

        uniform(size, self.bias)

        torch.nn.init.xavier_normal_(self.dep_embedding.weight)

        torch.nn.init.xavier_normal_(self.distance_embedding.weight)


    def forward(self, x, edge_index, edge_type, edge_distance, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_distance=edge_distance, edge_norm=edge_norm)

    def message(self, x_i, x_j, edge_index_j, size_i, edge_type, edge_distance, edge_norm, ptr):

        alpha = self.dep_embedding(edge_type)
        # alpha = torch.matmul(alpha, self.edge_trans1)
        # alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = alpha.reshape(-1, self.dep_emb_dim)

        gamma = self.distance_embedding(edge_distance)
        # gamma = torch.matmul(gamma, self.edge_trans2)
        # gamma = F.leaky_relu(gamma, self.negative_slope)
        gamma = gamma.reshape(-1, self.distance_emb_dim)
        # alpha = alpha + gamma

        trans_x_i = torch.matmul(x_i, self.Qusetion_weight)
        trans_x_j = torch.matmul(x_j, self.Key_weight)

        beta = torch.matmul(torch.cat([trans_x_i, trans_x_j, gamma], dim=-1), self.att_weight) + self.att_bias
        # beta = ((trans_x_i * trans_x_j).sum(dim=1)/ (trans_x_i.sum(dim=1) * trans_x_j.sum(dim=1) + 1e-4)).reshape(-1,1)
        beta = F.leaky_relu(beta, self.negative_slope)

        beta = softmax(beta, edge_index_j, ptr, size_i)

        edge_weight = torch.matmul(torch.cat([alpha, beta], dim=-1), self.sum_weight) + self.sum_bias
        edge_weight = F.leaky_relu(edge_weight, self.negative_slope)

        out = torch.matmul(x_j, self.neighbor_weight) * edge_weight

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root_weight is not None:
            if x is None:
                aggr_out = aggr_out + self.root_weight
            else:
                aggr_out = aggr_out + torch.matmul(x, self.root_weight)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,)

class ARGCN_dep_distance_conv_multi_head(MessagePassing):
    """
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.

        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, edge_feature_dim=2, num_heads=8,
                 root_weight=True, bias=True, **kwargs):
        super(ARGCN_dep_distance_conv_multi_head, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_feature_dim = edge_feature_dim

        self.num_heads = num_heads

        self.negative_slope = 0.2
        self.dropout = 0.5

        self.neighbor_weight = Param(torch.Tensor(in_channels, out_channels * self.num_heads))

        self.dep_emb_dim = 10
        self.dep_embedding = torch.nn.Embedding(num_embeddings=50, embedding_dim=self.dep_emb_dim)
        self.edge_trans1 = Param(torch.Tensor(self.dep_emb_dim, 1))

        self.distance_emb_dim = 100

        self.distance_embedding = PositionalEmbedding(demb=self.distance_emb_dim)
        # self.distance_embedding = torch.nn.Embedding(num_embeddings=30, embedding_dim=self.distance_emb_dim)

        self.edge_trans2 = Param(torch.Tensor(self.distance_emb_dim, 1))

        if root_weight:
            self.root_weight = Param(torch.Tensor(in_channels, out_channels))

        else:
            self.register_parameter('root', None)

        self.Qusetion_weight = Param(torch.Tensor(in_channels, out_channels * self.num_heads))
        self.Key_weight = Param(torch.Tensor(in_channels, out_channels * self.num_heads))

        self.att_dim = 1
        self.att_weight = Param(torch.Tensor(out_channels*2 + self.distance_emb_dim, self.att_dim * self.num_heads))

        self.sum_weight = Param(torch.Tensor(self.dep_emb_dim+self.att_dim, 1))

        self.head_weight = Param(torch.Tensor(self.out_channels * self.num_heads, self.out_channels))

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.neighbor_weight)
        uniform(size, self.root_weight)

        uniform(size, self.Qusetion_weight)
        uniform(size, self.Key_weight)

        uniform(size, self.edge_trans1)

        uniform(size, self.att_weight)

        uniform(size, self.sum_weight)

        uniform(size, self.head_weight)

        uniform(size, self.bias)

        torch.nn.init.xavier_normal_(self.dep_embedding.weight)

        # torch.nn.init.xavier_normal_(self.distance_embedding.weight)


    def forward(self, x, edge_index, edge_type, edge_distance, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_distance=edge_distance, edge_norm=edge_norm)

    def message(self, x_i, x_j, edge_index_j, size_i, edge_type, edge_distance, edge_norm, ptr):

        alpha = self.dep_embedding(edge_type)
        alpha = alpha.reshape(-1, self.dep_emb_dim)

        edge_distance = edge_distance.reshape(-1).float()
        gamma = self.distance_embedding(edge_distance)
        # gamma = torch.matmul(gamma, self.edge_trans2)
        # gamma = F.leaky_relu(gamma, self.negative_slope)
        gamma = gamma.reshape(-1, self.distance_emb_dim)
        # alpha = alpha + gamma

        trans_x_i = torch.matmul(x_i, self.neighbor_weight)
        trans_x_j = torch.matmul(x_j, self.neighbor_weight)

        trans_neighbor = torch.matmul(x_j, self.neighbor_weight)

        heads = []

        for head_idx in range(self.num_heads):
            q = trans_x_i[:, head_idx * self.out_channels:(head_idx+1) * self.out_channels]
            k = trans_x_j[:, head_idx * self.out_channels:(head_idx+1) * self.out_channels]
            v = trans_neighbor[:, head_idx * self.out_channels:(head_idx+1) * self.out_channels]

            attention_weight = self.att_weight[:, head_idx * self.att_dim:(head_idx+1) * self.att_dim]

            beta = torch.matmul(torch.cat([q, k, gamma], dim=-1), attention_weight)
            beta = F.leaky_relu(beta, self.negative_slope)
            beta = softmax(beta, edge_index_j, ptr, size_i)
            beta = F.dropout(beta, p=self.dropout, training=self.training)

            edge_weight = torch.matmul(torch.cat([alpha, beta], dim=-1), self.sum_weight)
            edge_weight = F.leaky_relu(edge_weight, self.negative_slope)

            head = v * edge_weight

            heads.append(head)

        out = torch.cat(heads, dim=-1)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = torch.matmul(out, self.head_weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root_weight is not None:
            if x is None:
                aggr_out = aggr_out + self.root_weight
            else:
                aggr_out = aggr_out + torch.matmul(x, self.root_weight)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,)

class ARGCN_dep_distance_conv_multi_head_v2(MessagePassing):
    """
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.

        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, edge_feature_dim=2, num_heads=4,
                 root_weight=True, bias=True, **kwargs):
        super(ARGCN_dep_distance_conv_multi_head_v2, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_feature_dim = edge_feature_dim

        self.num_heads = num_heads

        self.negative_slope = 0.2
        self.dropout = 0.5

        self.neighbor_weight = Param(torch.Tensor(in_channels, out_channels * self.num_heads))

        self.dep_emb_dim = 10
        self.dep_embedding = torch.nn.Embedding(num_embeddings=50, embedding_dim=self.dep_emb_dim)
        self.edge_trans1 = Param(torch.Tensor(self.dep_emb_dim, 1))

        self.distance_emb_dim = 100

        self.distance_embedding = PositionalEmbedding(demb=self.distance_emb_dim)
        # self.distance_embedding = torch.nn.Embedding(num_embeddings=30, embedding_dim=self.distance_emb_dim)

        self.edge_trans2 = Param(torch.Tensor(self.distance_emb_dim, 1))

        if root_weight:
            self.root_weight = Param(torch.Tensor(in_channels, out_channels))

        else:
            self.register_parameter('root', None)

        self.Qusetion_weight = Param(torch.Tensor(in_channels, out_channels * self.num_heads))
        self.Key_weight = Param(torch.Tensor(in_channels, out_channels * self.num_heads))

        self.att_dim = 1
        self.att_weight = Param(torch.Tensor(out_channels*2 + self.distance_emb_dim, self.att_dim * self.num_heads))

        self.sum_weight = Param(torch.Tensor(self.dep_emb_dim+self.att_dim, 1))

        self.head_weight = Param(torch.Tensor(self.out_channels, self.out_channels))

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.neighbor_weight)
        uniform(size, self.root_weight)

        uniform(size, self.Qusetion_weight)
        uniform(size, self.Key_weight)

        uniform(size, self.edge_trans1)

        uniform(size, self.att_weight)

        uniform(size, self.sum_weight)

        uniform(size, self.head_weight)

        uniform(size, self.bias)

        torch.nn.init.xavier_normal_(self.dep_embedding.weight)

        # torch.nn.init.xavier_normal_(self.distance_embedding.weight)


    def forward(self, x, edge_index, edge_type, edge_distance, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_distance=edge_distance, edge_norm=edge_norm)

    def message(self, x_i, x_j, edge_index_j, size_i, edge_type, edge_distance, edge_norm, ptr):

        alpha = self.dep_embedding(edge_type)
        alpha = alpha.reshape(-1, self.dep_emb_dim)

        edge_distance = edge_distance.reshape(-1).float()
        gamma = self.distance_embedding(edge_distance)
        # gamma = torch.matmul(gamma, self.edge_trans2)
        # gamma = F.leaky_relu(gamma, self.negative_slope)
        gamma = gamma.reshape(-1, self.distance_emb_dim)
        # alpha = alpha + gamma

        trans_x_i = torch.matmul(x_i, self.neighbor_weight)
        trans_x_j = torch.matmul(x_j, self.neighbor_weight)

        trans_neighbor = torch.matmul(x_j, self.neighbor_weight)

        heads = []

        for head_idx in range(self.num_heads):
            head_dim = int(self.out_channels/self.num_heads)
            q = trans_x_i[:, head_idx * self.out_channels:(head_idx+1) * self.out_channels]
            k = trans_x_j[:, head_idx * self.out_channels:(head_idx+1) * self.out_channels]
            v = trans_neighbor[:, head_idx * head_dim:(head_idx+1) * head_dim]

            attention_weight = self.att_weight[:, head_idx * self.att_dim:(head_idx+1) * self.att_dim]

            # print(q.shape,k.shape, attention_weight.shape)
            beta = torch.matmul(torch.cat([q, k, gamma], dim=-1), attention_weight)
            beta = F.leaky_relu(beta, self.negative_slope)
            beta = softmax(beta, edge_index_j, ptr, size_i)
            beta = F.dropout(beta, p=self.dropout, training=self.training)

            edge_weight = torch.matmul(torch.cat([alpha, beta], dim=-1), self.sum_weight)
            edge_weight = F.leaky_relu(edge_weight, self.negative_slope)

            head = v * edge_weight

            heads.append(head)

        out = torch.cat(heads, dim=-1)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = torch.matmul(out, self.head_weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root_weight is not None:
            if x is None:
                aggr_out = aggr_out + self.root_weight
            else:
                aggr_out = aggr_out + torch.matmul(x, self.root_weight)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,)