import torch
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

class ARGCN_dep_conv(MessagePassing):
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
        super(ARGCN_dep_conv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_feature_dim = edge_feature_dim

        self.negative_slope = 0.1
        self.dropout = 0.4

        self.dep_emb_dim = 100

        self.neighbor_weight = Param(torch.Tensor(in_channels, out_channels))
        self.dep_embedding = torch.nn.Embedding(num_embeddings=50, embedding_dim=self.dep_emb_dim)
        self.edge_trans1 = Param(torch.Tensor(self.dep_emb_dim, 1))

        self.att_weight = Param(torch.Tensor(out_channels*2, 1))

        if root_weight:
            self.root_weight = Param(torch.Tensor(in_channels, out_channels))

        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.neighbor_weight)
        uniform(size, self.root_weight)

        uniform(size, self.edge_trans1)

        uniform(size, self.att_weight)

        uniform(size, self.bias)


    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)

    def message(self, x_i, x_j, edge_index_j, size_i, edge_type, edge_norm, ptr):

        alpha = self.dep_embedding(edge_type)
        alpha = torch.matmul(alpha, self.edge_trans1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = alpha.reshape(-1, 1)

        trans_x_i = torch.matmul(x_i, self.root_weight)
        trans_x_j = torch.matmul(x_j, self.root_weight)

        beta = torch.matmul(torch.cat([trans_x_i, trans_x_j], dim=-1), self.att_weight)
        beta = F.leaky_relu(beta, self.negative_slope)

        alpha = alpha + beta

        edge_weight = softmax(alpha, edge_index_j, ptr, size_i)

        edge_weight = F.dropout(edge_weight, p=self.dropout, training=self.training)

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
        return '{}({}, {}, num_relations={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.edge_feature_dim)

