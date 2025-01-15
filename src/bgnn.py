import torch
from torch import nn

def SparseTensor(row, col, value, sparse_sizes):
    return torch.sparse_coo_tensor(indices=torch.stack([row, col]),
                                   values=value,
                                   size=sparse_sizes)

def gcn_norm(edge_index, add_self_loops=True):
    adj_t = edge_index.to_dense()
    if add_self_loops:
        adj_t = adj_t+torch.eye(*adj_t.shape, device=edge_index.device)
    deg = adj_t.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.)

    adj_t.mul_(deg_inv_sqrt.view(-1, 1))
    adj_t.mul_(deg_inv_sqrt.view(1, -1))
    edge_index = adj_t.to_sparse()
    return edge_index, None


class GCNConv(nn.Module):
    def __init__(self, size_u: int, size_v: int, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, dropout: float=0.4, share: bool= False,**kwargs):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout)
        self.share = share
        self.size_u = size_u
        self.size_v = size_v
        self.leaky_relu = nn.LeakyReLU()

        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None
        self._cached_adj_t = None

        if self.share:
            self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
            nn.init.xavier_uniform_(self.weight)
        else:
            self.weightr = nn.Parameter(torch.Tensor(in_channels, out_channels))
            nn.init.xavier_uniform_(self.weightr)
            self.weightd = nn.Parameter(torch.Tensor(in_channels, out_channels))
            nn.init.xavier_uniform_(self.weightd)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        # self.bn = nn.BatchNorm1d(in_channels)
        self.reset_parameters()

    def reset_parameters(self):
        # if self.bias is not None:
        #     nn.init.xavier_uniform_(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x, edge_index):
        if self.normalize:
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, self.add_self_loops)
            else:
                edge_index, edge_weight = cache[0], cache[1]
        # x = self.bn(x)
        if self.share:
            x = torch.matmul(x, self.weight)
        else:
            R = torch.matmul(x[:self.size_u,:],self.weightr)
            D = torch.matmul(x[self.size_u:,:],self.weightd)
            x = torch.cat([R,D],0)

        out = edge_index@x
        if self.bias is not None:
            out += self.bias
        out = self.dropout(out)
        out = self.leaky_relu(out)
    
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)