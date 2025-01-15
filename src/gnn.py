import torch
from torch import nn


def SparseTensor(row, col, value, sparse_sizes):
    return torch.sparse_coo_tensor(indices=torch.stack([row, col]),
                                   values=value,
                                   size=sparse_sizes)


class Self_Attention(nn.Module):
    def __init__(self, in_features, qk_dim, v_dim, attention_dropout_rate, head_size):
        super(Self_Attention, self).__init__()

        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.head_size = head_size

        self.att_dim = att_dim = qk_dim // head_size
        self.scale = att_dim ** -0.5
        self.aru = nn.LeakyReLU()
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.linear_q = nn.Linear(in_features, qk_dim, bias=False)
        self.linear_k = nn.Linear(in_features, qk_dim, bias=False)
        # self.linear_v = nn.Linear(in_features, self.v_dim, bias=False)


    def forward(self, x1, x2, v, adj):
        if adj.is_sparse:
            adj = adj.to_dense()

        batch_size= x1.size(0)

        q = x1.view(batch_size, self.head_size, -1)
        k = x2.view(batch_size, self.head_size, -1)
        v = v.view(batch_size, self.head_size, -1)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1).transpose(1, 2)
        v = v.transpose(0, 1)

        q = q * self.scale
        x = torch.matmul(q, k)

        zero_vec = -9e15 * torch.ones_like(adj)
        x = torch.where(adj > 0, x, zero_vec)
        x = torch.softmax(x, dim=-1)
        # x = self.att_dropout(x)
        x = x.matmul(v)

        x = x.transpose(0, 1).contiguous()
        x = x.view(batch_size, -1)

        return x

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__, self.qk_dim,
                                       self.v_dim, self.head_size)



class FeaTrans(nn.Module):
    def __init__(self, size_u,size_v,in_features,out_features,qk_dim,v_dim,head_size,attention_dropout_rate,dropout,gamma,bias=True,share=False):
        super(FeaTrans, self).__init__(),
        self.size_u=size_u
        self.size_v=size_v
        self.in_features = in_features
        self.out_features = out_features
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.share = share
        self.layer_norm = nn.LayerNorm(out_features)
        self.linear_q = nn.Linear(in_features, qk_dim, bias=False)
        self.linear_k = nn.Linear(in_features, qk_dim, bias=False)

        self.SAT = Self_Attention(in_features=in_features,qk_dim=qk_dim,v_dim=v_dim,
                                  attention_dropout_rate=attention_dropout_rate,head_size=head_size)


        if self.share:
            self.weight = nn.Parameter(torch.Tensor(out_features, out_features))
            nn.init.xavier_uniform_(self.weight)
        else:
            self.weightr = nn.Parameter(torch.Tensor(out_features, out_features))
            nn.init.xavier_uniform_(self.weightr)
            self.weightd = nn.Parameter(torch.Tensor(out_features, out_features))
            nn.init.xavier_uniform_(self.weightd)

        self.linear_v = nn.Linear(in_features,v_dim,bias=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.gamma = gamma

    def forward(self,x,edge_index):

        # x = self.layer_norm(x)
        x1 = self.linear_q(x)
        x2 = self.linear_k(x)
        v = self.linear_v(x)

        sat_x = self.SAT(x1,x2,v,edge_index)
    
        x = sat_x
        out = x
        if self.share:
            out = torch.matmul(x, self.weight)
        else:
            R = torch.matmul(x[:self.size_u,:],self.weightr)
            D = torch.matmul(x[self.size_u:,:],self.weightd)
            out = torch.cat([R,D],0)
        if self.bias is not None:
            out += self.bias
        out = self.dropout(out)
        out = self.leaky_relu(out)

        return out








