import torch
from torch import nn, optim
from functools import partial
from .model_help import BaseModel
from .dataset import FullGraphData
from . import MODEL_REGISTRY
from .gnn import FeaTrans


def SparseTensor(row, col, value, sparse_sizes):
    return torch.sparse_coo_tensor(indices=torch.stack([row, col]),
                                   values=value,
                                   size=sparse_sizes)


class EdgeDropout(nn.Module):
    def __init__(self, keep_prob=0.5):
        super(EdgeDropout, self).__init__()
        assert keep_prob > 0
        self.keep_prob = keep_prob
        self.register_buffer("p", torch.tensor(keep_prob))

    def forward(self, edge_index, edge_weight):
        if self.training:
            mask = torch.rand(edge_index.shape[1], device=edge_weight.device)
            mask = torch.floor(mask + self.p).type(torch.bool)
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask] / self.p
        return edge_index, edge_weight

    def forward2(self, edge_index):
        if self.training:
            mask = ((torch.rand(edge_index._values().size()) + (self.keep_prob)).floor()).type(torch.bool)
            rc = edge_index._indices()[:, mask]
            val = edge_index._values()[mask] / self.p
            return torch.sparse.FloatTensor(rc, val)
        return edge_index

    def __repr__(self):
        return '{}(keep_prob={})'.format(self.__class__.__name__, self.keep_prob)


class EncoderLayer(nn.Module):
    def __init__(self, size_u, size_v, in_features=64, out_features=64, qk_dim=64, attention_dropout_rate=0.3,
                 dropout=0.4, head_size=4, gamma=0.5, belta=0.5, temperature=0.1,bias=True, share=True, **kwargs):
        super(EncoderLayer, self).__init__()
        self.size_u = size_u
        self.size_v = size_v
        self.num_nodes = size_u + size_v
        self.share = share
        self.belta = belta
        self.con = BaseModel()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.a_encoder_a = FeaTrans(size_u=size_u, size_v=size_v, in_features=in_features, out_features=out_features,
                                    qk_dim=qk_dim,v_dim=out_features, head_size=head_size,
                                    attention_dropout_rate=attention_dropout_rate, dropout=dropout,
                                    gamma=gamma, bias=bias, share=share)
        self.a_encoder_s = FeaTrans(size_u=size_u, size_v=size_v, in_features=in_features, out_features=out_features,
                                    qk_dim=qk_dim,v_dim=out_features, head_size=head_size,
                                    attention_dropout_rate=attention_dropout_rate, dropout=dropout,
                                    gamma=gamma, bias=bias, share=share)
        # self.a_encoder_a = bgnn.GCNConv(size_u=size_u, size_v=size_v, in_channels=in_features, out_channels=out_features,
        #                                  add_self_loops=False,bias=bias,dropout=dropout, share=share)
        # self.a_encoder_s = bgnn.GCNConv(size_u=size_u, size_v=size_v, in_channels=in_features, out_channels=out_features,
        #                                   add_self_loops=False,bias=bias,dropout=dropout, share=share)

    def forward(self, a_x, s_x, a_edge_index, s_edge_index):
        sx1 = self.a_encoder_a(a_x, a_edge_index)
        ax1 = self.a_encoder_a(s_x, a_edge_index)
        # sx2 = self.a_encoder_s(s_x, s_edge_index)

        ax2 = self.a_encoder_s(a_x, s_edge_index)
        sx2 = self.a_encoder_s(s_x, s_edge_index)
        # ax2 = self.a_encoder_a(a_x, s_edge_index)

        # drug_loss = self.con.contract_loss(sx1[:self.size_u, :], sx2[:self.size_u, :], self.temperature) + \
        #             self.con.contract_loss(ax1[:self.size_u, :], ax2[:self.size_u, :], self.temperature)
        # dis_loss = self.con.contract_loss(sx1[self.size_u:, :], sx2[self.size_u:, :], self.temperature) + \
        #            self.con.contract_loss(ax1[self.size_u:, :], ax2[self.size_u:, :], self.temperature)
        drug_loss = 0
        dis_loss = 0

        ax = ax1 + ax2
        sx = sx1 + sx2

        # ax = ax2 + sx1

        return ax, sx, (drug_loss + dis_loss) / (self.size_u + self.size_v)


class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""

    def __init__(self, size_u, size_v, input_dim=None, dropout=0.4, alpha = 0.5, act=nn.Sigmoid, double=True):
        super(InnerProductDecoder, self).__init__()
        self.size_u = size_u
        self.size_v = size_v
        self.dropout = nn.Dropout(dropout)
        self.double = double
        self.alpha = alpha
        if input_dim:
            self.weights = nn.Linear(input_dim, input_dim, bias=False)
            nn.init.xavier_uniform_(self.weights.weight)
        self.act = act() if act is not None else nn.Identity()

    def forward(self, a_feature, s_feature):

        R = a_feature[:self.size_u]
        D = a_feature[self.size_u:]
        s_R = s_feature[:self.size_u]
        s_D = s_feature[self.size_u:]
        if hasattr(self, "weights"):
            D = self.weights(D)
        if self.double:
            score_r = R @ D.T
            score_r = self.act(score_r)

            score_d = s_D @ s_R.T
            score_d = self.act(score_d.T)

            score = (score_r + score_d) / 2.0
        else:
            score_r = self.act(R @ D.T)
            score = score_r

        return score


class Embedding(nn.Module):
    def __init__(self, size_u, size_v, embedding_dim, dropout=0.5):
        super(Embedding, self).__init__()
        self.num_nodes = size_u + size_v
        self.size_u = size_u
        self.size_v = size_v

        self.embedding = nn.Sequential(nn.Linear(self.num_nodes, embedding_dim, bias=False),nn.Dropout(dropout))

        self.output_dim = embedding_dim

    def forward(self, a_x, s_x):
        ax_emb = self.embedding(a_x)
        sx_emb = self.embedding(s_x)

        return ax_emb, sx_emb


@MODEL_REGISTRY.register()
class DFDRNN(BaseModel):
    DATASET_TYPE = "FullGraphDataset"

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DFDRNN model config")
        parser.add_argument("--embedding_dim", default=128, type=int)
        parser.add_argument("--qk_dim", default=128, type=int)
        parser.add_argument("--lr", type=float, default=8e-3)
        parser.add_argument("--layer_num", default=3, type=int)
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--attention_dropout", type=float, default=0.3)
        parser.add_argument("--edge_dropout", default=0.2, type=float)
        parser.add_argument("--head_size", default=2, type=int)
        parser.add_argument("--loss_fn", type=str, default="bce", choices=["bce", "focal"])
        parser.add_argument("--gamma", type=float, default=0.3)
        return parent_parser

    def __init__(self, size_u, size_v, dropout=0.4, bias=True, qk_dim=64, embedding_dim=64,
                 edge_dropout=0.2, lr=0.05, pos_weight=1.0, head_size=4, attention_dropout=0.5, hidden_layers=(64, 32),
                 layer_num=3, gamma=0.3, belta=1.0, loss_fn="bce", temperature=0.1, share=False, **kwargs):
        super(DFDRNN, self).__init__()
        self.size_u = size_u
        self.size_v = size_v
        self.num_nodes = size_u + size_v
        self.use_embedding = False
        self.in_dim = self.num_nodes

        cached = True if edge_dropout == 0.0 else False
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.layer_num = layer_num

        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.loss_fn = partial(self.bce_loss_fn, pos_weight=self.pos_weight)

        self.edge_dropout = EdgeDropout(keep_prob=1 - edge_dropout)
        self.short_embedding = Embedding(embedding_dim=embedding_dim, size_u=size_u, size_v=size_v, dropout=dropout)

        encoder = [EncoderLayer(size_u=size_u, size_v=size_v, in_features=embedding_dim, out_features=embedding_dim,
                                qk_dim=qk_dim, attention_dropout_rate=attention_dropout, dropout=dropout,
                                head_size=head_size, gamma=gamma, belta=belta, temperature=temperature, bias=bias, share=share, **kwargs)]
        for layer in range(1, layer_num):
            encoder.append(
                EncoderLayer(size_u=size_u, size_v=size_v, in_features=embedding_dim, out_features=embedding_dim,
                             qk_dim=qk_dim, attention_dropout_rate=attention_dropout, dropout=dropout,
                             head_size=head_size, gamma=gamma, belta=belta, temperature=temperature, bias=bias, share=share, **kwargs))
        self.encoders = nn.ModuleList(encoder)
        # self.attention = nn.Parameter(torch.tensor([[[1]],[[1/2]],[[1/3]]]))
        self.attention = nn.Parameter(torch.ones(layer_num, 1, 1) / layer_num)
        self.decoder = InnerProductDecoder(size_u=size_u, size_v=size_v, input_dim=0,
                                           dropout=dropout)
        # self.decoder = InteractionDecoder(embedding_dim=embedding_dim,size_u=size_u,size_v=size_v)
        self.save_hyperparameters()

    def step(self, batch: FullGraphData):
        s_x = batch.s_x
        a_edge_index, a_edge_weight = batch.a_edge[:2]
        s_edge_index, s_edge_weight = batch.s_edge[:2]
        label = batch.label
        predict,contract_loss = self.forward(s_x, a_edge_index, a_edge_weight, s_edge_index, s_edge_weight)

        # if not self.training:
        predict = predict[batch.valid_mask.reshape(*predict.shape)]
        label = label[batch.valid_mask]
        ans = self.loss_fn(predict=predict, label=label)

        ans["predict"] = predict.reshape(-1)
        ans["label"] = label.reshape(-1)
        return ans

    def forward(self, s_x, a_edge_index, a_edge_weight, s_edge_index, s_edge_weight):

        a_edge_index, a_edge_weight = self.edge_dropout(a_edge_index, a_edge_weight)

        a_edge_index = SparseTensor(row=a_edge_index[0], col=a_edge_index[1],
                                    value=a_edge_weight,
                                    sparse_sizes=(self.num_nodes, self.num_nodes))
        s_edge_index = SparseTensor(row=s_edge_index[0], col=s_edge_index[1],
                                    value=s_edge_weight,
                                    sparse_sizes=(self.num_nodes, self.num_nodes))

        #######
        a_x = a_edge_index.to_dense()

        ax, sx = self.short_embedding(a_x, s_x)
        a_x, s_x = ax, sx
        layerout_a = [ax]
        layerout_s = [sx]
        contract_loss = 0
        for encoder in self.encoders:
            ax, sx, loss = encoder(a_x, s_x, a_edge_index, s_edge_index)
            a_x = ax + layerout_a[-1]
            s_x = sx + layerout_s[-1]
            contract_loss += loss
            layerout_a.append(a_x)
            layerout_s.append(s_x)
        ax = torch.stack(layerout_a[1:])
        sx = torch.stack(layerout_s[1:])
        attention = torch.softmax(self.attention, dim=0)
        ax = torch.sum(ax * attention, dim=0)
        sx = torch.sum(sx * attention, dim=0)
        contract_loss = contract_loss/self.layer_num

        score = self.decoder(ax, sx)

        return score,contract_loss

    def training_step(self, batch, batch_idx=None):
        return self.step(batch)

    def validation_step(self, batch, batch_idx=None):
        return self.step(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=.1 * self.lr, weight_decay=1e-5)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1 * self.lr, max_lr=self.lr,
                                                   gamma=0.995, mode="exp_range", step_size_up=40,
                                                   cycle_momentum=False)
        return [optimizer], [lr_scheduler]
