import torch
from collections import namedtuple
from src import DATA_TYPE_REGISTRY
from src.dataloader import Dataset

def gcn_norm(edge_index, add_self_loops=True):
    adj_t = edge_index.to_dense()
    if add_self_loops:
        adj_t = adj_t+torch.eye(*adj_t.shape)
    deg = adj_t.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.)

    adj_t.mul_(deg_inv_sqrt.view(-1, 1))
    adj_t.mul_(deg_inv_sqrt.view(1, -1))
    edge_index = adj_t.to_sparse()
    return edge_index

FullGraphData = namedtuple("FullGraphData", ["a_edge", "s_edge",  
                                             "s_x","label", "interaction_pair", "valid_mask"])

@DATA_TYPE_REGISTRY.register()
class FullGraphDataset(Dataset):
    def __init__(self, dataset, mask, fill_unkown=True, **kwargs):
        super(FullGraphDataset, self).__init__(dataset, mask, fill_unkown=True, **kwargs)
        assert fill_unkown, "fill_unkown need True!"
        self.data = self.build_data()

    def build_data(self):
        a_edge = self.get_union_edge(union_type="uv-vu")
        s_edge = self.get_union_edge(union_type="u-v")

        # a_x = self.getx_a()
        s_x = self.getx_s()

        # a_x = a_x.to_sparse()
        s_x = s_x.to_sparse()
        # norm_a_x = gcn_norm(edge_index=a_x, add_self_loops=False).to_dense()
        # a_x = norm_a_x * torch.norm(a_x) / torch.norm(norm_a_x)

        norm_s_x = gcn_norm(edge_index=s_x, add_self_loops=False).to_dense()
        s_x = norm_s_x * torch.norm(s_x) / torch.norm(norm_s_x)

        data = FullGraphData(s_x = s_x,
                             a_edge=a_edge,
                             s_edge=s_edge,
                             label=self.label,
                             valid_mask=self.valid_mask,
                             interaction_pair=self.interaction_edge,
                             )
        return data

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.data