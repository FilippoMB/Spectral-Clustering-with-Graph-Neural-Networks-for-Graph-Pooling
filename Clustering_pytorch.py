import os.path as osp
import torch
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GraphConv, dense_mincut_pool
from torch_geometric import utils
from torch_geometric.nn import Sequential
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from sklearn.metrics import normalized_mutual_info_score as NMI

torch.manual_seed(0) # for reproducibility

# Load data
dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

# Normalized adjacency matrix
data.edge_index, data.edge_weight = gcn_norm(  
                data.edge_index, data.edge_weight, data.num_nodes,
                add_self_loops=False, dtype=data.x.dtype)

class Net(torch.nn.Module):
    def __init__(self, 
                 mp_units,
                 mp_act,
                 in_channels, 
                 n_clusters, 
                 mlp_units=[],
                 mlp_act="Identity"):
        super().__init__()
        
        mp_act = getattr(torch.nn, mp_act)(inplace=True)
        mlp_act = getattr(torch.nn, mlp_act)(inplace=True)
        
        # Message passing layers
        mp = [
            (GraphConv(in_channels, mp_units[0]), 'x, edge_index, edge_weight -> x'),
            mp_act
        ]
        for i in range(len(mp_units)-1):
            mp.append((GraphConv(mp_units[i], mp_units[i+1]), 'x, edge_index, edge_weight -> x'))
            mp.append(mp_act)
        self.mp = Sequential('x, edge_index, edge_weight', mp)
        out_chan = mp_units[-1]
        
        # MLP layers
        self.mlp = torch.nn.Sequential()
        for units in mlp_units:
            self.mlp.append(Linear(out_chan, units))
            out_chan = units
            self.mlp.append(mlp_act)
        self.mlp.append(Linear(out_chan, n_clusters))
        

    def forward(self, x, edge_index, edge_weight):
        
        # Propagate node feats
        x = self.mp(x, edge_index, edge_weight) 
        
        # Cluster assignments (logits)
        s = self.mlp(x) 
        
        # Obtain MinCutPool losses
        adj = utils.to_dense_adj(edge_index, edge_attr=edge_weight)
        _, _, mc_loss, o_loss = dense_mincut_pool(x, adj, s)
        
        return torch.softmax(s, dim=-1), mc_loss, o_loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = Net([16], "ELU", dataset.num_features, dataset.num_classes).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


def train():
    model.train()
    optimizer.zero_grad()
    _, mc_loss, o_loss = model(data.x, data.edge_index, data.edge_weight)
    loss = mc_loss + o_loss
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test():
    model.eval()
    clust, _, _ = model(data.x, data.edge_index, data.edge_weight)
    return NMI(clust.max(1)[1].cpu(), data.y.cpu())
    

patience = 50
best_nmi = 0
for epoch in range(1, 10000):
    train_loss = train()
    nmi = test()
    print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, NMI: {nmi:.3f}')
    if nmi > best_nmi:
        best_nmi = nmi
        patience = 50
    else:
        patience -= 1     
    if patience == 0:
        break
