import os
import torch
import numpy as  np
import os.path as osp
from torch.optim import Adam
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch_geometric
import networkx as nx
from .utils import corruption
from .model import MyDGI, DSTCWeightEncoder

def train_model(args, data, adj, coords, attn_LRs, spatial_regularization_strength, device):
    model = MyDGI(
        hidden_channels=10,
        encoder=DSTCWeightEncoder(n_input=args.n_input, n_clusters=args.n_clusters, v=1.0, generated_data_fold=osp.join(args.data_path, args.data_name)),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption
    ).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    G = nx.from_numpy_array(adj.cpu().numpy()).to_undirected()
    edge_index = (torch_geometric.utils.convert.from_networkx(G).to(device)).edge_index
    # prepare training data
    data_obj = Data(edge_index=edge_index, x=data) 
    data_obj.num_nodes = data.shape[0] 
    data_obj.train_mask = data_obj.val_mask = data_obj.test_mask = data_obj.y = None

    transform = T.RandomLinkSplit(num_val=0.0, num_test=0.0, is_undirected=True, 
                                    add_negative_train_samples=False, split_labels=True)
    train_data, _ , _ = transform(data_obj)
    coords = torch.tensor(coords).float().to(device)
    model.train()
    min_loss = np.inf
    # edge_subset_sz = int(1e6)
    best_params = model.state_dict()
    from time import time
    st = time()
    for epoch in range(args.num_epoch):
        epoch_loss = 0.0
        z = model.encoder(data, attn_LRs) # data is gene expression

        penalty_3 = model.encoder.sp_loss(z, train_data.pos_edge_label_index)
        l_sp = spatial_regularization_strength * penalty_3

        # loss = graph_loss

        optimizer.zero_grad()
        # graph_loss.backward()
        l_sp.backward()
        optimizer.step()
        # epoch_loss += graph_loss.item()
        epoch_loss += l_sp.item()

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            best_params = model.state_dict()

        if (epoch + 1) % 10 == 0:
            print('====> Epoch: {}, Loss: {:.5f}'.format(epoch + 1, epoch_loss))
            print(f'time: {time() - st: .3f}, l_sp: {l_sp.item(): .4f}')
            st = time()

    print(f'Min Loss: {min_loss}')

    model.eval()
    model.load_state_dict(best_params)
    h = model.encoder(data, attn_LRs, print_lambda=True)
    embedding_save_filepath = args.embedding_data_path + '/spot_embed.npy'    
    save_dir = os.path.dirname(embedding_save_filepath)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(embedding_save_filepath, h.cpu().detach().numpy())