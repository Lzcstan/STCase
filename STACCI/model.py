import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import InnerProductDecoder, DeepGraphInfomax
from torch_geometric.utils import (negative_sampling, remove_self_loops, add_self_loops)


EPS = 1e-15


class LRWeightGraphAttentionLayer(nn.Module):
    """
    LR GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, hidden_features, LRs, dropout=0.0, alpha=0.2, concat=False):
        super(LRWeightGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.alpha = alpha
        self.concat = concat

        self.LRs = LRs
        self.LR2ids = {key: idx for idx, key in enumerate(LRs)}

        # [setattr(self, f'lambda_{k}', nn.Parameter(torch.ones(size=(1,)))) for k, v in LRs.items()]
        self.lambda_LRs = nn.Parameter(torch.ones(size=(len(LRs), 1)))

        self.lr_proj = nn.Linear(in_features * 2, out_features, bias=False)
        self.h_prime_norm = nn.BatchNorm1d(out_features) # XXX

    def forward(self, h, attn_LRs):
        h_lr = torch.zeros(h.shape[0], self.hidden_features).to(h.device)
        lambda_LRs = F.softmax(self.lambda_LRs, dim=0)
        for interaction in self.LRs:
            attn_k_LR = attn_LRs[interaction]
            # lambda_k = getattr(self, f'lambda_{interaction}')
            idx = self.LR2ids[interaction]

            tmp = torch.matmul(attn_k_LR.T, h)
            h_lr += tmp * lambda_LRs[idx]

        h_lr = torch.cat((h, h_lr), dim=1) # Cat_EFT
        h_prime = self.lr_proj(h_lr)
        h_prime = self.h_prime_norm(h_prime)
        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
class MyDGI(DeepGraphInfomax):
    r"""The Deep Graph Infomax model from the
    `"Deep Graph Infomax" <https://arxiv.org/abs/1809.10341>`_
    paper based on user-defined encoder and summary model :math:`\mathcal{E}`
    and :math:`\mathcal{R}` respectively, and a corruption function
    :math:`\mathcal{C}`.

    Args:
        hidden_channels (int): The latent space dimensionality.
        encoder (torch.nn.Module): The encoder module :math:`\mathcal{E}`.
        summary (callable): The readout function :math:`\mathcal{R}`.
        corruption (callable): The corruption function :math:`\mathcal{C}`.
    """
    def forward(self, *args, **kwargs):
        """Returns the latent space for the input arguments, their
        corruptions and their summary representation."""
        pos_z = self.encoder(*args, **kwargs)
        cor = self.corruption(*args, **kwargs)
        cor = cor if isinstance(cor, tuple) else (cor, )
        neg_z = self.encoder(*cor)
        summary = self.summary(pos_z, *args, **kwargs)
        return pos_z, neg_z, summary
    
class DSTCWeightEncoder(nn.Module):
    def __init__(self, n_input, n_clusters, v=1, generated_data_fold='generated/BRCA_Visium_10x'):
        super(DSTCWeightEncoder, self).__init__()

        with open(generated_data_fold + f'/Interaction2symbol', 'rb') as fp:
            self.LR_dict = pickle.load(fp)
        # GCN for inter information
        # self.gnn_1 = LRWeightGraphAttentionLayer(n_input, n_input, self.LR_dict, feat_syms=self.highly_symbols, device=device)
        # self.gnn_2 = LRWeightGraphAttentionLayer(n_input, n_input, self.LR_dict, feat_syms=self.highly_symbols, device=device)
        self.gnn_z = LRWeightGraphAttentionLayer(n_input, 10, n_input, self.LR_dict)
        # self.gnn_o = GraphAttentionLayer(10, n_clusters)

        self.decoder = InnerProductDecoder()
    
    def forward(self, x, attn, print_lambda=False):
        # DNN Module
        tra = F.relu(x)
        
        sigma = 0.5

        # GAT Module
        # h1 = self.gnn_1(x, adj)
        # latent = self.gnn_2((1-sigma)*h1 + sigma*tra, adj)
        # h = self.gnn_z((1-sigma)*latent + sigma*x, adj)
        # h = self.gnn_z((1-sigma)*h1 + sigma*x, adj)
        h = self.gnn_z(x, attn)
        # h_prime = self.gnn_o(h, adj)

        # calculate the logits
        # predict = F.softmax(h_prime, dim=1)

        if print_lambda:
            print(
                dict(sorted(
                    dict(
                        zip(self.LR_dict.keys(), F.softmax(self.gnn_z.lambda_LRs, dim=0)[:, 0].detach().cpu().numpy())
                    ).items(), 
                    key=lambda x: x[1], 
                    reverse=True
                ))
            )

        return h
    
    def graph_loss(self, z, pos_edge_index, neg_edge_index=None):

        self.decoded = self.decoder(z, pos_edge_index, sigmoid=True)
        pos_loss = -torch.log(self.decoded + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss
    
    def sp_loss(self, z, pos_edge_index, neg_edge_index=None):

        self.decoded = self.decoder(z, pos_edge_index, sigmoid=True)
        pos_loss = -torch.log(self.decoded + EPS).mean()

        return pos_loss