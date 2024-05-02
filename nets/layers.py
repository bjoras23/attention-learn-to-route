import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import math

class GATMHAEfficient(nn.Module):
    # Graph Attention Network message passing layer
    # https://arxiv.org/abs/1710.10903
    # https://nn.labml.ai/graphs/gat/index.html

    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            leakyReLu_slope=0.2
    ):
        super(GATMHAEfficient, self).__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        assert embed_dim%n_heads==0, f"{embed_dim=}, {n_heads=}"

        # k heads linear transformation
        self.W = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.head_dim))
        
        self.Wal = nn.Parameter(torch.Tensor(self.n_heads, self.head_dim, 1))
        self.War = nn.Parameter(torch.Tensor(self.n_heads, self.head_dim, 1))

        self.activation = nn.LeakyReLU(negative_slope=leakyReLu_slope)
        self.nonlinear = nn.ReLU()
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


    def forward(self, h, mask=None):
        """
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        h' = ∣∣​hᵢ′ᵏ = ∣∣σ(∑αᵢⱼᵏ​gⱼᵏ​)
        αᵢⱼᵏ = SoftMaxⱼ(LeakyReLU((aᵏ)ᵀ[gᵢᵏ|| gⱼᵏ])​)
        gᵏ = hWᵏ
        """
        batch_size, graph_size, input_dim = h.size()
        #gᵏ = hWᵏ
        g = (h.contiguous().view(-1, input_dim) @ self.W)
        

        # (aᵏ)ᵀ[gᵢᵏ|| gⱼᵏ]) (This is just a more efficient way to pass through the layer and concatenate)
        a_i = (g @ self.Wal).view(self.n_heads, batch_size, graph_size, 1)  
        a_j = (g @ self.War).view(self.n_heads, batch_size, 1, graph_size)
        # e_ijk = LeakyReLU((aᵏ)ᵀ[gᵢᵏ|| gⱼᵏ])
        e_ijk = self.activation(a_i + a_j)


        # Optionally apply mask to prevent attention TODO check mask
        if mask is not None:
            mask = mask.view(1, batch_size, graph_size, graph_size).expand_as(e_ijk)
            e_ijk[mask] = -np.inf

        # αᵢⱼᵏ = SoftMaxⱼ(eᵢⱼᵏ​)
        # (n_heads, batch_size, graph_size, graph_size)
        attn = F.softmax(e_ijk, dim=-1)

        # hᵢ′ᵏ​ = σ(∑αᵢⱼᵏ​gⱼᵏ​); σ = nonlinear function (e.g. ReLu)
        # (batch_size, graph_size, n_heads, head_dim)
        h_prime = self.nonlinear(torch.einsum('hbij,hbjf->bihf', attn, g.view(self.n_heads, batch_size, graph_size, self.head_dim)))

        # TODO, average is often used for output layer

        # hᵢ′​ = ∣∣​hᵢ′ᵏ
        # Concatenation of heads (batch_size, graph_size, embed_dim)
        return h_prime.reshape(batch_size, graph_size, self.n_heads*self.head_dim)




class GATv2MHA(nn.Module):
    # GATv2 message passing layer
    # https://arxiv.org/abs/2105.14491
    # https://nn.labml.ai/graphs/gatv2/index.html

    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            share_weights=False,
            share_linear=False,
            leakyReLu_slope=0.2
    ):
        super(GATv2MHA, self).__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.shared_weights = share_weights
        self.shared_linear = share_linear

        assert embed_dim%n_heads==0, f"{embed_dim=}, {n_heads=}"
    
        self.Wl = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.head_dim))
        
        if share_weights:
            self.Wr = self.Wl
        else:
            self.Wr = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.head_dim))

        self.Wak = nn.Parameter(torch.Tensor(self.n_heads, self.head_dim, 1))

        self.activation = nn.LeakyReLU(negative_slope=leakyReLu_slope)
        self.nonlinear = nn.ReLU()
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


    def forward(self, h, mask=None):
        """
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        h' = ∣∣​hᵢ′ᵏ = ∣∣σ(∑αᵢⱼᵏ​gⱼᵏ​)
        αᵢⱼᵏ = SoftMaxⱼ((aᵏ)ᵀLeakyReLU(W[hᵢᵏ|| hⱼᵏ])​)
        gᵏ = hWᵏ
        """

        batch_size, graph_size, input_dim = h.size()

        h = h.contiguous().view(-1, input_dim)
        # Again, more efficient way to pass through layer and concatenate 
        g_l = (h @ self.Wl).view(self.n_heads, batch_size, graph_size, self.head_dim)
        g_r = (h @ self.Wr).view(self.n_heads, batch_size, graph_size, self.head_dim)

        # LeakyReLU(W[hᵢᵏ|| hⱼᵏ])
        e_ijk = self.activation(g_l.view(self.n_heads, batch_size, graph_size, 1, self.head_dim) + g_r.view(self.n_heads, batch_size, 1, graph_size, self.head_dim))
        # (aᵏ)ᵀLeakyReLU(W[hᵢᵏ|| hⱼᵏ])
        # (n_heads, batch_size, graph_size, graph_size)
        e_ijk = (e_ijk.contiguous().view(self.n_heads, -1, self.head_dim) @ self.Wak).view(self.n_heads, batch_size, graph_size, graph_size)

        # Optionally apply mask to prevent attention TODO check mask
        if mask is not None:
            mask = mask.view(1, batch_size, graph_size, graph_size).expand_as(e_ijk)
            e_ijk[mask] = -np.inf

        # αᵢⱼᵏ = SoftMaxⱼ(eᵢⱼᵏ​)
        # (n_heads, batch_size, graph_size, graph_size)
        attn = F.softmax(e_ijk, dim=-1)

        # hᵢ′ᵏ​ = σ(∑αᵢⱼᵏ​grⱼᵏ​); σ = nonlinear function (e.g. ReLu)
        # (batch_size, graph_size, n_heads, head_dim)
        h_prime = self.nonlinear(torch.einsum('hbij,hbjf->bihf', attn, g_r))

        # TODO, average is often used for output layer

        # hᵢ′​ = ∣∣​hᵢ′ᵏ
        # Concatenation of heads (batch_size, graph_size, embed_dim)
        return h_prime.reshape(batch_size, graph_size, self.n_heads*self.head_dim)
    