import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import time
import math

gatv2conv = GATv2Conv
class GATMHA(nn.Module):
    # Graph Attention Network message passing layer
    # https://arxiv.org/abs/1710.10903
    # https://nn.labml.ai/graphs/gat/index.html

    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            share_linear=False,
            leakyReLu_slope=0.2
    ):
        super(GATMHA, self).__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.shared_linear = share_linear

        assert embed_dim%n_heads==0, f"{embed_dim=}, {n_heads=}"

        if self.shared_linear:
            # linear transformation then split into heads
            self.W = nn.Linear(input_dim, embed_dim, bias=False)
        else:
            # k heads linear transformation
            self.W = nn.ModuleList([nn.Linear(input_dim, self.head_dim, bias=False) for _ in range(n_heads)])
        
        self.Wak = nn.ModuleList(
            [nn.Linear(self.head_dim * 2, 1, bias=False) for _ in range(n_heads)]
        )
        # TODO slope arg
        self.activation = nn.LeakyReLU(negative_slope=leakyReLu_slope)
        self.nonlinear = nn.ReLU()

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
        batch_size, graph_size, _ = h.size()

        if self.shared_linear:
            # gᵢ = hᵢW
            # (batch_size, graph_size, embed_dim)
            g = self.W(h)
            # gᵢᵏ = split(hᵢW)
            #(n_heads, batch_size, graph_size, head_dim)
            g = g.view(batch_size, graph_size, self.n_heads, self.head_dim).permute((2,0,1,3))
       # gᵢᵏ = hᵢWᵏ
       #(n_heads, batch_size, graph_size, head_dim)
        else:
            g = torch.stack([w(h) for w in self.W], dim=0)

        # auxiliary
        sizes = [1 for _ in range(g.dim())]
        sizes[-2] = graph_size


        # g_concat = [gᵢᵏ|| gⱼᵏ]
        # (n_heads, batch_size, graph_size, graph_size, head_dim*2)
        g_concat = torch.cat(
            [g.repeat_interleave(graph_size, dim=-2), g.repeat(sizes)], dim=-1
        ).view(self.n_heads, batch_size, graph_size, graph_size, self.head_dim * 2)

        #eᵢⱼᵏ​ = LeakyReLU((aᵏ)ᵀ[gᵢᵏ|| gⱼᵏ])
        # (n_heads, batch_size, graph_size, graph_size)
        e_ijk = self.activation(torch.stack([w(g_concat[i]) for i, w in enumerate(self.Wak)], dim=0)).squeeze(-1)

        # Optionally apply mask to prevent attention TODO check mask
        if mask is not None:
            mask = mask.view(1, batch_size, graph_size, graph_size).expand_as(e_ijk)
            e_ijk[mask] = -np.inf

        # αᵢⱼᵏ = SoftMaxⱼ(eᵢⱼᵏ​)
        # (n_heads, batch_size, graph_size, graph_size)
        attn = F.softmax(e_ijk, dim=-1)

        # hᵢ′ᵏ​ = σ(∑αᵢⱼᵏ​gⱼᵏ​); σ = nonlinear function (e.g. ReLu)
        # (batch_size, graph_size, n_heads, head_dim)
        h_prime = self.nonlinear(torch.einsum('hbij,hbjf->bihf', attn, g))

        # TODO, average is often used for output layer

        # hᵢ′​ = ∣∣​hᵢ′ᵏ
        # Concatenation of heads (batch_size, graph_size, embed_dim)
        h_prime = h_prime.reshape(batch_size, graph_size, self.n_heads*self.head_dim)

        return h_prime
    

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
        # TODO slope arg
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
        g = (h.contiguous().view(-1, input_dim) @ self.W)
        
        a_i = (g @ self.Wal).view(self.n_heads, batch_size, graph_size, 1)  
        a_j = (g @ self.War).view(self.n_heads, batch_size, 1, graph_size)
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
        h_prime = h_prime.reshape(batch_size, graph_size, self.n_heads*self.head_dim)

        return h_prime




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
    
    
        # k heads linear transformation
        self.Wl = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.head_dim))
        
        if share_weights:
            # linear transformation then split into heads
            self.Wr = self.Wl
        else:
            # k heads linear transformation
            self.Wr = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.head_dim))

        
        self.Wak = nn.Parameter(torch.Tensor(self.n_heads, self.head_dim, 1))

        # TODO slope arg
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
        # if self.shared_linear:
        #     # glᵢᵏ = hᵢWl; grⱼᵏ = hⱼWr 
        #     #(n_heads, batch_size, graph_size, head_dim)
        #     g_l = self.Wl(h).view(batch_size, graph_size, self.n_heads, self.head_dim).permute((2,0,1,3))
        #     g_r = self.Wr(h).view(batch_size, graph_size, self.n_heads, self.head_dim).permute((2,0,1,3))
        # # glᵢᵏ = hᵢWlᵏ; grᵢᵏ = hᵢWrᵏ
        # #(n_heads, batch_size, graph_size, head_dim)
        # else:
        #     g_l = torch.stack([w(h) for w in self.Wl], dim=0)
        #     g_r = torch.stack([w(h) for w in self.Wl], dim=0)

        h = h.contiguous().view(-1, input_dim)
        g_l = (h @ self.Wl).view(self.n_heads, batch_size, graph_size, self.head_dim)
        g_r = (h @ self.Wr).view(self.n_heads, batch_size, graph_size, self.head_dim)

        #eᵢⱼᵏ​ = (aᵏ)ᵀLeakyReLU(W[hᵢᵏ|| hⱼᵏ])
        # (n_heads, batch_size, graph_size, graph_size)
        e_ijk = self.activation(g_l.view(self.n_heads, batch_size, graph_size, 1, self.head_dim) + g_r.view(self.n_heads, batch_size, 1, graph_size, self.head_dim))
        e_ijk = (e_ijk.contiguous().view(8, -1, self.head_dim) @ self.Wak).view(self.n_heads, batch_size, graph_size, graph_size)

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
    