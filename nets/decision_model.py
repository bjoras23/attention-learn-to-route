import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches
from nets.layers import FeedForward
from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many
from problems.vrp.state_cvrptw import StateCVRPTW


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class DecisionModelFixed(NamedTuple):
    """
    Context is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """

    node_embeddings: torch.Tensor
    graph_embed_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return DecisionModelFixed(
            node_embeddings=self.node_embeddings[key],
            graph_embed_projected=self.graph_embed_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key],
        )


class DecisionModel(nn.Module):
    # Model for dynamic cvrp with lead time constraints
    # This model decides which packages to deliver at each shift

    def __init__(
        self,
        embed_dim,
        hidden_dim,
        problem,
        route_model=None,
        n_encode_layers=2,
        tanh_clipping=10.0,
        mask_inner=True,
        mask_logits=True,
        normalization="batch",
        n_heads=8,
        checkpoint_encoder=False,
        shrink_size=None,
        pen_coef=0,
        attention="transformer",
    ):
        super(DecisionModel, self).__init__()

        # cvrptw Model used for route calculation
        # assert route_model is not None, "Route model not selected in decision model"
        self.route_model = route_model

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.pen_coef = pen_coef

        # coords + demand + deadline
        node_dim = 4
        # Embedding of last node + remaining_capacity + curr_time
        step_context_dim = embed_dim + 2

        self.init_embed = nn.Linear(node_dim, embed_dim)

        # Special embedding projection for depot node
        # coords
        self.init_embed_depot = nn.Linear(2, embed_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embed_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization,
            attention=attention,
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeds = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.use_car = nn.Linear(2 * embed_dim, 2, bias=False)

        # depot coords, capacity, current time
        vehicle_node_dim = 4
        # vehicle embedding
        self.car_ff = nn.Sequential(
            nn.Linear(vehicle_node_dim, embed_dim),
            FeedForward(embed_dim, embed_dim * 2),
        )

        self.project_step_context = nn.Linear(step_context_dim, embed_dim, bias=False)
        self.project_graph_embed = nn.Linear(embed_dim, embed_dim, bias=False)

        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_final_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=False)
        assert embed_dim % n_heads == 0

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        import time
        state = self.problem.make_state(input)
        ll = []
        ll_car = []
        routes = []
        while not state.last_shift():
            embeddings, _ = self.embedder(self._init_embed(input), mask=state.get_mask())
            log_p_use_car, log_p, route, cost, pen, cur_t = self._inner(input, embeddings, state)
            ll.append(log_p)
            routes.append(route)
            ll_car.append(log_p_use_car)
            state = state.update(cost, pen, route, cur_t)

        # Can't delay in last shift
        embeddings, _ = self.embedder(self._init_embed(input), mask=state.get_mask())
        log_p, route, cost, pen, cur_t = self._inner(input, embeddings, state, last_shift=True)
        ll.append(log_p)
        routes.append(route)
        state = state.update(cost, pen, route, cur_t)

        assert state.all_finished(), "No all deliveries were done"

        costs = state.get_final_costs()
        pens = state.get_final_pens()
        rewards = costs + (pens*self.pen_coef)
        ll = torch.concat(ll, dim=-1).sum(-1)
        ll_car = torch.concat(ll_car, dim=-1).sum(-1)
        ll = ll + ll_car
        if return_pi:
            return costs, pens, rewards, ll, pi

        return costs, pens, rewards, ll

    def _init_embed(self, input):
        # Maybe for this model depot is useless?
        # Maybe not for graph context
        features = (
            "demand",
            "deadline",
        )

        # TODO add shift times to depot
        return torch.cat(
            (
                self.init_embed_depot(input["depot"])[:, None, :],
                self.init_embed(
                    torch.cat(
                        (input["loc"], *(input[feat][:, :, None] for feat in features)),
                        -1,
                    )
                ),
            ),
            1,
        )

    def _inner(self, input, embeddings, state, last_shift=False):
        mask = state.get_mask()
        # Skip car choice

        # mean the masked embeddings
        graph_embed = torch.sum(~mask.unsqueeze(-1) * embeddings, dim=-2) / torch.sum(
            ~mask, dim=-1, keepdim=True
        )
        if not last_shift:
            # car embed (batch_size, embed_dim)
            car_embed = self.car_ff(state.car())   # projected graph embeddings
            # graph_embed = self.project_graph_embed(graph_embed)
            # Use_car == 1 if vehicle is to be used, else it delays
            log_p_use_car = torch.log_softmax(
                self.use_car(torch.cat((graph_embed, car_embed), dim=-1)), dim=-1
            )
            use_car = self._select_node(log_p_use_car.exp())
            log_p_use_car = log_p_use_car.gather(1, use_car[:, None])

            mask = mask | ~use_car.to(torch.bool).unsqueeze(-1)

        # Route for the instances that will use car
        route_state = StateCVRPTW.initialize(
            input, mask=mask, cur_t=state.cur_t
        )
        fixed = self._precompute(embeddings, graph_embed)
        log_ps = []
        routes = []
        one_route = not last_shift

        # The projection of the node embeddings for the attention is calculated once up front
        selected = torch.Tensor([23]) 
        while (last_shift and not route_state.all_finished()) or (not torch.all(selected==0)):
            log_p = self._get_log_p(fixed, route_state, route_state.get_mask())
            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected = self._select_node(log_p.exp())
            route_state = route_state.update(selected, one_route=one_route)

            log_ps.append(log_p)
            routes.append(selected)

        cost = route_state.get_final_costs()
        pen = route_state.get_final_pens()
        # Collected lists, return Tensor
        routes = torch.stack(routes, 1)
        log_ps = torch.stack(log_ps, 1).gather(2, routes.unsqueeze(-1)).squeeze(-1)

        if last_shift:
            return log_ps, routes, cost, pen, route_state.cur_t
        return log_p_use_car, log_ps, routes, cost, pen, route_state.cur_t

    def _precompute(self, embeddings, graph_embed):
        # fixed context = (batch_size, embed_dim)
        fixed_context = self.project_graph_embed(graph_embed)

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = (
            self.project_node_embeds(embeddings).chunk(3, dim=-1)
        )

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed),
            self._make_heads(glimpse_val_fixed),
            logit_key_fixed.contiguous(),
        )
        return DecisionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p(self, fixed, state, mask):

        # Compute query = context node embedding
        query = fixed.graph_embed_projected + self.project_step_context(
            self._get_parallel_step_context(fixed.node_embeddings, state)
        )

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = (
            fixed.glimpse_key,
            fixed.glimpse_val,
            fixed.logit_key,
        )

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(
            query, glimpse_K, glimpse_V, logit_K, mask
        )

        log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p

    def _get_parallel_step_context(self, embeddings, state):
        """
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param state
        :return: (batch_size, num_steps, context_dim)
        """
        current_node = state.get_current_node()

        # Embedding of previous node + remaining capacity
        return torch.cat(
            (
                embeddings[:,current_node[-1]].squeeze(1),
                self.problem.VEHICLE_CAPACITY - state.used_capacity,
                state.cur_t,
            ),
            -1,
        )

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, 1, key_size)
        glimpse_Q = query.view(batch_size, self.n_heads, 1, key_size).permute(1, 0, 2, 3)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, graph_size)
        compatibility = torch.matmul(
            glimpse_Q, glimpse_K.transpose(-2, -1)
        ) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None,:,:].expand_as(compatibility)] = - math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, embedding_dim)
        glimpse = self.project_final_Q(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, 1, self.n_heads * val_size)
        )

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask.squeeze(1)] = -math.inf

        return logits, glimpse.squeeze(-2)

    # def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

    #     batch_size, embed_dim = query.size()
    #     key_size = val_size = embed_dim // self.n_heads

    #     # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, 1, key_size)
    #     glimpse_Q = query.view(batch_size, self.n_heads, 1, key_size).permute(1, 0, 2, 3)
    #     glimpse_K = glimpse_K.view(batch_size, -1, self.n_heads, key_size).permute(2,0,3,1)
    #     glimpse_V = glimpse_V.view(batch_size, -1, self.n_heads, key_size).permute(2,0,1,3)
    #     # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, graph_size)
    #     compatibility = (torch.matmul(glimpse_Q, glimpse_K) / math.sqrt(glimpse_Q.size(-1)))

    #     if mask is not None:
    #         mask_compatibility = mask.view(1, batch_size, 1, -1).expand_as(compatibility)
    #         compatibility = compatibility.masked_fill(mask_compatibility, -torch.inf)

    #     # Batch matrix multiplication to compute heads (n_heads, batch_size, val_size)
    #     final_Q = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V).permute(1, 2, 0, 3).contiguous().view(-1, 1, self.n_heads * val_size)

    #     # Project to get glimpse/updated context node embedding (batch_size, embedding_dim)
    #     final_Q = self.project_final_Q(final_Q)

    #     # like concatenation project W[W2final_Q || W1 embeds]
    #     logits = self.project_out(final_Q + logit_K)

    #     # From the logits compute the probabilities by clipping, masking and softmax
    #     if self.tanh_clipping > 0:
    #         logits = torch.tanh(logits) * self.tanh_clipping
    #     if mask is not None:
    #         logits[:,:,0][mask] = -torch.inf
    #     log_p = torch.log_softmax(logits / self.temp, dim=-1)
    #     return log_p

    def _select_node(self, probs):
        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(-1)
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"

        return selected

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(
                input[0], pi
            ),  # Don't need embeddings as input to get_costs
            (
                input,
                self.embedder(self._init_embed(input))[0],
            ),  # Pack input with embeddings (additional input)
            batch_rep,
            iter_rep,
        )

    def _make_heads(self, v):

        return (
            v.contiguous()
            .view(v.size(0), v.size(1), self.n_heads, -1)
            .permute(2, 0, 1, 3)
        )  # (n_heads, batch_size, graph_size, head_dim)
