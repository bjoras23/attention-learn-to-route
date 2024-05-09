import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateDCVRPTW(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    demand: torch.Tensor
    arrival_t: torch.Tensor
    deadline: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    visited: torch.Tensor  # Keeps track of nodes that have been visited
    future: torch.Tensor # Mask future nodes (don't have access to them in cur_t)
    release_t: torch.Tensor
    cur_t: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    # Results
    costs: list
    pens: list
    routes: list

    VEHICLE_CAPACITY = 1.0  # Hardcoded
    TIMESTEP = 0.15 # TODO Shift time step harcoded for now 
    FIRST_SHIFT = 0.1 # TODO Shift time step harcoded for now 
    LAST_SHIFT = 1.0 - TIMESTEP
    
    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            visited=self.visited[key],
            future_=self.future[key],
            cur_t=self.cur_t[key]
        )

    @staticmethod
    def initialize(input):

        depot = input['depot']
        loc = input['loc']
        demand = input['demand']
        arrival_t = input['arrival_t']
        deadline = input['deadline']
        # Add deadline for depot == 1, meaning its the day length
        deadline = torch.cat((torch.ones(deadline.size()[:-1])[:, None], deadline), -1)
        batch_size, n_loc, _ = loc.size()
        cur_t = torch.zeros(batch_size, 1, dtype=torch.float)+StateDCVRPTW.FIRST_SHIFT+StateDCVRPTW.TIMESTEP
        return StateDCVRPTW(
            coords=torch.cat((depot[:, None, :], loc), -2),
            demand=demand,
            arrival_t=arrival_t,
            deadline=deadline,
            ids=torch.arange(batch_size, dtype=torch.int64)[:, None],  # Add steps dimension
            visited=torch.zeros((batch_size, n_loc+1), dtype=torch.bool),
            future=torch.cat((torch.zeros((batch_size, 1), dtype=torch.bool), arrival_t > cur_t), dim=-1),
            release_t=torch.zeros((batch_size, n_loc+1), dtype=torch.float),
            cur_t=cur_t,
            costs=[],
            pens=[],
            routes=[],
            i=torch.zeros(1, dtype=torch.int64)  # Vector with length num_steps
        )

    def _get_future_mask(self):
        batch_size, _, _ = self.loc.size()
        return torch.cat((torch.zeros((batch_size, 1), dtype=torch.bool), self.arrival_t > self.cur_t), dim=-1)

    def get_final_costs(self):
        return torch.stack(self.costs, dim=-1).sum(-1)
    
    def get_final_pens(self):
        # TODO Right now pens are calculated in route
        return torch.stack(self.pens, dim=-1).sum(-1)
        
    def update(self, cost, pen, route, cur_t):
        visited = self.visited.scatter(1, route, True)
        visited[:, 0] = False # Depot
        cur_t[(cur_t == self.cur_t) & (cur_t < self.LAST_SHIFT)] += self.TIMESTEP # TODO trip may end after shift
        future = torch.cat((torch.zeros((self.future.size()[0], 1), dtype=torch.bool), self.arrival_t > cur_t), dim=-1)
        self.costs.append(cost)
        self.pens.append(pen)
        self.routes.append(route)

        return self._replace(
            visited=visited, cur_t=cur_t, future=future, costs=self.costs, 
            pens=self.pens, routes=self.routes, i=self.i + 1
        )

    def last_shift(self):
        return (self.cur_t >= self.LAST_SHIFT).all()

    def all_finished(self):
        # All visited except depot
        return self.visited[:,1:].all()
        
    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_mask(self):
        """Mask visited nodes and future, not yet known nodes"""
        return self.future | self.visited

    def get_route_mask(self, pi):
        """Mask visited, future and non selected nodes"""
        mask = self.future | self.visited | pi.to(torch.bool)
        mask[:,0] = False
        return mask
    
    def car(self):
        return torch.cat((self.coords[:,0], self.cur_t, torch.Tensor([self.VEHICLE_CAPACITY]).expand_as(self.cur_t).to(self.cur_t.device)), dim=-1)   

    def construct_solutions(self, actions):
        return actions
