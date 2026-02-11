import torch

from torch.distributions.uniform import Uniform
from .objective import compute_max_link_utilization, compute_total_flow, compute_concurent_flow


def aggregate_by_segments(x: torch.Tensor, num_agents: int) -> torch.Tensor:
    """
    Aggregate a 1D tensor into `num_agents` contiguous segments by summation.

    Args:
        x: 1D tensor of shape [N]
        num_agents: number of segments

    Returns:
        out: 1D tensor of shape [num_agents],
             where out[i] = sum of x in the i-th contiguous segment.
             The last segment may be longer or shorter.
    """
    assert x.dim() == 1, "Input x must be 1D"
    N = x.size(0)
    assert num_agents > 0 and num_agents <= N

    base_len = N // num_agents
    remainder = N % num_agents

    out = []
    start = 0
    for i in range(num_agents):
        # Option A: put all remainder into the last segment
        if i == num_agents - 1:
            end = N
        else:
            end = start + base_len

        out.append(x[start:end].sum())
        start = end

    return torch.stack(out)


class PramEnv(object):
    """Environment for multi-agent reinforcement learning approach to multi-commodity flow problems.
    
    This environment is designed to solve multi-commodity flow problems using multi-agent systems,
    with three possible objectives: Maximum Link Utilization (MLU), Maximum Total Flow (MTF),
    or Maximum Concurrent Flow (MCF).
    """
    
    def __init__(
        self, 
        args,
        p2e, 
        num_nodes,
        capacities,
        accelerator,
        paths_to_edges,
        commodities_to_paths,
        num_agents=None,
        sample_min=0.1**5,
        sample_max=1.0,
    ):
        """
        Initialize the MARL environment for multi-commodity flow problems.
        
        Parameters:
        - args: Arguments object containing the objective setting
        - p2e: Paths to edges mapping information (edge indices and path indices)
        - num_nodes: Number of nodes in the network
        - capacities: Capacities of each edge in the network
        - accelerator: Device accelerator (CPU/GPU) for computations
        - paths_to_edges: Matrix mapping paths to their constituent edges
        - commodities_to_paths: Matrix mapping commodities to their possible paths
        - num_agents: Number of agents in the MARL system (optional)
        - sample_min: Minimum value for sampling actions
        - sample_max: Maximum value for sampling actions
        """
        super(PramEnv, self).__init__()
        self.accelerator = accelerator
        self.objective = args.objective.upper()
        self.sample_min, self.sample_max = sample_min, sample_max
        self.num_agents, self.num_nodes = num_agents, num_nodes
        self.p2e = p2e.to(accelerator.device)
        self.capacities = capacities.float().to(accelerator.device)
        self.paths_to_edges = paths_to_edges.float().to(accelerator.device) 
        self.commodities_to_paths = commodities_to_paths.float().to(accelerator.device) 

    def step(self, raw_action, observation, num_marl_step=0):
        """
        Execute one step in the environment based on the action taken.
        
        Parameters:
        - raw_action: Action tensor from the agent(s)
        - observation: Current state observation
        - num_marl_step: Step counter for MARL training (default 0)
        
        Returns:
        - Reward based on the selected objective function

        Borrowed from: https://github.com/harvard-cns/teal/lib/teal_env.py
        Return an approximate reward for action for each node pair.
        To make function fast and scalable on GPU, we only calculate delta.
        We assume when changing action in one node pair:
        (1) The change in edge utilization is very small;
        (2) The bottleneck edge in a path does not change due to (1).
        For evary path after change:
            path_flow/max(util, 1) =>
            (path_flow+delta_path_flow)/max(util+delta_util, 1)
            if util < 1:
                reward = - delta_path_flow
            if util > 1:
                reward = - delta_path_flow/(util+delta_util)
                    + path_flow*delta_util/(util+delta_util)/util
                    approx delta_path_flow/util - path_flow/util^2*delta_util
        """
        if self.objective == 'MLU':
            return self.MLU_reward(raw_action, observation, num_marl_step)
        elif self.objective == 'MTF':
            return self.MTF_reward(raw_action, observation, num_marl_step)
        elif self.objective == 'MCF':
            return self.MCF_reward(raw_action, observation, num_marl_step)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def polish_w_cur(self, raw_action, observation):
        """
        Adjust path weights to ensure they respect capacity constraints.
        
        This function modifies the path weights to prevent exceeding edge capacities,
        using an iterative approach to redistribute flow while maintaining feasibility.
        
        Parameters:
        - raw_action: Raw action tensor representing path weights
        - observation: Current demand observation
        
        Returns:
        - Adjusted path flow that respects capacity constraints
        """
        path_flow = self.get_flow_from_weight(raw_action, observation).squeeze()
        total_flow = torch.sum(path_flow)

        flow_on_edges = self.paths_to_edges.transpose(0, 1).matmul(path_flow)
        excess_flow = torch.relu(flow_on_edges - self.capacities)
        Total_excess_flow = excess_flow.sum() 
        max_flow = total_flow - Total_excess_flow

        while Total_excess_flow != 0:
            path_flow = path_flow * max_flow / total_flow
            flow_on_edges = self.paths_to_edges.transpose(0, 1).matmul(path_flow)
            excess_flow = torch.relu(flow_on_edges - self.capacities)
            Total_excess_flow = excess_flow.sum() 
            max_flow = max_flow - Total_excess_flow

        # path_flow = self.post_tuning(path_flow, observation, method='LP')
        return path_flow

    def get_flow_from_weight(self, weight, demand):
        """
        Calculate path flows based on path weights and demand values.
        
        Parameters:
        - weight: Path selection weights tensor
        - demand: Demand tensor for commodities
        
        Returns:
        - Flow values distributed according to the path weights and demands
        """
        commodity_total_weight = self.commodities_to_paths.matmul(weight.transpose(0, 1))
        paths_over_total = self.commodities_to_paths.transpose(0, 1).matmul(1.0 / torch.clamp_min(commodity_total_weight, 1e-7))
        weight = weight.transpose(0,1).mul(paths_over_total).transpose(0,1)
        flow = (weight.reshape(1, -1, demand.shape[-1]) * demand.unsqueeze(1)).reshape(weight.shape).transpose(0, 1)
        return flow
    
    def MLU_reward(self, action, current_demand, num_samples):
        """
        Compute reward based on Maximum Link Utilization (MLU) objective.
        
        This function calculates rewards aimed at minimizing the maximum link utilization
        across the network, which corresponds to load balancing in network flow problems.
        
        Parameters:
        - action: Agent's action (path selection weights)
        - current_demand: Current demand values for commodities
        - num_samples: Number of samples to estimate the reward gradient
        
        Returns:
        - Reward tensor calculated according to MLU objective
        """
        num_agents = self.num_nodes if self.num_agents is None else self.num_agents
        reward = torch.zeros(num_agents).to(self.accelerator.device)
        num_path_node = self.paths_to_edges.shape[0]

        wi = action.detach()
        di = current_demand
        path_flow = self.get_flow_from_weight(wi, di).squeeze()

        # edge_flow = torch_scatter.scatter(path_flow[self.p2e[0]], self.p2e[1])  # use it instead if torch_scatter is available
        num_edges = int(self.p2e[1].max()) + 1
        edge_flow = torch.zeros(num_edges, device=path_flow.device)
        edge_flow.index_add_(
            0,
            self.p2e[1],                 # edge_idx
            path_flow[self.p2e[0]]       # values
        )

        util = edge_flow/self.capacities
        distribution = Uniform(
                torch.ones(wi.shape).to(self.accelerator.device)*self.sample_min,
                torch.ones(wi.shape).to(self.accelerator.device)*self.sample_max)
        
        # find link with max utilization
        max_util_edge = util.argmax()

        # prepare paths related to max_util_edge
        max_util_paths = torch.zeros(num_path_node).to(self.accelerator.device)
        max_util_paths[self.p2e[0, self.p2e[1] == max_util_edge]] =\
            1/self.capacities[max_util_edge]
        
        for _ in range(num_samples):
            sample = distribution.rsample()

            delta_path_flow = self.get_flow_from_weight(sample, di).squeeze() - path_flow
            delta_path_flow = torch.sparse_coo_tensor(
                torch.stack(
                    [torch.arange(current_demand.shape[-1])
                        .to(self.accelerator.device).repeat_interleave(num_path_node//current_demand.shape[-1]),
                        torch.arange(num_path_node).to(self.accelerator.device)]),
                delta_path_flow,
                [current_demand.shape[-1], num_path_node])
                
            rewardi = torch.sparse.mm(delta_path_flow, max_util_paths.reshape(-1, 1)).flatten()
            
            reward += aggregate_by_segments(rewardi, num_agents)

        avg_reward = compute_max_link_utilization(action, current_demand, self.capacities, self.paths_to_edges, self.commodities_to_paths)
        return reward / num_samples + torch.cat([r.unsqueeze(0) for r in avg_reward], dim=0).unsqueeze(-1)

    def MTF_reward(self, action, current_demand, num_samples):
        """
        Compute reward based on Maximum Total Flow (MTF) objective.
        
        This function calculates rewards aimed at maximizing the total flow
        that can be routed through the network without violating capacity constraints.
        
        Parameters:
        - action: Agent's action (path selection weights)
        - current_demand: Current demand values for commodities
        - num_samples: Number of samples to estimate the reward gradient
        
        Returns:
        - Reward tensor calculated according to MTF objective
        """
        num_agents = self.num_nodes if self.num_agents is None else self.num_agents
        reward = torch.zeros(num_agents).to(self.accelerator.device)
        num_path_node = self.paths_to_edges.shape[0]

        wi = action.detach()
        di = current_demand
        path_flow = self.get_flow_from_weight(wi, di).squeeze()

        # edge_flow = torch_scatter.scatter(path_flow[self.p2e[0]], self.p2e[1])  # use it instead if torch_scatter is available
        num_edges = int(self.p2e[1].max()) + 1
        edge_flow = torch.zeros(num_edges, device=path_flow.device)
        edge_flow.index_add_(
            0,
            self.p2e[1],                 # edge_idx
            path_flow[self.p2e[0]]       # values
        )

        util = edge_flow/self.capacities
        distribution = Uniform(
                torch.ones(wi.shape).to(self.accelerator.device)*self.sample_min,
                torch.ones(wi.shape).to(self.accelerator.device)*self.sample_max)
        
        edge_util = util
        values = edge_util[self.p2e[1]]
        index = self.p2e[0]
        num_paths = int(index.max()) + 1

        # path bottleneck util
        # util, path_bottleneck = torch_scatter.scatter_max(util[self.p2e[1]], self.p2e[0])
        util = torch.full(
            (num_paths,),
            -float('inf'),
            device=values.device,
            dtype=values.dtype
        )
        util.index_reduce_(0, index, values, reduce='amax')

        # path bottleneck edge
        mask = values == util[index]
        pos = torch.arange(values.size(0), device=values.device)
        path_bottleneck = torch.full(
            (num_paths,), -1, device=values.device, dtype=torch.long
        )
        path_bottleneck.index_reduce_(
            0, index, pos.masked_fill(~mask, -1), reduce='amax'
        )
        path_bottleneck = self.p2e[1][path_bottleneck]
        
        # prepare -path_flow/util^2 for reward
        coef = path_flow/util**2
        coef[util < 1] = 0
        # coef = torch_scatter.scatter(coef, path_bottleneck).reshape(-1, 1)
        out = torch.zeros(
            int(path_bottleneck.max()) + 1,
            device=coef.device,
            dtype=coef.dtype
        )
        out.index_add_(0, path_bottleneck, coef)
        coef = out.reshape(-1, 1)

        # prepare path_util to bottleneck edge_util
        bottleneck_p2e = torch.sparse_coo_tensor(
            self.p2e, (1/self.capacities)[self.p2e[1]],
            [num_path_node, self.capacities.shape[0]])
        
        rewardi = torch.zeros(current_demand.shape[-1]).to(self.accelerator.device)
        
        # sample raw_actions and change each node pair at a time for reward
        for _ in range(num_samples):
            sample = distribution.rsample()

            # add -delta_path_flow if util < 1 else -delta_path_flow/util
            delta_path_flow = self.get_flow_from_weight(sample, di).squeeze() - path_flow
            rewardi += -(delta_path_flow/(1+(util-1).relu()))\
                .reshape(-1, num_path_node//current_demand.shape[-1]).sum(-1)
            
            # add path_flow/util^2*delta_util for each path
            delta_path_flow = torch.sparse_coo_tensor(
                torch.stack(
                    [torch.arange(current_demand.shape[-1])
                        .to(self.accelerator.device).repeat_interleave(num_path_node//current_demand.shape[-1]),
                        torch.arange(num_path_node).to(self.accelerator.device)]),
                delta_path_flow,
                [current_demand.shape[-1], num_path_node])
            # get utilization changes on edge
            # do not use torch_sparse.spspmm()
            # "an illegal memory access was encountered" in large topology
            delta_util = torch.sparse.mm(delta_path_flow, bottleneck_p2e)
            rewardi += torch.sparse.mm(delta_util, coef).flatten()

        reward += aggregate_by_segments(rewardi, num_agents)
        avg_reward = compute_total_flow(action, current_demand, self.capacities, self.paths_to_edges, self.commodities_to_paths)
        return reward / num_samples + torch.cat([r.unsqueeze(0) for r in avg_reward], dim=0).unsqueeze(-1)

    def MCF_reward(self, action, current_demand, num_samples):
        """
        Compute reward based on Maximum Concurrent Flow (MCF) objective.
        
        This function calculates rewards aimed at maximizing the fraction of each demand
        that can be satisfied simultaneously, which corresponds to fair flow allocation.
        
        Parameters:
        - action: Agent's action (path selection weights)
        - current_demand: Current demand values for commodities
        - num_samples: Number of samples to estimate the reward gradient
        
        Returns:
        - Reward tensor calculated according to MCF objective
        """
        num_agents = self.num_nodes if self.num_agents is None else self.num_agents
        reward = torch.zeros(num_agents).to(self.accelerator.device)
        num_path_node = self.paths_to_edges.shape[0]

        wi = action.detach()
        di = current_demand
        path_flow = self.get_flow_from_weight(wi, di).squeeze()

        # edge_flow = torch_scatter.scatter(path_flow[self.p2e[0]], self.p2e[1])  # use it instead if torch_scatter is available
        num_edges = int(self.p2e[1].max()) + 1
        edge_flow = torch.zeros(num_edges, device=path_flow.device)
        edge_flow.index_add_(
            0,
            self.p2e[1],                 # edge_idx
            path_flow[self.p2e[0]]       # values
        )

        util = edge_flow/self.capacities
        distribution = Uniform(
                torch.ones(wi.shape).to(self.accelerator.device)*self.sample_min,
                torch.ones(wi.shape).to(self.accelerator.device)*self.sample_max)
        
        beta = 5  # e.g., 5 ~ 20, hard-min
            
        edge_util = util
        values = edge_util[self.p2e[1]]
        index = self.p2e[0]
        num_paths = int(index.max()) + 1

        # path bottleneck util
        util = torch.full(
            (num_paths,),
            -float('inf'),
            device=values.device,
            dtype=values.dtype
        )
        util.index_reduce_(0, index, values, reduce='amax')

        # path bottleneck edge
        mask = values == util[index]
        pos = torch.arange(values.size(0), device=values.device)
        path_bottleneck = torch.full(
            (num_paths,), -1, device=values.device, dtype=torch.long
        )
        path_bottleneck.index_reduce_(
            0, index, pos.masked_fill(~mask, -1), reduce='amax'
        )
        path_bottleneck = self.p2e[1][path_bottleneck]

        # prepare -path_flow/util^2 for reward
        coef = path_flow/util**2
        coef[util < 1] = 0
        out = torch.zeros(
            int(path_bottleneck.max()) + 1,
            device=coef.device,
            dtype=coef.dtype
        )
        out.index_add_(0, path_bottleneck, coef)
        coef = out.reshape(-1, 1)

        # prepare path_util to bottleneck edge_util
        bottleneck_p2e = torch.sparse_coo_tensor(
            self.p2e, (1/self.capacities)[self.p2e[1]],
            [num_path_node, self.capacities.shape[0]])
            
        di_clip = torch.clamp_min(di, 1e-7).squeeze()
            
        # -----------------------------
        # current λ_k = delivered / demand
        # -----------------------------
        pair_flow = path_flow.reshape(-1, num_path_node//current_demand.shape[-1]).sum(-1)   # [K]
        lambda_cur = pair_flow / di_clip                       # [K]

        # -----------------------------
        # soft-min weights
        # -----------------------------
        weights = torch.softmax(-beta * lambda_cur, dim=0)          # [K]

        rewardi = torch.zeros(current_demand.shape[-1]).to(self.accelerator.device)

        # sample raw_actions and change each node pair at a time for reward
        for _ in range(num_samples):
            sample = distribution.rsample()

            # add -delta_path_flow if util < 1 else -delta_path_flow/util
            delta_path_flow = self.get_flow_from_weight(sample, di).squeeze() - path_flow

            # pair-level delta flow
            delta_pair_flow = delta_path_flow \
                .reshape(-1, num_path_node//current_demand.shape[-1]).sum(-1)                # [K]
                
            # ---- (1) soft-min MCF reward ----
            rewardi += weights * (delta_pair_flow / di_clip)

            # add path_flow/util^2*delta_util for each path
            delta_path_flow = torch.sparse_coo_tensor(
                torch.stack(
                    [torch.arange(current_demand.shape[-1])
                        .to(self.accelerator.device).repeat_interleave(num_path_node//current_demand.shape[-1]),
                        torch.arange(num_path_node).to(self.accelerator.device)]),
                delta_path_flow,
                [current_demand.shape[-1], num_path_node])
            
            delta_util = torch.sparse.mm(delta_path_flow, bottleneck_p2e)
            rewardi += torch.sparse.mm(delta_util, coef).flatten()

        reward += aggregate_by_segments(rewardi, num_agents)
        avg_reward = compute_concurent_flow(action, current_demand, self.capacities, self.paths_to_edges, self.commodities_to_paths)
        return reward / num_samples + torch.cat([r.unsqueeze(0) for r in avg_reward], dim=0).unsqueeze(-1)

    # def post_tuning(self, raw_action, observation, method=None):
    #     method = 'ADMM' if method is None else method
    #     method = method.upper()
    #     assert method in ['ADMM', 'LP', 'RNN'], 'method must be one of [ADMM, LP, RNN]'