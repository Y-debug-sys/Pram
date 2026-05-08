import torch


def MLU(
    path_demands,
    true_demands,
    capacities,
    paths_to_edges,
    commodities_to_paths
):
    """
    Compute Maximum Link Utilization (MLU) objective values.
    
    Calculates the maximum and average link utilization for a given flow assignment.
    MLU measures how close the network comes to saturating its links by finding the
    highest ratio of flow to capacity across all links.
    
    Args:
        path_demands (torch.Tensor): Flow values assigned to each path for each commodity
        true_demands (torch.Tensor): Original demand values for each commodity
        capacities (torch.Tensor): Capacity values for each edge in the network
        paths_to_edges (torch.Tensor): Binary matrix mapping paths to edges they use
        commodities_to_paths (torch.Tensor): Binary matrix mapping commodities to their paths
        
    Returns:
        list: Two lists containing max and average link utilization values for each batch sample
              Format: [max_utils_list, avg_utils_list]
    """
    losses = [[], []]

    # Reshape path demands to group by commodity
    plan_demands = path_demands.reshape(path_demands.shape[0], true_demands.shape[-1], -1)
    plan_demands = torch.sum(plan_demands, dim=-1).repeat(1, 1, plan_demands.shape[-1])
    # Calculate weights for each path relative to total demand for its commodity
    path_weights = (path_demands / torch.clamp_min(plan_demands, min=1e-7)).reshape(path_demands.shape[0], -1) 

    capacities = capacities.unsqueeze(-1)
    for i in range(true_demands.shape[0]):
        wi = torch.transpose(path_weights[[i]], 0, 1)
        di = true_demands[[i]]

        # Calculate total weight assigned to each commodity
        commodity_total_weight = commodities_to_paths.matmul(wi)
        # Normalize weights by total commodity weight
        paths_over_total = commodities_to_paths.transpose(0, 1).matmul(1.0 / torch.clamp_min(commodity_total_weight, 1e-7))
        wi = wi.mul(paths_over_total)

        # Distribute demand across paths according to weights
        tmp_demand_on_paths = commodities_to_paths.transpose(0, 1).matmul(di.transpose(0, 1)) 
        demand_on_paths = tmp_demand_on_paths.mul(wi) 
        # Calculate flow on each edge by aggregating path flows
        flow_on_edges = paths_to_edges.transpose(0, 1).matmul(demand_on_paths) 
        
        # Calculate congestion as ratio of flow to capacity
        congestion = flow_on_edges.divide(capacities) 
        max_utils = torch.max(congestion.flatten()) 
        losses[0].append(max_utils.item())
        avg_utils = torch.mean(congestion.flatten())
        losses[1].append(avg_utils.item())

    return losses


def MTF(path_demands, true_demands):
    """
    Compute Maximum Total Flow (MTF) objective values.
    
    Calculates the total flow achieved as a fraction of total demand. MTF aims to maximize
    the total amount of flow sent through the network regardless of fairness among commodities.
    
    Args:
        path_demands (torch.Tensor): Flow values assigned to each path for each commodity
        true_demands (torch.Tensor): Original demand values for each commodity
        
    Returns:
        list: Two lists containing total flow fraction values for each batch sample
              Format: [max_flow_fraction_list, avg_flow_fraction_list]
    """
    losses = [[], []]
    # Reshape and sum path demands to get total flow per commodity
    plan_demands = path_demands.reshape(path_demands.shape[0], true_demands.shape[-1], -1)
    plan_demands = torch.sum(plan_demands, dim=-1)
    for i in range(true_demands.shape[0]): 
        pi = plan_demands[[i]]  # Planned demands for commodity i
        di = true_demands[[i]]  # True demands for commodity i

        # Cap planned flow at the actual demand value
        pi[pi > di] = di[pi > di]

        total_flow = torch.sum(di)      # Total demand
        max_flow = torch.sum(pi)        # Achieved flow
        
        max_flow = max_flow / total_flow  # Fraction of demand satisfied
        losses[0].append(max_flow.item())
        losses[1].append(max_flow.item())

    return losses


def MCF(path_demands, true_demands):
    """
    Compute Maximum Concurrent Flow (MCF) objective values.
    
    Calculates the maximum concurrent flow achieved, which measures the largest fraction
    of all demands that can be simultaneously satisfied. This objective promotes fairness
    among commodities by maximizing the minimum fraction of demand satisfied across all commodities.
    
    Args:
        path_demands (torch.Tensor): Flow values assigned to each path for each commodity
        true_demands (torch.Tensor): Original demand values for each commodity
        
    Returns:
        list: Two lists containing min and average concurrent flow fractions for each batch sample
              Format: [min_concurrent_flow_list, avg_concurrent_flow_list]
    """
    losses = [[], []]
    for i in range(true_demands.shape[0]): 
        pi = path_demands[[i]]  # Path demands for commodity i
        di = true_demands[[i]]  # True demands for commodity i

        # Reshape to calculate total flow per commodity from all paths
        pi = pi.reshape(1, di.shape[-1], -1)
        si = torch.sum(pi, dim=-1)
        # Scale path flows to match true demands
        scale_factor = di / torch.clamp_min(si, min=1e-7)
        scale_factor = torch.clamp_max(scale_factor, max=1.0)
        pi = pi * scale_factor.unsqueeze(-1).repeat(1, 1, pi.shape[-1])
        pi = torch.sum(pi, dim=-1)

        # Calculate concurrent flow as fraction of demand satisfied per commodity
        concurrent_flow = pi / torch.clamp_min(di, min=1e-7)
        # Set concurrent flow to 1.0 for commodities with zero demand
        concurrent_flow[di==0] = 1.

        avg_concurrent = torch.mean(concurrent_flow)
        max_concurrent = torch.min(concurrent_flow)  # Minimum fraction across all commodities
        losses[0].append(max_concurrent.item())  # Minimum concurrent flow
        losses[1].append(avg_concurrent.item())  # Average concurrent flow

    return losses
