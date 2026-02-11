import torch


def compute_max_link_utilization(
    path_weights,
    true_demands,
    capacities,
    paths_to_edges,
    commodities_to_paths,
    normalize=False
):
    """
    Compute the maximum link utilization for multi-commodity flow problem.
    
    This function calculates how congested the most utilized link is in the network
    when routing demands according to the given path weights.
    
    Parameters:
    - path_weights: Path selection weights tensor [num_commodities, num_paths]
    - true_demands: Actual demand values tensor [num_commodities, 1]
    - capacities: Edge capacity values tensor [num_edges]
    - paths_to_edges: Binary matrix indicating which edges belong to which paths [num_paths, num_edges]
    - commodities_to_paths: Binary matrix indicating which paths are available for which commodities [num_commodities, num_paths]
    - normalize: Whether to normalize the loss values
    
    Returns:
    - List of maximum link utilization values for each commodity
    """
    losses = []
    capacities = capacities.unsqueeze(-1)
    for i in range(true_demands.shape[0]):
        wi = torch.transpose(path_weights[[i]], 0, 1)
        di = true_demands[[i]]

        # \sum weights of one pair = 1.0
        commodity_total_weight = commodities_to_paths.matmul(wi)
        paths_over_total = commodities_to_paths.transpose(0, 1).matmul(1.0 / torch.clamp_min(commodity_total_weight, 1e-7))
        wi = wi.mul(paths_over_total)

        tmp_demand_on_paths = commodities_to_paths.transpose(0, 1).matmul(di.transpose(0, 1)) 
        demand_on_paths = tmp_demand_on_paths.mul(wi) 
        flow_on_edges = paths_to_edges.transpose(0, 1).matmul(demand_on_paths) 
        congestion = flow_on_edges.divide(capacities) 

        max_utils = torch.max(congestion.flatten(), dim = 0).values 

        if normalize:
            loss = 1.0 - max_utils if max_utils.item() == 0.0 else max_utils / max_utils.item()
            losses.append(loss) 
        else:
            losses.append(max_utils)

    return losses


def compute_total_flow(
    path_weights,
    true_demands,
    capacities,
    paths_to_edges,
    commodities_to_paths,
    normalize=False
):
    """
    Compute the total achievable flow considering capacity constraints.
    
    This function calculates how much flow can be routed through the network
    without exceeding edge capacities, adjusting path weights accordingly.
    
    Parameters:
    - path_weights: Path selection weights tensor [num_commodities, num_paths]
    - true_demands: Actual demand values tensor [num_commodities, 1]
    - capacities: Edge capacity values tensor [num_edges]
    - paths_to_edges: Binary matrix indicating which edges belong to which paths [num_paths, num_edges]
    - commodities_to_paths: Binary matrix indicating which paths are available for which commodities [num_commodities, num_paths]
    - normalize: Whether to normalize the loss values
    
    Returns:
    - List of negative flow ratios (to maximize flow) for each commodity
    """
    losses = []
    capacities = capacities.unsqueeze(-1)
    path_weights = torch.clamp_min(path_weights, min=1e-8)
    for i in range(true_demands.shape[0]): 
        wi = path_weights[[i]]
        di = true_demands[[i]]

        # \sum weights of one pair = 1.0
        commodity_total_weight = commodities_to_paths.matmul(wi.transpose(0,1))
        paths_over_total = commodities_to_paths.transpose(0, 1).matmul(1.0 / torch.clamp_min(commodity_total_weight, 1e-7))
        wi = wi.transpose(0,1).mul(paths_over_total).transpose(0,1)

        wi = (wi.reshape(1, -1, di.shape[-1]) * di.unsqueeze(1)).reshape(wi.shape).transpose(0, 1)

        total_flow = torch.sum(di)

        flow_on_edges = paths_to_edges.transpose(0, 1).matmul(wi)
        excess_flow = torch.relu(flow_on_edges - capacities)
        Total_excess_flow = excess_flow.sum() 
        max_flow = total_flow - Total_excess_flow

        while Total_excess_flow != 0:
            wi = wi * max_flow / total_flow
            flow_on_edges = paths_to_edges.transpose(0, 1).matmul(wi)
            excess_flow = torch.relu(flow_on_edges - capacities)
            Total_excess_flow = excess_flow.sum() 
            max_flow = max_flow - Total_excess_flow
        
        max_flow_ratio = max_flow / total_flow

        if normalize:
            loss = - max_flow_ratio if max_flow_ratio.item() == 0.0 else - max_flow_ratio / max_flow_ratio.item()
            losses.append(loss) 
        else:
            losses.append(- max_flow_ratio)

    return losses


def compute_concurent_flow(
    path_weights,
    true_demands,
    capacities,
    paths_to_edges,
    commodities_to_paths,
    normalize=False,
    mean=False
):
    """
    Compute concurrent flow satisfaction for multi-commodity flow problem.
    
    This function measures how much of each demand can be satisfied simultaneously
    while respecting network capacity constraints.
    
    Parameters:
    - path_weights: Path selection weights tensor [num_commodities, num_paths]
    - true_demands: Actual demand values tensor [num_commodities, 1]
    - capacities: Edge capacity values tensor [num_edges]
    - paths_to_edges: Binary matrix indicating which edges belong to which paths [num_paths, num_edges]
    - commodities_to_paths: Binary matrix indicating which paths are available for which commodities [num_commodities, num_paths]
    - normalize: Whether to normalize the loss values
    - mean: Whether to return average concurrent flow or minimum concurrent flow
    
    Returns:
    - List of negative concurrent flow values (to maximize concurrent flow) for each commodity
    """
    losses = []
    capacities = capacities.unsqueeze(-1)
    path_weights = torch.clamp_min(path_weights, min=1e-8)
    for i in range(true_demands.shape[0]): 
        wi = path_weights[[i]]
        di = true_demands[[i]]

        # \sum weights of one pair = 1.0
        commodity_total_weight = commodities_to_paths.matmul(wi.transpose(0,1))
        paths_over_total = commodities_to_paths.transpose(0, 1).matmul(1.0 / torch.clamp_min(commodity_total_weight, 1e-7))
        wi = wi.transpose(0,1).mul(paths_over_total).transpose(0,1)

        wi = (wi.reshape(1, -1, di.shape[-1]) * di.unsqueeze(1)).reshape(wi.shape)
        wi = wi.transpose(0, 1)

        flow_on_edges = paths_to_edges.transpose(0, 1).matmul(wi)  # [num_edges, 1]
        # edge_ratios = torch.clamp(capacities / (flow_on_edges + 1e-7), max=1e6)

        total_flow = torch.sum(di)

        excess_flow = torch.relu(flow_on_edges - capacities)
        Total_excess_flow = excess_flow.sum() 
        max_flow = total_flow - Total_excess_flow

        while Total_excess_flow != 0:
            wi = wi * max_flow / total_flow
            flow_on_edges = paths_to_edges.transpose(0, 1).matmul(wi)
            excess_flow = torch.relu(flow_on_edges - capacities)
            Total_excess_flow = excess_flow.sum() 
            max_flow = max_flow - Total_excess_flow

        # max_concurrent = torch.min(edge_ratios)  # scalar
        concurrent_flow = wi.reshape(1, -1, di.shape[-1]).sum(dim=1) / torch.clamp_min(di, min=1e-7)
        concurrent_flow[di==0] = 1.0

        if mean:
            avg_concurrent = torch.mean(concurrent_flow)
            losses.append(- avg_concurrent)
        else: 
            max_concurrent = torch.min(concurrent_flow)
            if normalize:
                loss = - max_concurrent if max_concurrent.item() == 0.0 else - max_concurrent / max_concurrent.item()
                losses.append(loss) 
            else:
                losses.append(- max_concurrent)

    return losses