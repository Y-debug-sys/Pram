import torch

from itertools import tee


# taken from `pairwise` in Itertools Recipes:
# https://docs.python.org/3/library/itertools.html
def path_to_edge_list(path):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(path)
    next(b, None)
    return zip(a, b)


def sol_dict_to_tensor(sol_dict, num_paths, commodity_list):
    """
    Convert sol_dict to a tensor of shape (num_commodities, num_paths),
    then flatten it to 1D.

    Args:
        sol_dict: dict, {(src, dst): [flow1, flow2, ...]}
        num_paths: int, number of paths per commodity
        commodity_list: list of (src, dst) tuples, defines the order

    Returns:
        tensor: shape (1, num_commodities * num_paths)
    """
    flows = []
    for commod_key in commodity_list:
        path_flows = sol_dict.get(commod_key, [])
        padded = path_flows[:num_paths] + [0.0] * (num_paths - len(path_flows))
        flows.append(padded)

    flow_tensor = torch.tensor(flows, dtype=torch.float32)  # (num_commodities, num_paths)
    return flow_tensor.flatten().unsqueeze(0)
