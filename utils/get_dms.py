import numpy as np
import pandas as pd
import networkx as nx

from collections import defaultdict


def read_demands_from_csv(demands_fname):
    """
    Read the demand matrices from a csv file.
    Args:
        demands_fname (str): path to the csv file containing the demand matrices

    Returns:
        np.ndarray: demand matrices
    """
    df = pd.read_csv(demands_fname, header=None)
    return df.values


def gravity_model(num_samples, G, scale_factor):
    """
    Generate demand matrices using the gravity model.
    Args:
        num_samples (int): number of demand matrices to generate
        G (nx.DiGraph): directed graph representing the network topology
        scale_factor (float): constant factor to scale the generated demands
    Returns:
        np.ndarray: generated demand matrices of shape (num_samples, num_nodes, num_nodes)
    """
    num_nodes = len(G.nodes)
    sccs = nx.strongly_connected_components(G)

    dms = np.empty((num_samples, num_nodes, num_nodes), dtype=np.float32)

    for scc in sccs:
        in_cap_sum, out_cap_sum = defaultdict(float), defaultdict(float)
        for u in scc:
            for v in G.predecessors(u):
                in_cap_sum[u] += G[v][u]['capacity']
            for v in G.successors(u):
                out_cap_sum[u] += G[u][v]['capacity']

        in_cap_sum, out_cap_sum = dict(in_cap_sum), dict(out_cap_sum)

        in_total_cap = sum(in_cap_sum.values())
        out_total_cap = sum(out_cap_sum.values())

        for u in scc:
            norm_u = out_cap_sum[u] / out_total_cap
            for v in scc:
                if u == v:
                    continue
                frac = norm_u * in_cap_sum[v] / \
                    (in_total_cap - in_cap_sum[u])

                for idx in range(num_samples):
                    dms[idx, u, v] = max(np.random.normal(frac, frac / 4), 0.0)

    return (dms * scale_factor).reshape(num_samples, -1)


def possion_model(num_samples, G, lam, decay, const_factor):
    """
    Generate demand matrices using the Poisson model.
    Args:
        num_samples (int): number of demand matrices to generate
        G (nx.DiGraph): directed graph representing the network topology
        lam (float): base rate for the Poisson distribution
        decay (float): decay factor for the Poisson distribution based on distance
        const_factor (float): constant factor to scale the generated demands
    Returns:
        np.ndarray: generated demand matrices of shape (num_samples, num_nodes, num_nodes)
    """
    num_nodes = len(G.nodes)

    distances = np.zeros((num_nodes, num_nodes), dtype=np.int)
    dist_iter = nx.shortest_path_length(G)

    for src, dist_dict in dist_iter:
        for target, dist in dist_dict.items():
            distances[src, target] = dist

    dms = np.stack([
                np.array([[np.random.poisson(lam * (decay**dist)) for dist in row] 
                          for row in distances], dtype=np.float32)
                for _ in range(num_samples)
            ])
    
    # No traffic between node and itself
    diag_idx = np.arange(num_nodes)
    dms[:, diag_idx, diag_idx] = 0.0

    return (dms * const_factor).reshape(num_samples, -1)


def bimodal_model(num_samples, G, fraction, low_range, high_range):
    """
    Generate demand matrices using a bimodal distribution.
    Args:
        num_samples (int): number of demand matrices to generate
        G (nx.DiGraph): directed graph representing the network topology
        fraction (float): fraction of demands to be sampled from the low range
        low_range (tuple): (min, max) range for the low demand values
        high_range (tuple): (min, max) range for the high demand values
    Returns:
        np.ndarray: generated demand matrices of shape (num_samples, num_nodes, num_nodes)
    """
    assert 0.0 <= low_range[0] < low_range[1] < high_range[0] < high_range[1]
    num_nodes = len(G.nodes)

    dms = np.zeros((num_samples, num_nodes, num_nodes), dtype=np.float32)
    inds = np.random.choice(
            2, (num_samples, num_nodes, num_nodes), p=[fraction, 1 - fraction]
            ).astype(bool)
    
    dms = np.empty((num_samples, num_nodes, num_nodes), dtype=np.float32)

    dms[inds] = np.random.uniform(
        low_range[0], low_range[1], size=inds.sum()
    )

    dms[~inds] = np.random.uniform(
        high_range[0], high_range[1], size=(~inds).sum()
    )

    # No traffic between node and itself
    diag_idx = np.arange(num_nodes)
    dms[:, diag_idx, diag_idx] = 0.0

    return dms.reshape(num_samples, -1)

