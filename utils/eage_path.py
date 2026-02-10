import torch
import numpy as np

from scipy.sparse import csr_matrix, lil_matrix


def get_paths_to_edges(topology, paths):
    """Get the paths_to_edges matirx, [num_paths, num_edges]
       paths_to_edges[i, j] = 1 if edge j is in path i
       
    Constructs a sparse matrix that maps paths to edges in a network topology.
    
    Args:
        topology: NetworkX graph representing the network topology
        paths: Dictionary mapping (source, destination) tuples to a list of paths,
               where each path is a list of edges represented as (node_i, node_j) tuples
    
    Returns:
        scipy.sparse.csr_matrix: Sparse matrix of shape [num_paths, num_edges] where
                                 each row corresponds to a path and each column to an edge.
                                 A value of 1 indicates the edge is part of the path.
    """
    num_nodes = topology.number_of_nodes()
    num_edges = topology.number_of_edges()
    paths_arr = []

    # Get the adjacency matrix of the topology.
    adj = np.zeros((num_nodes, num_nodes))
    for s in range(num_nodes):
        for d in range(num_nodes):
            if s == d:
                continue
            if d in topology[s]:
                adj[s,d] = 1

    eid = 0
    edges_map = dict()
    # Get the map from the edge to the edge id.
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i,j] == 1:
                edges_map[(i, j)] = eid
                eid += 1
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            for p in paths[(i, j)]:
                # Map each edge in the path to its corresponding edge ID
                p_ = [edges_map[e] for e in p]
                # Create a binary vector indicating which edges are in this path
                p__ = np.zeros((int(num_edges),))
                for k in p_:
                    p__[k] = 1
                paths_arr.append(p__)

    return csr_matrix(np.stack(paths_arr))


def get_commodities_to_paths(topology, num_paths, paths):
    """Get the commodities_to_paths matrix, [num_commodities, num_paths]
       commodities_to_paths[i, j] = 1 if path j is a candidate path for commodity i
       
    Constructs a sparse matrix that maps commodities (source-destination pairs) to paths.
    
    Args:
        topology: NetworkX graph representing the network topology
        num_paths: Total number of paths across all source-destination pairs
        paths: Dictionary mapping (source, destination) tuples to a list of paths
    
    Returns:
        scipy.sparse.csr_matrix: Sparse matrix of shape [num_commodities, num_paths] where
                                 each row corresponds to a commodity (s-d pair) and each column 
                                 to a path. A value of 1 indicates the path is a candidate 
                                 for the commodity.
    """
    num_nodes = topology.number_of_nodes()
    # Create a sparse matrix to track which paths belong to which commodity
    commodities_to_paths = lil_matrix((num_nodes * (num_nodes - 1), num_paths))
    commid = 0  # commodity index
    pathid = 0  # path index
    for src in range(num_nodes):
        for dst in range(num_nodes):
            if src == dst:
                continue
            # For each path of the current source-destination pair
            for _ in paths[(src, dst)]:
                commodities_to_paths[commid, pathid] = 1
                pathid += 1
            commid += 1
    return csr_matrix(commodities_to_paths)


def build_p2e_from_paths(topology, paths):
    """
    Build p2e (path-to-edge COO) from the inputs of get_paths_to_edges.

    Creates a coordinate format (COO) sparse matrix that maps paths to edges
    in a network topology. This representation is more efficient for certain
    graph neural network operations.

    Args:
        topology: NetworkX graph representing the network topology
        paths: Dictionary mapping (source, destination) tuples to a list of paths,
               where each path is a list of edges represented as (node_i, node_j) tuples

    Returns:
        torch.LongTensor: Tensor of shape [2, num_path_edge_incidence] where:
                         - p2e[0] contains path indices
                         - p2e[1] contains corresponding edge indices
                         Each column represents an incidence between a path and an edge
    """

    num_nodes = topology.number_of_nodes()

    # ---------- 1. build edge -> idx ----------
    # Create adjacency matrix representation of the topology
    adj = np.zeros((num_nodes, num_nodes))
    for s in range(num_nodes):
        for d in range(num_nodes):
            if s != d and d in topology[s]:
                adj[s, d] = 1

    # Map each edge to a unique index
    edges_map = {}
    eid = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i, j] == 1:
                edges_map[(i, j)] = eid
                eid += 1

    # ---------- 2. build p2e ----------
    # Create lists for the coordinate format (COO) representation
    src = []  # path indices
    dst = []  # edge indices

    path_idx = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            # For each path between nodes i and j
            for path in paths[(i, j)]:
                # Add an entry for each edge in the path
                for e in path:          # e = (u, v)
                    src.append(path_idx)      # record the path index
                    dst.append(edges_map[e])  # record the edge index
                path_idx += 1

    # Create the COO tensor with path and edge indices
    p2e = torch.tensor([src, dst], dtype=torch.long)
    return p2e