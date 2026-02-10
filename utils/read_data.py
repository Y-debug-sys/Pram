import os
import math
import json
import numpy as np
import networkx as nx

from .get_dms import gravity_model, possion_model, bimodal_model
from .get_paths import compute_ksp_paths, read_paths_from_file


def get_capacities_from_graph(graph):
    """
    Extract capacities from graph edges.

    Args:
        graph (networkx.Graph): Input graph with edge capacity attributes

    Returns:
        list: List of capacities for each edge in the graph
    """
    capacities = [float(data['capacity']) for u, v, data in graph.edges(data=True)]
    return capacities


def read_graph_from_json(topology_filename):
    """
    Read the graph from a json file.
    Args:
        topology_filepath (str): path to the json file containing the topology
        
    Returns:
        networkx.Graph: graph object
        list: capacities of the links
    """
    assert topology_filename.endswith('.json')
    with open(topology_filename) as f:
        data = json.load(f)

    graph = nx.readwrite.json_graph.node_link_graph(data)
    capacities = get_capacities_from_graph(graph)

    return graph, capacities


def read_graph_from_graphml(topology_filename):
    """
    Read a graph from a GraphML file.
    Args:
        topology_filepath (str): path to the graphml file containing the topology
        
    Returns:
        networkx.Graph: graph object
        list: capacities of the links
    """
    assert topology_filename.endswith('.graphml')
    file_G = nx.read_graphml(topology_filename).to_directed()
    if isinstance(file_G, nx.MultiDiGraph):
        file_G = nx.DiGraph(file_G)

    graph, capacities = [], []
    # Pick largest strongly connected component
    for scc_ids in nx.strongly_connected_components(file_G):
        scc = file_G.subgraph(scc_ids)
        if len(scc) > len(graph):
            graph = scc
    
    graph = nx.convert_node_labels_to_integers(graph)
    # For TZ topologies, assume every link has 1000 Mbps of capacity
    for u, v in graph.edges():
        graph[u][v]['capacity'] = 1000.0
        capacities.append(1000.0)
    
    return graph, capacities


def read_graph_data(args):
    """
    Reads graph data based on arguments, with option to synthesize demand matrices from GraphML or read from JSON format.

    Args:
        args: Argument object containing the following attributes:
            - synthesis (int): Synthesis type identifier (0 for no synthesis, 1-3 for different synthesis methods)
            - topo_fname (str): Topology file path
            - dm_fname (str): Demand matrix file path
            - syn_num (int): Number of synthesized nodes
            - scale (float): Scale factor
            - num_paths (int): Number of k-shortest paths to compute
            - lam (float): Lambda parameter for Poisson model
            - decay (float): Decay parameter
            - fraction (float): Fraction parameter for bimodal model

    Returns:
        tuple: A tuple containing:
            - topo: Topology of the graph
            - capacities: List of edge capacities
            - paths: List of k-shortest paths for all node pairs
    """
    # Determine whether to synthesize demand matrices or read from existing files
    if args.synthesis != 0:
        topo, capacities = read_graph_from_graphml(args.topo_fname)

        # Check if demand matrix file exists, otherwise generate synthetic demands
        if not os.path.exists(args.dm_fname):

            if args.synthesis == 1:
                # Generate demands using gravity model
                dms = gravity_model(args.syn_num, topo, args.scale * 10)

            elif args.synthesis == 2:
                # Generate demands using Poisson model
                dms = possion_model(args.syn_num, topo, args.lam, args.decay, args.scale * 10)

            elif args.synthesis == 3:
                # Generate demands using bimodal model with specified ranges
                dms = bimodal_model(args.syn_num, topo, args.fraction, 
                                    low_range=[0, math.sqrt(args.scale) / 10], 
                                    high_range=[args.scale / 5, args.scale * 20])

            else:
                raise NotImplementedError
            
            # Save generated demands to numpy file
            np.save(args.dm_fname, dms)

    else:
        # Read topology from JSON file
        topo, capacities = read_graph_from_json(args.topo_fname) 

    # Generate corresponding text filename from topology filename
    base, _ = os.path.splitext(args.topo_fname)
    fname = base + '.txt'
    num_nodes = len(topo.nodes())

    # Compute k-shortest paths if path file doesn't exist
    if not os.path.exists(fname): 
        # Create all node pairs for path computation
        pairs = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
        # Compute k-shortest paths and save to text file
        _ = compute_ksp_paths(k=args.num_paths, pairs=pairs, graph=topo, save2txt=True, 
                              filepath=fname, transform=True)

    # Load precomputed paths from file
    paths = read_paths_from_file(filepath=fname, num_nodes=num_nodes, convert=True)

    return topo, capacities, paths


if __name__ == '__main__':
    pass