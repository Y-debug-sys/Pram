# import os
import networkx as nx

from itertools import islice
from collections import defaultdict


def node_ids_to_edge_tuple(node_ids):
    """Convert the node list path to the edge list path."""
    return [(node1, node2) for node1, node2 in zip(node_ids, node_ids[1:])]


def compute_ksp_paths(k, pairs, graph, save2txt=False, filepath=None, transform=False):
    """
    Computes or loads the k-shortest paths for each source-destination pair.

    Args:
        k (int): The number of shortest paths to compute for each pair.
        pairs (Iterable (np.array, list, OrderedSet... etc.)): A list of source-destination node pairs (src, dst).
        graph (networkx.Graph): The graph to compute the paths on.
        save2txt (bool): Whether to save the computed paths to a txt file.
        filepath (str): The path to save the txt file.
        transform (bool): Whether to transform the paths from edge tuples to node lists.

    Returns:
        dict: A dictionary where the keys are (src, dst) tuples and the values are lists of 
            k paths represented as edge tuples.

    Process:
        1. Compute the k-shortest paths for each source-destination pair in the given pairs list.
        - Use `networkx.shortest_simple_paths` to compute the paths.
        - If a pair has fewer than k paths, replicate the first path until the number of paths equals k.
        - Store the paths as edge tuples by converting node IDs using `node_ids_to_edge_tuple`.
        3. Save the computed k-shortest paths to the txt file for future use.
        4. Return the dictionary containing the k-shortest paths.
    """
    # Initialize the paths dictionary to store k-shortest paths for each node pair
    paths = dict()
    print(f"[Computing {k} Shortest Paths]")
    
    # Iterate through each source-destination pair to compute k-shortest paths
    for src, dst in pairs:
        # Get up to k shortest simple paths from source to destination
        all_paths = list(islice(nx.shortest_simple_paths(graph, src, dst, weight=None), k))
        
        # Ensure each node pair has exactly k paths by duplicating the first path if needed
        while len(all_paths) != k:
            all_paths.append(all_paths[0])
            
        # Convert each path from node list representation to edge tuple representation
        paths[(src, dst)] = [node_ids_to_edge_tuple(all_paths[i]) for i in range(k)]

    # Optionally save the computed paths to a text file for future use
    if save2txt:
        # Save the paths to a text file, such that each line contains the top-k shortest paths for a pair of nodes.
        with open(filepath, 'w') as file:
            for (src, dst), path_list in paths.items():
                # Convert each path to a string representation showing the sequence of nodes
                path_strs = ['-'.join(map(str, [edge[0] for edge in path] + [path[-1][1]])) for path in path_list]
                file.write(f"({src}, {dst}): {'; '.join(path_strs)}\n")
            file.close()

    # Optionally transform the paths from edge tuples to node lists
    if transform:
        # Transform the paths from edge tuples to node lists
        transformed_paths = {}
        for (src, dst), path_list in paths.items():
            # Convert each path to a node list representation
            transformed_paths[(src, dst)] = [[edge[0] for edge in path] + [path[-1][1]] for path in path_list]
            
        return transformed_paths
    
    return paths


def read_paths_from_file(filepath, num_nodes, convert=False):
    """Get the paths from the file."""
    # Initialize a dictionary to store paths for each source-destination pair
    paths = defaultdict(list)
    pid = 0
    
    # Open and read the file containing precomputed paths
    with open(filepath, 'r') as f:
        lines = sorted(f.readlines())
        # Create a mapping from node pair strings to the full path strings
        lines_dict = {line.split(":")[0] : line for line in lines if line.strip() != ""}
        
        # Iterate through all possible source-destination pairs
        for src in range(num_nodes):
            for dst in range(num_nodes):
                # Skip pairs where source equals destination
                if src == dst:
                    continue
                    
                # Find the line containing paths for the current source-destination pair
                if "(%d, %d)" % (src, dst) in lines_dict:
                    line = lines_dict["(%d, %d)" % (src, dst)].strip()
                else:
                    line = [l for l in lines if l.startswith("(%d, %d):" % (src, dst))]
                    if line == []:
                        continue
                    line = line[0]
                    line = line.strip()
                
                if not line: continue
                
                # Parse the source and destination nodes from the line
                i, j = list(map(int, line.split(":")[0].replace("(", "").replace(")", "").split(", ")))
                
                # Extract individual paths from the line
                paths_ = line.split(":")[1].split("; ")
                
                # Process each path in the list
                for p_ in paths_:
                    # Convert the path string to a list of node IDs
                    node_list = list(map( int, p_.split("-")))
                    
                    # Convert the path to either edge list or node list based on the convert flag
                    if convert:
                        # Convert node list to edge tuple representation
                        eage_list = node_ids_to_edge_tuple(node_list)
                        paths[(i, j)].append(eage_list)
                    else:
                        # Keep the path in node list representation
                        paths[(i, j)].append(node_list)
                        
                    pid += 1

    paths = dict(paths)
    return paths

