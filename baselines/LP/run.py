import os
import torch
import argparse
import numpy as np

from tqdm import tqdm 
from .problem import Problem
from .LP_solver import LPSolver
from .objectives import MLU, MTF, MCF
from .pred_utils import moving_average
from .graph_utils import sol_dict_to_tensor

from data.build_dataloader import build_dataloader
from utils.get_paths import read_paths_from_file
from utils.eage_path import get_paths_to_edges, get_commodities_to_paths
from utils.read_data import read_graph_from_graphml, read_graph_from_json


def add_default_args(parser):
    """
    Add default command-line arguments for the LP solver.
    
    Configures the argument parser with parameters controlling the experiment
    setup, including topology selection, path counts, prediction settings,
    and optimization objectives.
    
    Args:
        parser (argparse.ArgumentParser): Parser to add arguments to
        
    Returns:
        argparse.ArgumentParser: Parser with added arguments
    """
    parser.add_argument('--seed', type=int, default = 12345)
    parser.add_argument('--num_paths', type=int, default=4)
    parser.add_argument("--topology", type=str, default='GEANT', 
                        choices=['Abilene', 'GEANT', 'CERNET', 'Meta-DB', 'Meta-WEB',
                                  'KDL', 'GtsCe', 'Colt', 'UsCarrier', 'Cogentco'],
                        help="Name of the topology to be used.")
    parser.add_argument("--topo_fname", type=str, default='./data/topology/GEANT.json', 
                        help="Name of .json file the topology was stored.")
    parser.add_argument("--dm_fname", type=str, default='./data/demand/GEANT.csv', 
                        help="Name of .csv file the real-world demand matrices were stored.")
    
    parser.add_argument('--hist_len', type=int, default = 12)
    parser.add_argument('--graphml', type=lambda x: x.lower() == "true", default = "False")
    parser.add_argument('--use_pred', type=lambda x: x.lower() == "true", default = "False")
    parser.add_argument('--method_name', type=str, default = 'LP',
                        choices = ['LP'], help="Baseline method to use")
    
    parser.add_argument("--objective", type=str, default='MTF', choices=['MLU', 'MTF', 'MCF'],
                        help="Objective function for optimization.") 
    parser.add_argument('--result_path', type=str, default='./results/', 
                        help='Location of computed weights or/and objectives.')

    return parser

def parse_args():
    """
    Parse command-line arguments for the LP solver.
    
    Creates an argument parser with default arguments and parses
    the command-line inputs.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)
    
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # Load network topology and capacity data
    if args.graphml:
        topo, capacities = read_graph_from_graphml(args.topo_fname)
        fname = f'./Topologies/topology_zoo/{args.topology}.txt'
    else:
        topo, capacities = read_graph_from_json(args.topo_fname) 
        fname = f'./Topologies/real_network/{args.topology}.txt'

    num_nodes = len(topo.nodes()) 
    
    # Create a mask to exclude diagonal elements (self-loops) from consideration
    mask = np.ones((num_nodes, num_nodes), dtype=bool)
    np.fill_diagonal(mask, 0)
    mask = mask.flatten()

    # Read paths from file for the given topology
    paths_dict = read_paths_from_file(filepath=fname, num_nodes=num_nodes, convert=False)
    _, _, test_loader = build_dataloader(topo, args.dm_fname, 32, 1, 1, args.hist_len, 
                                         split_ratio=(0.7, 0.1, 0.2), delete_loop=False)
    
    # Extract historical and target traffic matrices
    histories = test_loader.dataset.tm_seqences.float() 
    targets = test_loader.dataset.tm_preds.float() 

    # Read paths in converted format for matrix operations
    paths = read_paths_from_file(filepath=fname, num_nodes=num_nodes, convert=True)

    # Build sparse matrix mapping paths to edges
    p_matrix = get_paths_to_edges(topo, paths=paths)
    pm_coo = p_matrix.tocoo()
    paths_to_edges = torch.sparse_coo_tensor(np.vstack((pm_coo.row, pm_coo.col)), \
                                             torch.FloatTensor(pm_coo.data), 
                                             torch.Size(pm_coo.shape)) 
    
    # Build sparse matrix mapping commodities to their paths
    c_matrix = get_commodities_to_paths(topo, num_paths=p_matrix.shape[0], paths=paths)
    cm_coo = c_matrix.tocoo()
    commodities_to_paths = torch.sparse_coo_tensor(np.vstack((cm_coo.row, cm_coo.col)), \
                                                   torch.FloatTensor(cm_coo.data), 
                                                   torch.Size(cm_coo.shape)) 

    # List to store path demands computed for each time step
    path_demands = []

    # If using predicted traffic matrices, compute predictions
    if args.use_pred:
        args.method_name += '_pred'
        predictions = moving_average(histories, horizon=1).squeeze(1)

    setting = '{}_{}'.format(args.topology, args.method_name)

    # Process each test sample in the dataset
    for i in tqdm(range(len(test_loader.dataset)), total=len(test_loader.dataset)): 
        if args.use_pred:
            # Use predicted traffic matrix
            problem = Problem(topo, predictions[i].reshape(num_nodes, num_nodes).cpu().numpy(), seed=args.seed)
        else:
            # Use actual target traffic matrix
            problem = Problem(topo, targets[i].reshape(num_nodes, num_nodes).cpu().numpy(), seed=args.seed)

        # Solve the linear programming problem for the current traffic matrix
        lp = LPSolver(paths_dict, args.objective, args.num_paths)
        lp.solve(problem)
        
        # Convert solution dictionary to tensor format
        path_demand = sol_dict_to_tensor(lp.sol_dict, args.num_paths, problem.all_commodity_list)

        path_demands.append(path_demand)
    
    # Concatenate all path demands into a single tensor
    path_demands = torch.concat(path_demands, dim=0)
    
    # Convert capacities to tensor and move to appropriate device
    capacities = torch.Tensor(capacities, device=path_demands.device)
    
    # Apply mask to targets to exclude diagonal elements
    mask = torch.from_numpy(mask).to(targets.device)

    # Evaluate the solution using the specified objective function
    if args.objective == 'MLU':
        results = MLU(path_demands, targets[:, mask], capacities, paths_to_edges, commodities_to_paths)
    
    elif args.objective == 'MTF':
        results = MTF(path_demands, targets[:, mask])

    elif args.objective == 'MCF':
        results = MCF(path_demands, targets[:, mask])

    else:
        raise NotImplementedError()
    
    # Print the mean of the primary and secondary metrics
    print(f'{args.method_name}, {args.objective}: ', sum(results[0])/len(results[0]), ' Avg.:', sum(results[1])/len(results[1]))

    # Save the results to the specified path
    results_save_path = os.path.join(args.result_path, setting) 
    if not os.path.exists(results_save_path): 
        os.makedirs(results_save_path) 

    save_filename = f'/{args.objective}_results.npy'
    np.save(results_save_path + save_filename, np.array(results))