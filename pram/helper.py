import argparse


def parse_args():
    """
    Parse command line arguments for the Pram script.
    
    Returns:
        argparse.Namespace: Object containing parsed arguments
    """
    parser = argparse.ArgumentParser(description='Script for running Pram') 
    
    # Basic configuration
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducible experiments')
    
    # Topology and data configuration
    parser.add_argument("--topology", type=str, default='GEANT', 
                        choices=['Abilene', 'GEANT', 'CERNET', 'Meta-DB', 'Meta-WEB', 'B4',
                                  'KDL', 'GtsCe', 'Colt', 'UsCarrier', 'Cogentco'],
                        help="Name of the network topology to be used.")
    parser.add_argument("--topo_fname", type=str, default='./data/topology/GEANT.json', 
                        help="Path to the .json file where the topology is stored.")
    parser.add_argument("--dm_fname", type=str, default='./data/demand/GEANT.csv', 
                        help="Path to the .csv or .npy file containing real-world demand matrices.")
    parser.add_argument("--num_paths", type=int, default=4, help="Number of optimized paths to consider during search.")
    parser.add_argument("--objective", type=str, default='MLU', choices=['MLU', 'MTF', 'MCF'],
                        help="Objective function for optimization (Maximum Link Utilization, Maximum Throughput Fraction, Minimum Cost Flow).")
    
    # Model hyperparameters
    parser.add_argument("--weight_std", type=int, default=-1,  
                        help="Hyperparameter for sampling weights; -1 indicates learned standard deviation.")
    parser.add_argument("--mllm_name", type=str, default='Llama', 
                        help="Multi-modal large language model used in Pram.")
    parser.add_argument("--d_mllm", type=int, default=4096, 
                        help="Token embedding dimension of the MLLM.")
    parser.add_argument('--mllm_layers', type=int, default=8, 
                        help='Number of layers used in the MLLM.')
    parser.add_argument("--d_model", type=int, default=256,
                        help="Hidden dimension of the cross-attention mechanism.")
    parser.add_argument('--n_heads', type=int, default=4, 
                        help='Number of cross-attention heads.')
    parser.add_argument('--history_len', type=int, default=12, 
                        help='Length of history demand sequence.')
    parser.add_argument('--num_agents', type=int, default=None, 
                        help='Number of MARL agents.')
    # parser.add_argument('--context_len', type=int, default=32, 
    #                     help='Trainable context embedding length.')
    
    # Execution parameters
    parser.add_argument('--is_training', type=int, default=1, help='Running status (1 for training, 0 for evaluation)')
    parser.add_argument("--scale", type=int, default=10**9, help="Normalized scale factor for demands.")
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/', 
                        help='Directory location for Pram checkpoints.') 
    parser.add_argument('--result_path', type=str, default='./results/', 
                        help='Directory location for computed weights and/or objectives.')
    
    # Training parameters
    parser.add_argument('--num_itrs', type=int, default=3, help='Number of repeated experiments.')
    parser.add_argument('--train_epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training inputs.')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='Batch size for evaluation inputs.')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for model adaptation.')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout probability')
    parser.add_argument('--num_marl_samples', type=int, default=5, help='Number of MARL samples.')

    # Synthetic data generation
    parser.add_argument("--synthesis", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Synthetic data type: 0 - Real data, 1 - Gravity model, 2 - Poisson model, 3 - Bimodal model.")
    parser.add_argument("--syn_num", type=int, default=3000, 
                        help="Number of generated demand matrices for synthetic data.")
    
    # Poisson distribution parameters
    parser.add_argument('--lam', type=float, default=0.5, help='Lambda parameter for Poisson distribution.')
    parser.add_argument('--decay', type=float, default=0.5, help='Decay factor for Poisson distribution.')
    # parser.add_argument('--const_factor', type=float, default=1, help='Scale factor')

    # Bimodal distribution parameters
    parser.add_argument('--fraction', type=float, default=0.5, help='Fraction of small demand in bimodal distribution.')

    args = parser.parse_args()
        
    return args