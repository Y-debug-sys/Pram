import os
import torch
import random
import numpy as np

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs

from pram.solver import PramSolver
from pram.helper import parse_args
from pram.model_qwen import PramModel
from pram.divider import draw_paths_in_group, draw_paths_per_source

from utils.read_data import read_graph_data
from env.logger import ExperimentLoggingManager
from data.build_dataloader import build_dataloader


def _t2n(x):
    """
    Convert a tensor to a numpy array.
    
    Detaches the tensor from the computation graph and moves it to CPU
    before converting to a numpy array.
    
    Args:
        x (torch.Tensor): Input tensor to convert
        
    Returns:
        numpy.ndarray: Numpy array representation of the input tensor
    """
    return x.detach().cpu().numpy()

def set_seed(seed=2026):
    """
    Set random seeds for reproducible experiments.
    
    Sets the random seed for numpy, python's random module, and pytorch
    to ensure reproducible results across runs.
    
    Args:
        seed (int): Random seed value to use for all random number generators (default 2026)
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = True 
    torch.backends.cudnn.benchmark = True 

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    # Set random seed for reproducibility
    set_seed(args.seed) 

    # Configure distributed training settings
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)  # DDP config
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')  # DeepSpeed config
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)  # Accelerator config

    # Load topology, raw capacities, and paths for the network
    topo, raw_capacities, paths = read_graph_data(args)
    num_nodes = len(topo.nodes())

    # Generate visualization of paths if not already present
    dname = f'./pram/{args.topology}_figs'
    if not os.path.exists(dname) or not os.listdir(dname): 
        if args.num_agents:
            # Draw paths grouped by agents
            draw_paths_in_group(topo, paths, out_dir=dname, scale=args.scale, num_agents=args.num_agents)
        else: 
            # Draw paths per source node
            draw_paths_per_source(topo, paths, out_dir=dname, scale=args.scale)

    # Run multiple iterations of training and testing
    for ii in range(args.num_itrs):
        # Initialize logging and result management
        logging_manager = ExperimentLoggingManager()
        logger = logging_manager.get_logger(args.objective)
        model_save_path = logging_manager.get_model_directory()
        result_save_path = logging_manager.get_result_directory()

        # Build data loaders for training, validation, and testing
        train_loader, valid_loader, test_loader = build_dataloader(topo, args.dm_fname, args.batch_size, args.scale, 
                                                                   args.eval_batch_size, args.history_len, 
                                                                   split_ratio=(0.7, 0.1, 0.2))
        logger.log('Iteration: {}, Topology: {}, Objective: {}'.format(ii, args.topology, args.objective))
        logger.log('Training size: {}, Validation size: {}, Testing size: {}'.format(len(train_loader.dataset), 
                                                                              len(valid_loader.dataset), len(test_loader.dataset)))
        logger.log('Checkpoint directory: {}, Result directory: {}'.format(model_save_path, result_save_path))
        
        # Initialize the Pram model with specified parameters
        model = PramModel(args.d_mllm, num_nodes, args.num_paths, args.mllm_name, args.mllm_layers, args.d_model, 
                          mcf_objective=args.objective, len_context=num_nodes, dropout=args.dropout)
        logger.log('Model architecture: {}'.format(model))

        # Prepare optimizer with only trainable parameters
        trained_parameters = []
        for p in model.parameters(): 
            if p.requires_grad is True: 
                trained_parameters.append(p)
        optimizer = torch.optim.Adam(trained_parameters, lr=args.learning_rate)

        # Prepare all components for distributed training with Accelerate
        train_loader, valid_loader, test_loader, model, optimizer = accelerator.prepare(train_loader, valid_loader, 
                                                                                        test_loader, model, optimizer)
        
        # Normalize capacities and initialize the Pram solver
        capacities = train_loader.dataset.normalize(torch.tensor(raw_capacities))
        solver = PramSolver(args, model, topo, paths, accelerator, train_loader, valid_loader, test_loader,
                            optimizer, image_path=dname, capacities=capacities)

        if args.is_training: 
            # Train the model if in training mode
            solver.adapt(model_save_path, logger) 
        else:
            # Load pre-trained model if in inference mode
            solver.load_checkpoint(model_save_path, logger)

        # Test the model and save results
        results_ii = solver.test(logger)
        save_filename = os.path.join(result_save_path, f'{args.objective}_results.npy')
        np.save(save_filename, results_ii)