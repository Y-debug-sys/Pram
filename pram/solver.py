import time
import torch
import numpy as np

from .prompt import PNGLoader
from env.marl_env import PramEnv
from .early_stopping import EarlyStopping
from utils.eage_path import get_paths_to_edges, get_commodities_to_paths, build_p2e_from_paths


class PramSolver(object): 
    """
    Solver class for the PRAM model to handle multi-commodity flow problems.
    
    This class manages the training, validation, and testing of the PRAM model
    for solving multi-commodity flow problems with different objectives (MLU, MTF, MCF).
    
    Args:
        args: Arguments containing hyperparameters and configuration
        model: The PRAM model to be trained
        topology: Network topology (graph) for the problem
        paths: Dictionary mapping (source, destination) pairs to paths
        accelerator: Accelerator for distributed training
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        optimizer: Optimizer for model training
        image_path: Path to the image files for visualization
        capacities: Capacities of the network edges
        objective: Objective function ('MLU', 'MTF', or 'MCF')
        synthetic: Boolean indicating if using synthetic data
    """

    def __init__(
        self,
        args,
        model,
        topology, 
        paths,
        accelerator, 
        train_loader, 
        valid_loader, 
        test_loader,
        optimizer, 
        image_path,
        capacities,
        synthetic=False
    ):
        super(PramSolver, self).__init__()

        # Set the objective function based on the specified objective
        self.objective = args.objective.upper()
        if self.objective == 'MLU':
            self.obj_test = self.LU
        elif self.objective == 'MTF': 
            self.obj_test = self.TF
        elif self.objective == 'MCF':
            self.obj_test = self.CF
        else:
            raise NotImplementedError()
        
        # Load images for visualization
        self.img_loader = PNGLoader(image_path)
        self.imgs = self.img_loader.get_all() 

        # Build path-to-edge representation and path matrices
        p2e = build_p2e_from_paths(topology, paths=paths)
        p_matrix = get_paths_to_edges(topology, paths=paths)
        pm_coo = p_matrix.tocoo()
        paths_to_edges = torch.sparse_coo_tensor(np.vstack((pm_coo.row, pm_coo.col)), \
                                                 torch.FloatTensor(pm_coo.data), 
                                                 torch.Size(pm_coo.shape)) 
        
        c_matrix = get_commodities_to_paths(topology, num_paths=p_matrix.shape[0], paths=paths)
        cm_coo = c_matrix.tocoo()
        commodities_to_paths = torch.sparse_coo_tensor(np.vstack((cm_coo.row, cm_coo.col)), \
                                                       torch.FloatTensor(cm_coo.data), 
                                                       torch.Size(cm_coo.shape)) 

        # Initialize early stopping and environment
        self.early_stopping = EarlyStopping(accelerator, args.patience, verbose=True) 
        self.env = PramEnv(args, p2e, len(topology.nodes()), capacities, accelerator, paths_to_edges, 
                           commodities_to_paths, num_agents=args.num_agents)
        
        # Store essential components and parameters
        self.args, self.model = args, model 
        self.num_steps = args.num_marl_samples
        self.topology, self.optimizer = topology, optimizer
        self.accelerator, self.from_topo_zoo = accelerator, synthetic
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader 

        # Move tensors to the appropriate device
        self.capacities = capacities.float().to(accelerator.device)
        self.paths_to_edges = paths_to_edges.float().to(accelerator.device) 
        self.commodities_to_paths = commodities_to_paths.float().to(accelerator.device) 
    
    def adapt(self, save_path, logger, report_freq=10):
        """
        Main training loop for the PRAM model.
        
        Args:
            save_path: Path to save the best model checkpoint
            logger: Logger for recording training progress
            report_freq: Frequency of reporting training metrics
        """
        time_now = time.time()
        logger.log("Start training...")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            self.model.train()
            epoch_time = time.time()
            train_losses, valid_losses = [], [[], []]
            train_steps = len(self.train_loader)

            for idx, (dms, preds) in enumerate(self.train_loader):
                iter_count += 1
                self.optimizer.zero_grad()

                # Move data to the appropriate device
                dms = dms.float().to(self.accelerator.device) 
                preds = preds.float().to(self.accelerator.device) 
                loss = 0
                accumulated_size = dms.shape[0]

                for i in range(accumulated_size):
                    torch.cuda.empty_cache()
                    # Get action and log probability from the model
                    action, log_probability = self.model(self.imgs, dms[i:i+1], self.from_topo_zoo)
                    # Step in the environment and get reward
                    reward = self.env.step(action, preds[i:i+1], self.num_steps)
                    train_losses.append(reward.mean().item())

                    # Calculate the loss using policy gradient
                    loss_initial = - log_probability * reward
                    norm_factor = torch.clamp_min(torch.abs(loss_initial).detach(), min=1e-7)
                    loss = torch.nan_to_num(loss_initial / norm_factor, nan=1.0, posinf=None, neginf=None).mean()
                    loss = loss / accumulated_size
                    self.accelerator.backward(loss)

                if (idx + 1) % report_freq == 0: 
                    self.accelerator.print( 
                        "\titers: {0}, epoch: {1} | real-time reward: {2:.7f}".format(idx + 1, epoch + 1, abs(train_losses[-1])))
                    speed = (time.time() - time_now) / (iter_count+1)
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - idx)
                    self.accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                self.optimizer.step()

            self.accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time)) 
            self.accelerator.print("Validating ...")

            self.model.eval()
            with torch.no_grad():
                # Validation loop
                for i, (dms, preds) in enumerate(self.valid_loader):
                    dms = dms.float().to(self.accelerator.device) 
                    preds = preds.float().to(self.accelerator.device)
                    for i in range(dms.shape[0]):
                        torch.cuda.empty_cache()
                        # Get action from the model in test mode
                        action, _ = self.model(self.imgs, dms[i:i+1], self.from_topo_zoo, True)
                        # Calculate results for validation
                        result = self.get_result(action, preds[i:i+1])
                        valid_losses[0] += result[0]
                        valid_losses[1] += result[1]
                
            # Calculate average losses for logging
            valid_loss = sum(valid_losses[0])/len(valid_losses[0])
            train_loss = np.average(train_losses)
            self.accelerator.print("Epoch: {0} | Train Objective: {1:.7f}, Valid Objective: {2:.7f}".format(epoch + 1, abs(train_loss), abs(valid_loss)))

            # Check for early stopping
            self.early_stopping(valid_loss, self.model, save_path)
            if self.early_stopping.early_stop:
                self.accelerator.print("Early stopping")
                break
            
        self.accelerator.wait_for_everyone()
        logger.log('Finished training! Cost time: {0}s'.format(time.time() - time_now))

    def load_checkpoint(self, path, logger):
        """
        Load a model checkpoint from the specified path.
        
        Args:
            path: Path to the directory containing the checkpoint
            logger: Logger for recording loading progress
        """
        # Load the state dictionary from the checkpoint
        trainable_state_dict = torch.load(path + '/' + 'checkpoint.pt', map_location=self.accelerator.device)
        logger.log('Loaded model from {}'.format(path))
        model_state_dict = self.model.state_dict()

        # Copy parameters from the checkpoint to the model
        for name, param in trainable_state_dict.items(): 
            if name in model_state_dict: 
                model_state_dict[name].copy_(param)
            elif name.replace("module.", "") in model_state_dict:
                model_state_dict[name.replace("module.", "")].copy_(param)
            else: 
                self.accelerator.print(f'Not load {name}')

        self.model.load_state_dict(model_state_dict)
        logger.log('Model loaded.')

    @torch.no_grad()
    def test(self, logger):
        """
        Evaluate the model on the test dataset.
        
        Args:
            logger: Logger for recording evaluation progress
        
        Returns:
            results: Evaluation results containing objective values
        """
        self.model.eval()
        results = [[], []] 
        logger.log("Evaluating...")

        for i, (dms, preds) in enumerate(self.test_loader): 
            dms = dms.float().to(self.accelerator.device) 
            preds = preds.float().to(self.accelerator.device) 

            for i in range(dms.shape[0]):
                torch.cuda.empty_cache()
                # Get action from the model in test mode
                action, _ = self.model(self.imgs, dms[i:i+1], self.from_topo_zoo, True)
                # Calculate results for testing
                result = self.get_result(action, preds[i:i+1])
                results[0] += result[0]
                results[1] += result[1]
        
        # Log the average results
        logger.log(f'Average {self.objective}: {abs(sum(results[0])/len(results[0]))}, \
                   Average Mean {self.objective}: {abs(sum(results[1])/len(results[1]))}') 
        
        return np.array(results) 
    
    @torch.no_grad()
    def get_result(self, action, current_demand):
        """
        Calculate the objective result based on the action and current demand.
        
        Args:
            action: Action taken by the model (path weights)
            current_demand: Current demand matrix
        
        Returns:
            Result of the objective function evaluation
        """
        return self.obj_test(action, current_demand, self.capacities, self.paths_to_edges, self.commodities_to_paths)

    @staticmethod
    def LU(
        path_weights,
        true_demands,
        capacities,
        paths_to_edges,
        commodities_to_paths
    ):
        """
        Calculate Loss for Maximum Link Utilization (MLU) objective.
        
        Args:
            path_weights: Weights assigned to different paths
            true_demands: Actual demands between node pairs
            capacities: Capacities of the network edges
            paths_to_edges: Matrix mapping paths to edges
            commodities_to_paths: Matrix mapping commodities to paths
        
        Returns:
            losses: List containing max and average link utilization values
        """
        losses = [[], []]
        capacities = capacities.unsqueeze(-1)
        for i in range(true_demands.shape[0]):
            wi = torch.transpose(path_weights[[i]], 0, 1)
            di = true_demands[[i]]

            # Calculate weights for each commodity
            commodity_total_weight = commodities_to_paths.matmul(wi)
            paths_over_total = commodities_to_paths.transpose(0, 1).matmul(1.0 / torch.clamp_min(commodity_total_weight, 1e-7))
            wi = wi.mul(paths_over_total)

            # Calculate flow on edges
            tmp_demand_on_paths = commodities_to_paths.transpose(0, 1).matmul(di.transpose(0, 1)) 
            demand_on_paths = tmp_demand_on_paths.mul(wi) 
            flow_on_edges = paths_to_edges.transpose(0, 1).matmul(demand_on_paths) 
            congestion = flow_on_edges.divide(capacities) 

            # Calculate max and average utilizations
            max_utils = torch.max(congestion.flatten()) 
            avg_utils = torch.mean(congestion.flatten()) 

            losses[0].append(max_utils.item())
            losses[1].append(avg_utils.item())

        return losses
    
    @staticmethod
    def TF(
        path_weights,
        true_demands,
        capacities,
        paths_to_edges,
        commodities_to_paths
    ):
        """
        Calculate flow ratios using the TF (Traffic Flow) algorithm
        
        Args:
            path_weights: weights assigned to different paths
            true_demands: actual traffic demands for each commodity
            capacities: capacity constraints for edges in the network
            paths_to_edges: mapping matrix from paths to edges
            commodities_to_paths: mapping matrix from commodities to paths
            
        Returns:
            losses: list containing flow ratio metrics for each commodity
        """
        # Initialize loss tracking arrays
        losses = [[], []]
        
        # Reshape capacities to match dimensions for computation
        capacities = capacities.unsqueeze(-1)
        path_weights = torch.clamp_min(path_weights, min=1e-8)
        
        # Process each demand/commodity individually
        for i in range(true_demands.shape[0]): 
            # Extract path weights and demand for current commodity
            wi = path_weights[[i]]
            di = true_demands[[i]]

            # Calculate total weight per commodity by summing all paths for that commodity
            commodity_total_weight = commodities_to_paths.matmul(wi.transpose(0,1))
            
            # Compute ratio of each path's contribution to its commodity's total weight
            paths_over_total = commodities_to_paths.transpose(0, 1).matmul(1.0 / torch.clamp_min(commodity_total_weight, 1e-7))
            
            # Adjust path weights based on their proportional contribution to each commodity
            wi = wi.transpose(0,1).mul(paths_over_total).transpose(0,1)

            # Scale path weights by demand values to get actual flows
            wi = (wi.reshape(1, -1, di.shape[-1]) * di.unsqueeze(1)).reshape(wi.shape).transpose(0, 1)

            # Calculate total flow for current commodity
            total_flow = torch.sum(di)

            # Compute flow on each edge by aggregating flows from all paths using that edge
            flow_on_edges = paths_to_edges.transpose(0, 1).matmul(wi)
            
            # Calculate excess flow beyond capacity on each edge
            excess_flow = torch.relu(flow_on_edges - capacities)
            Total_excess_flow = excess_flow.sum() 
            
            # Calculate maximum achievable flow given capacity constraints
            max_flow = total_flow - Total_excess_flow

            # Iteratively adjust flows to respect capacity constraints
            while Total_excess_flow != 0:
                # Reduce path weights proportionally to remaining capacity
                wi = wi * max_flow / total_flow
                
                # Recalculate flows on edges with adjusted weights
                flow_on_edges = paths_to_edges.transpose(0, 1).matmul(wi)
                
                # Recalculate excess flow
                excess_flow = torch.relu(flow_on_edges - capacities)
                Total_excess_flow = excess_flow.sum() 
                
                # Update max flow considering new excess flow
                max_flow = max_flow - Total_excess_flow
        
            # Calculate the ratio of achievable flow to requested flow
            max_flow_ratio = max_flow / total_flow

            # Store flow ratio metrics for this commodity
            losses[0].append(max_flow_ratio.item())
            losses[1].append(max_flow_ratio.item())

        return losses
    
    @staticmethod
    def CF(
        path_weights,
        true_demands,
        capacities,
        paths_to_edges,
        commodities_to_paths
    ):
        """
        Calculate concurrent flow metrics using the CF (Concurrent Flow) algorithm
        
        Args:
            path_weights: weights assigned to different paths
            true_demands: actual traffic demands for each commodity
            capacities: capacity constraints for edges in the network
            paths_to_edges: mapping matrix from paths to edges
            commodities_to_paths: mapping matrix from commodities to paths
            
        Returns:
            losses: list containing average and minimum concurrent flow metrics
        """
        # Initialize loss tracking arrays
        losses = [[], []]
        
        # Reshape capacities to match dimensions for computation
        capacities = capacities.unsqueeze(-1)
        path_weights = torch.clamp_min(path_weights, min=1e-8)
        
        # Process each demand/commodity individually
        for i in range(true_demands.shape[0]): 
            # Extract path weights and demand for current commodity
            wi = path_weights[[i]]
            di = true_demands[[i]]

            # Calculate total weight per commodity by summing all paths for that commodity
            commodity_total_weight = commodities_to_paths.matmul(wi.transpose(0,1))
            
            # Compute ratio of each path's contribution to its commodity's total weight
            paths_over_total = commodities_to_paths.transpose(0, 1).matmul(1.0 / torch.clamp_min(commodity_total_weight, 1e-7))
            
            # Adjust path weights based on their proportional contribution to each commodity
            wi = wi.transpose(0,1).mul(paths_over_total).transpose(0,1)

            # Scale path weights by demand values to get actual flows
            wi = (wi.reshape(1, -1, di.shape[-1]) * di.unsqueeze(1)).reshape(wi.shape).transpose(0, 1)

            # Calculate total flow for current commodity
            total_flow = torch.sum(di)

            # Compute flow on each edge by aggregating flows from all paths using that edge
            flow_on_edges = paths_to_edges.transpose(0, 1).matmul(wi)
            
            # Calculate excess flow beyond capacity on each edge
            excess_flow = torch.relu(flow_on_edges - capacities)
            Total_excess_flow = excess_flow.sum() 
            
            # Calculate maximum achievable flow given capacity constraints
            max_flow = total_flow - Total_excess_flow

            # Iteratively adjust flows to respect capacity constraints
            while Total_excess_flow != 0:
                # Reduce path weights proportionally to remaining capacity
                wi = wi * max_flow / total_flow
                
                # Recalculate flows on edges with adjusted weights
                flow_on_edges = paths_to_edges.transpose(0, 1).matmul(wi)
                
                # Recalculate excess flow
                excess_flow = torch.relu(flow_on_edges - capacities)
                Total_excess_flow = excess_flow.sum() 
                
                # Update max flow considering new excess flow
                max_flow = max_flow - Total_excess_flow

            # Calculate concurrent flow as ratio of actual flow to demand for each path
            concurrent_flow = wi.reshape(1, -1, di.shape[-1]).sum(dim=1) / torch.clamp_min(di, min=1e-7)
            
            # Set concurrent flow to 1.0 for commodities with zero demand
            concurrent_flow[di==0] = 1.0

            # Calculate average and minimum concurrent flow ratios
            avg_concurrent = torch.mean(concurrent_flow)
            max_concurrent = torch.min(concurrent_flow)

            # Store concurrent flow metrics for this commodity
            losses[0].append(avg_concurrent.item())
            losses[1].append(max_concurrent.item())

        return losses
