import torch
import numpy as np

from torch.utils.data import Dataset
from utils.get_dms import read_demands_from_csv


class MCFDataset(Dataset):

    """ Traffic Engineering Dataset. 

    Args:
        topology_filename (str): path to the json file containing the topology
        # opts_filename (str): path to the txt file containing the optimal values
        tms_filename (str): path to the csv file containing the traffic matrices
        window_size (int): size of the sliding window
        mode (str): mode of the dataset (train, valid, test)
        test_ratio (float): ratio of the test set
        valid_ratio (float): ratio of the validation set
    """

    def __init__(
        self,
        topology, 
        demands_fname,
        window_size=12,
        mode='train',
        test_ratio=0.2,
        valid_ratio=0.1,
        scale=10**9,
        delete_loop=True
    ):
        super(MCFDataset, self).__init__()

        self.topo, self.scale = topology, scale
        
        self.edge_index = None 
        self.node_features = None 
        self.edge_ids_per_path = None 

        self.edges_map = {(i, j): eid for eid, (i, j) in enumerate(self.topo.edges())}

        num_nodes = len(self.topo.nodes())
        mask = np.ones((num_nodes, num_nodes), dtype=bool)
        if delete_loop:
            np.fill_diagonal(mask, 0)
        self.mask = mask.flatten()

        # optimal_values = self.read_opts_from_txt(opts_filename)[window_size:]
        try:
            dms = read_demands_from_csv(demands_fname)
        except Exception as e:
            dms = np.load(demands_fname)

        tms = self.normalize(dms[:, self.mask])
        tm_preds = torch.from_numpy(tms[window_size:, :])
        tm_seqences = np.array(self.sample_silding_windows(tms, window_size))
        tm_seqences = torch.from_numpy(tm_seqences)

        if mode == 'train':
            self.train_tms = tms[:int((1 - test_ratio - valid_ratio) * len(tm_seqences))]
            self.tm_seqences = tm_seqences[:int((1 - test_ratio - valid_ratio) * len(tm_seqences)), :]
            self.tm_preds = tm_preds[:int((1 - test_ratio - valid_ratio) * len(tm_seqences)), :]
            # self.optimal_values = optimal_values[:int((1 - test_ratio - valid_ratio) * len(tm_seqences))]
        elif mode == 'test':
            self.tm_seqences = tm_seqences[- int(test_ratio * len(tm_seqences)):, :]
            self.tm_preds = tm_preds[- int(test_ratio * len(tm_seqences)):, :]
            # self.optimal_values = optimal_values[- int(test_ratio * len(tm_seqences)):]
        elif mode == 'valid':
            self.tm_seqences = tm_seqences[- int((valid_ratio + test_ratio) * len(tm_seqences)): - int(test_ratio * len(tm_seqences)), :]
            self.tm_preds = tm_preds[- int((valid_ratio + test_ratio) * len(tm_seqences)): - int(test_ratio * len(tm_seqences)), :]
            # self.optimal_values = optimal_values[- int((valid_ratio + test_ratio) * len(tm_seqences)): - int(test_ratio * len(tm_seqences))]
        else:
            raise ValueError(f"Invalid mode: {mode}. Please choose between 'train', 'valid' and 'test'.")

    def get_padded_edge_ids_per_path(self, pij) -> torch.Tensor:
        """
        Retrieves or computes the padded edge IDs for each path in pij.

        Args:
            pij (dict): Dictionary where the keys are (src, dst) pairs and values are lists of paths 
                        represented as edge tuples.

        Returns:
            torch.Tensor: A tensor of padded edge IDs for each path. The tensor is padded with -1 
                        to ensure all paths have the same length, and has shape 
                        (num_paths, max_path_length).

        Process:
            1. Compute the edge IDs for each path:
            - For each (src, dst) pair in pij, retrieve the list of paths.
            - For each path, convert its edges to indices using the edges_map.
            - Append the edge indices to a list.
            2. Pad the edge index lists so that all paths have the same length, using -1 as the 
            padding value.
            3. Convert the padded edge indices to a tensor of int64 type and save it to the file.
            4. Return the padded edge IDs tensor.
        """

        if self.edge_ids_per_path is not None:
            return self.edge_ids_per_path

        paths_edges_list = []
        for key in pij.keys():
            for path in pij[key]:
                edges_list = []
                for edge in path:
                    index = self.edges_map[edge]
                    edges_list.append(index)
                paths_edges_list.append(torch.tensor(edges_list, dtype=torch.int32))
                
        padded_edge_ids_per_path = torch.nn.utils.rnn.pad_sequence(paths_edges_list, batch_first=True,
                                                                   padding_value=-1.0)
        padded_edge_ids_per_path = padded_edge_ids_per_path.to(dtype=torch.int64)
        
        self.edge_ids_per_path = padded_edge_ids_per_path
        return padded_edge_ids_per_path
    
    def sample_silding_windows(self, tms, window_size):
        """
        Create a sliding window of traffic matrices.

        Args:
            tms (numpy.array): traffic matrices
            window_size (int): size of the sliding window

        Returns:
            list: list of sliding windows
        """

        windows = []
        for i in range(len(tms) - window_size):
            window = tms[i:i + window_size]
            windows.append(window)

        return windows

    def __len__(self):
        return self.tm_seqences.shape[0]
    
    def __getitem__(self, index):
        """ Get item from the dataset. """
        return self.tm_seqences[index], self.tm_preds[index]
    
    def normalize(self, x): 
        """ Normalize the input tensor. """
        return x / self.scale

    def denormalize(self, x): 
        """ Denormalize the input tensor. """
        return x * self.scale
    
    def get_tm_histories_std(self):
        """Get the standard deviation of the train traffic per s-d pairs."""
        tm_hist_std = np.std(self.train_tms, axis=0)
        return tm_hist_std
