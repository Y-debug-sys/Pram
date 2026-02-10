from .dataset import MCFDataset
from torch.utils.data import DataLoader


def build_dataloader(
    topology, 
    demands_fname, 
    batch_size, 
    scale=10**9, 
    eval_batch_size=None, 
    window_size=12, 
    split_ratio=(0.7, 0.1, 0.2),
    delete_loop=True
):
    """ Build dataloaders for training, validation and testing.

    Args:
        topology (graph): topology 
        demands_fname (str): path to the csv file containing the demand matrices
        batch_size (int): batch size
        window_size (int): size of the sliding window
        split_ratio (tuple): ratio to split the dataset into training, validation and testing sets
        delete_loop (bool): delete self-looped commodites (the diagonal of demand matrix)
    
    Returns:
        tuple: training, validation and testing dataloaders
    """

    eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
    _, valid_ratio, test_ratio = split_ratio
    
    train_dataset = MCFDataset(topology, demands_fname, window_size, scale=scale, delete_loop=delete_loop,
                               mode='train', test_ratio=test_ratio, valid_ratio=valid_ratio)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                  num_workers=4, persistent_workers=True)

    valid_dataset = MCFDataset(topology, demands_fname, window_size, scale=scale, delete_loop=delete_loop,
                               mode='valid', test_ratio=test_ratio, valid_ratio=valid_ratio)
    valid_dataloader = DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=False, pin_memory=True,
                                  num_workers=4, persistent_workers=True)
    
    test_dataset = MCFDataset(topology, demands_fname, window_size, scale=scale, delete_loop=delete_loop,
                              mode='test', test_ratio=test_ratio, valid_ratio=valid_ratio)
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, pin_memory=True,
                                  num_workers=4, persistent_workers=True)

    return train_dataloader, valid_dataloader, test_dataloader
