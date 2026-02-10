import torch
from torch import nn


class DtypeAlignWrapper(nn.Module):
    """
    A wrapper module that ensures input tensors have the same dtype as the module's parameters.
    
    This wrapper aligns the dtype of input tensors with the dtype of the wrapped module's 
    parameters before passing them to the module's forward method. This is particularly useful 
    when working with mixed precision training where inputs might have different dtypes 
    than model weights.
    
    Args:
        module (nn.Module): The neural network module to wrap
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        """
        Forward pass that aligns input tensor dtypes with module parameter dtypes.
        
        This method checks the dtype of the module's parameters and converts 
        all floating point input tensors to match that dtype before forwarding
        to the wrapped module.
        
        Returns:
            Output of the wrapped module with aligned dtypes
        """
        # Find the dtype of the module's weight parameters
        weight_dtype = None
        for name, param in self.module.named_parameters():
            if 'weight' in name:
                weight_dtype = param.dtype
                break
        
        # If no weight parameter found, try to get dtype from any parameter
        if weight_dtype is None:
            try:
                weight_dtype = next(self.module.parameters()).dtype
            except StopIteration:
                # If no parameters exist, return result without modification
                return self.module(*args, **kwargs)

        def to_dtype(x):
            """
            Recursively convert tensors to the target dtype while preserving non-floating types.
            
            Args:
                x: Input tensor, tuple, list, dict, or other object
            
            Returns:
                Object with tensors converted to target dtype
            """
            if isinstance(x, torch.Tensor):
                if x.is_floating_point():
                    return x.to(weight_dtype)
                else:
                    return x
            elif isinstance(x, tuple):
                return tuple(to_dtype(item) for item in x)
            elif isinstance(x, list):
                return [to_dtype(item) for item in x]
            elif isinstance(x, dict):
                return {k: to_dtype(v) for k, v in x.items()}
            else:
                return x

        # Apply dtype conversion to all arguments and keyword arguments
        args = [to_dtype(arg) for arg in args]
        kwargs = {k: to_dtype(v) for k, v in kwargs.items()}

        return self.module(*args, **kwargs)

    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped module if not found in this wrapper.
        
        This allows accessing attributes and methods of the wrapped module transparently.
        
        Args:
            name (str): Name of the attribute to access
        
        Returns:
            Attribute from either this wrapper or the wrapped module
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)