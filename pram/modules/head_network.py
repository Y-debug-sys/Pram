import torch

from torch import nn
from .bf_wrapper import DtypeAlignWrapper
from torch.distributions.normal import Normal


class PolicyHead(nn.Module):
    """
    A policy head network that generates action probabilities for reinforcement learning.
    
    This network takes an embedding as input and outputs the mean and standard deviation
    of a normal distribution for each action dimension. It can operate in both training
    and testing modes, using stochastic sampling during training and deterministic
    outputs during testing.
    
    Args:
        embedding_size (int): Size of the input embedding vector
        hidden_size (int): Size of the hidden layers
        output_size (int): Size of the output policy (number of actions)
        std (float): Standard deviation for the policy. If negative, learns std via neural network
        obj (str): Objective function type ('mlu' for sigmoid activation, others for ReLU)
        log_std_min (float): Minimum value for clamping learned log standard deviation
        log_std_max (float): Maximum value for clamping learned log standard deviation
    """

    def __init__(
        self, 
        embedding_size,
        hidden_size,
        output_size,
        std=1.,
        obj='mlu',
        log_std_min=-10,
        log_std_max=5
    ):
        super(PolicyHead, self).__init__()

        # Create a projection network with two linear layers with ReLU activations
        # Uses DtypeAlignWrapper to ensure consistent data types during mixed precision training
        self.proj = nn.Sequential(
            DtypeAlignWrapper(nn.Linear(embedding_size, hidden_size)),
            nn.ReLU(),
            DtypeAlignWrapper(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU()
        )

        # Store standard deviation parameters
        self.std = std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Conditionally create a network to learn standard deviation
        # If std < 0, then std is learned via a neural network; otherwise it's fixed
        if std < 0:
            self.log_std_proj = DtypeAlignWrapper(nn.Linear(hidden_size, output_size))

        # Linear layer to project to output size for mean calculation
        self.mean_proj = DtypeAlignWrapper(nn.Linear(hidden_size, output_size))
        
        # Choose final activation function based on objective type
        if obj.upper()=='MLU':
            self.final_act = nn.Sigmoid()  # For MLU (Maximum Link Utilization) objective
        else:
            self.final_act = nn.ReLU()     # For other objectives

    def forward(self, embedding):
        """
        Forward pass to compute mean and standard deviation of the action distribution.
        
        Args:
            embedding (torch.Tensor): Input embedding tensor of shape [batch_size, embedding_size]
        
        Returns:
            tuple: A tuple containing:
                - mean (torch.Tensor): Mean of the action distribution [batch_size, output_size]
                - std (torch.Tensor): Standard deviation of the action distribution [batch_size, output_size]
        """
        # Process embedding through the projection network
        x = self.proj(embedding)
        # Compute mean of the action distribution
        mean = self.mean_proj(x)

        # Learn standard deviation via neural network if std < 0
        if self.std < 0:
            # Compute raw log standard deviation
            log_std = self.log_std_proj(x)
            # Clamp log standard deviation to prevent extreme values
            log_std_clamped = torch.clamp(
                log_std,
                min=self.log_std_min,
                max=self.log_std_max)
            # Convert to standard deviation
            std = torch.exp(log_std_clamped)
        
        # Use fixed standard deviation if std >= 0
        else:
            std = self.std

        return mean, std
    
    def evaluate(self, token_embedding, test=False):
        """
        Evaluate the policy to generate actions and their log probabilities.
        
        In training mode (test=False), samples actions from the distribution and computes
        log probabilities. In testing mode (test=True), returns deterministic actions
        based on the mean of the distribution.
        
        Args:
            token_embedding (torch.Tensor): Input embedding tensor [batch_size, embedding_size]
            test (bool): If True, use deterministic evaluation; if False, use stochastic sampling
        
        Returns:
            tuple: A tuple containing:
                - weights (torch.Tensor): Final action weights after activation [batch_size, output_size]
                - log_probability (torch.Tensor or None): Log probability of the sampled actions
                  or None if in test mode
        """
        # Get mean and std from forward pass
        mean, std = self.forward(token_embedding)

        # Test mode: deterministic evaluation
        if test:
            distribution = None
            weights = mean.detach()  # Use mean without gradients
            log_probability = None

        # Train mode: stochastic evaluation
        else:
            # Create normal distribution with computed mean and std
            distribution = Normal(mean, std)
            # Sample actions using reparameterization trick (rsample)
            sample = distribution.rsample()
            weights = sample
            # Compute log probability of the sampled actions
            log_probability = distribution.log_prob(sample).sum(axis=-1)

        # Apply final activation function (Sigmoid for MLU, ReLU for others)
        return self.final_act(weights), log_probability