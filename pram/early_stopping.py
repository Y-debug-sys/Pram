import numpy as np


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    
    Monitors the validation loss during training and triggers early stopping if the
    loss does not improve for a specified number of consecutive epochs (patience).
    Saves the best model state when an improvement is detected.
    
    Attributes:
        accelerator: Accelerator object for distributed training (optional)
        patience (int): Number of epochs to wait before early stopping when no improvement occurs
        verbose (bool): If True, prints messages about model saving and early stopping status
        delta (float): Minimum change in validation loss to qualify as an improvement
        counter (int): Internal counter tracking consecutive non-improvement epochs
        best_score (float): Best validation loss score observed so far
        early_stop (bool): Flag indicating whether early stopping condition has been met
        val_loss_min (float): Minimum validation loss value observed so far
    """
    
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0):
        """
        Initialize the EarlyStopping object.
        
        Args:
            accelerator: Accelerator object for distributed training (optional)
            patience (int): Number of epochs to wait before early stopping when no improvement occurs (default 7)
            verbose (bool): If True, prints messages about model saving and early stopping status (default False)
            delta (float): Minimum change in validation loss to qualify as an improvement (default 0)
        """
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        """
        Evaluate whether to stop training based on current validation loss.
        
        Compares the current validation loss with the best observed loss and updates
        the internal state accordingly. If the loss hasn't improved beyond the delta
        threshold for more than 'patience' epochs, sets the early_stop flag to True.
        
        Args:
            val_loss (float): Current validation loss value
            model: Model object whose state should be saved if improvement is detected
            path (str): Directory path where the model checkpoint should be saved
        """
        score = -val_loss
        if self.best_score is None:
            # First validation loss - initialize best score and save model
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score - self.delta * self.best_score:
            # Validation loss has not improved beyond the threshold
            self.counter += 1

            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                # Early stopping condition met
                self.early_stop = True
        else:
            # Validation loss has improved beyond the threshold
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """
        Save the model checkpoint when validation loss improves.
        
        Saves the state dictionary of trainable parameters to the specified path
        when a new best validation loss is achieved.
        
        Args:
            val_loss (float): Current validation loss value
            model: Model object whose state should be saved
            path (str): Directory path where the model checkpoint should be saved
        """
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {abs(val_loss):.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {abs(val_loss):.6f}).  Saving model ...')

        # Update the minimum validation loss
        self.val_loss_min = abs(val_loss)
        
        # Extract trainable parameters from the model
        trainable_state_dict = {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}
        
        # Save the model state dictionary using the accelerator
        self.accelerator.save(trainable_state_dict, path + '/' + 'checkpoint.pt')