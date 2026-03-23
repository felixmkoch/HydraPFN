"""
Wrapper to make tabicl_prior compatible with the old prior interface used in HydraPFN.

The old priors return: (x, y, y_) or (x, y, y_, style)
The tabicl_prior returns: (X, y, d, seq_lens, train_sizes)

This wrapper adapts tabicl_prior to match the old interface.
"""

import torch
from typing import Callable, Optional, Tuple
from hydrapfn.utils import default_device, set_locals_in_self
from hydrapfn.priors.prior import PriorDataLoader


def get_batch_to_dataloader_tabicl(prior_dataset_fn: Callable):
    """
    Converts a tabicl_prior dataset generator into a DataLoader compatible with HydraPFN.
    
    Args:
        prior_dataset_fn: A callable that returns a PriorDataset instance from tabicl_prior
        
    Returns:
        A DataLoader class compatible with old prior interface
    """
    class TabiclPriorDataLoader(PriorDataLoader):
        """DataLoader wrapper for tabicl_prior datasets."""
        
        def __init__(self, num_steps, **kwargs):
            """
            Initialize the tabicl prior dataloader.
            
            Args:
                num_steps: Number of batches per epoch
                **kwargs: Additional arguments passed to prior_dataset_fn and get_batch
                    - num_features: Number of features (passed to prior)
                    - hyperparameters: Should contain 'batch_size' and other prior configs
                    - device: Device to use
                    - seq_len_maximum: Maximum sequence length
                    - eval_pos_seq_len_sampler: Sampler for eval position and seq length
                    - Other prior-specific kwargs
            """
            set_locals_in_self(locals())
            self.num_features = kwargs.get('num_features', 100)
            self.epoch_count = 0
            
            # Extract prior configuration
            self.hyperparameters = kwargs.get('hyperparameters', {})
            self.device = kwargs.get('device', default_device)
            self.seq_len_maximum = kwargs.get('seq_len_maximum', 512)
            
            # Create the prior dataset instance
            self.prior_dataset = prior_dataset_fn(
                batch_size=self.hyperparameters.get('batch_size', 32),
                max_features=self.num_features,
                max_seq_len=self.seq_len_maximum,
                device=self.device
            )
        
        def gbm(self, *args, eval_pos_seq_len_sampler, **kwargs):
            """Get batch method - matching old prior interface."""
            single_eval_pos, seq_len = eval_pos_seq_len_sampler()
            
            # Get batch from tabicl_prior (returns X, y, d, seq_lens, train_sizes)
            X, y, d, seq_lens, train_sizes = self.prior_dataset.get_batch()
            
            # Convert to old interface format: (x, y, y_)
            # X shape: (batch_size, seq_len, num_features) or NestedTensor
            # y shape: (batch_size, seq_len)
            # We use y as both target and prediction target for compatibility
            
            # Transpose to match old format: (T, B, H) where T is sequence length
            if hasattr(X, 'unbind'):  # NestedTensor case
                x = X
            else:
                x = X.transpose(0, 1)  # (batch_size, seq_len, num_features) -> (seq_len, batch_size, num_features)
            
            # y is already (batch_size, seq_len), transpose to (seq_len, batch_size)
            y_out = y.transpose(0, 1) if y is not None else None
            
            # Use y as target_y (y_) for old interface
            target_y = y_out
            
            return (None, x, y_out), target_y, single_eval_pos
        
        def __len__(self):
            return self.num_steps
        
        def get_test_batch(self):
            """Get a test batch without incrementing epoch count."""
            return self.gbm(**self.get_batch_kwargs, epoch=self.epoch_count, model=self.model if hasattr(self, 'model') else None)
        
        def __iter__(self):
            """Iterate through batches for one epoch."""
            assert hasattr(self, 'model'), "Please assign model with `dl.model = ...` before training."
            self.epoch_count += 1
            return iter(
                self.gbm(**self.get_batch_kwargs, epoch=self.epoch_count - 1, model=self.model) 
                for _ in range(self.num_steps)
            )
    
    return TabiclPriorDataLoader


class TabiclPriorWrapper:
    """
    Wrapper class to make tabicl_prior compatible with HydraPFN's prior interface.
    """
    
    def __init__(self, prior_dataset_class):
        """
        Initialize the wrapper with a tabicl prior dataset class.
        
        Args:
            prior_dataset_class: The PriorDataset class from tabicl_prior
        """
        self.prior_dataset_class = prior_dataset_class
        self.DataLoader = get_batch_to_dataloader_tabicl(prior_dataset_class)
        self.num_outputs = 1  # For compatibility with old priors
