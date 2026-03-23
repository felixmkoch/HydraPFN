"""
TabICL Prior module - In-Context Learning prior for tabular data.

This module provides an alternative prior implementation from TabICL that can be used
as a replacement for the traditional priors in hydrapfn.
"""

from .wrapper import TabiclPriorWrapper

# Import the main dataset class
try:
    from .dataset import PriorDataset
    
    # Create a wrapper instance for easy use in model_trainer
    tabicl_prior = TabiclPriorWrapper(PriorDataset)
except ImportError:
    tabicl_prior = None
    PriorDataset = None

__all__ = ['tabicl_prior', 'PriorDataset', 'TabiclPriorWrapper']
