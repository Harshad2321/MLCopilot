"""
MLCopilot Training Monitor
Captures metrics from PyTorch training loops using hooks.
"""

import logging
import time
import collections
from typing import List, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn

from .types import MetricSnapshot, MonitoringConfig

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """
    Non-invasive training monitor that attaches to PyTorch models and optimizers.
    Collects metrics via hooks and manual logging.
    """
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """
        Initialize the training monitor.
        
        Args:
            model: PyTorch model to monitor
            optimizer: PyTorch optimizer to monitor
        """
        self.model = model
        self.optimizer = optimizer
        
        # Metrics storage
        self.metrics_buffer: collections.deque = collections.deque(maxlen=MonitoringConfig.MAX_METRICS_BUFFER)
        self.current_epoch = 0
        self.current_batch = 0
        
        # Gradient tracking
        self.grad_norms: List[float] = []
        self.layer_grad_norms: Dict[str, float] = {}
        
        # Hook handles for cleanup
        self.hook_handles = []
        
        # Initial loss for tracking divergence
        self.initial_loss: Optional[float] = None
        
        # Batch counter for check intervals
        self.batch_counter = 0
        
    def attach(self):
        """Attach hooks to the model for gradient monitoring."""
        # Register backward hooks for per-parameter gradient norm capture
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Use default argument pattern to capture name correctly in closure
                def hook(grad, _name=name):
                    # Store per-parameter gradient norm during backward pass
                    self.layer_grad_norms[_name] = grad.norm(2).item()
                
                handle = param.register_hook(hook)
                self.hook_handles.append(handle)
    
    def detach(self):
        """Remove all hooks from the model."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
    
    def get_layer_grad_norms(self) -> Dict[str, float]:
        """Get per-layer gradient norms from the most recent backward pass."""
        return dict(self.layer_grad_norms)
    
    def log_batch(self, loss: float, val_loss: Optional[float] = None):
        """
        Log metrics for a single training batch.
        
        Args:
            loss: Training loss for current batch
            val_loss: Optional validation loss
        """
        # Set initial loss on first batch
        if self.initial_loss is None:
            self.initial_loss = loss
        
        # Calculate gradient norm
        grad_norm = self._calculate_grad_norm()
        
        # Get learning rate
        lr = self._get_learning_rate()
        
        # Get parameter statistics
        param_stats = self._calculate_param_stats()
        
        # Create metric snapshot
        snapshot = MetricSnapshot(
            epoch=self.current_epoch,
            batch=self.current_batch,
            loss=loss,
            grad_norm=grad_norm,
            learning_rate=lr,
            param_mean=param_stats['mean'],
            param_std=param_stats['std'],
            param_max=param_stats['max'],
            timestamp=time.time(),
            val_loss=val_loss
        )
        
        # Add to buffer (deque with maxlen handles overflow automatically)
        self.metrics_buffer.append(snapshot)
        
        self.current_batch += 1
        self.batch_counter += 1
    
    def log_epoch_end(self):
        """Mark the end of an epoch."""
        self.current_epoch += 1
        self.current_batch = 0
    
    def should_check(self) -> bool:
        """
        Determine if we should run anomaly detection.
        
        Returns:
            True if enough batches have passed since last check
        """
        # Don't check during warmup
        if len(self.metrics_buffer) < MonitoringConfig.WARMUP_BATCHES:
            return False
        
        # Check every N batches
        if self.batch_counter >= MonitoringConfig.CHECK_INTERVAL_BATCHES:
            self.batch_counter = 0
            return True
        
        return False
    
    def get_metrics(self) -> List[MetricSnapshot]:
        """Get all collected metrics."""
        return list(self.metrics_buffer)
    
    def get_recent_metrics(self, window: int = 20) -> List[MetricSnapshot]:
        """
        Get recent metrics within a window.
        
        Args:
            window: Number of recent batches to return
        
        Returns:
            List of recent metric snapshots
        """
        return self.metrics_buffer[-window:] if len(self.metrics_buffer) >= window else list(self.metrics_buffer)
    
    def get_moving_average_loss(self, window: int = 20) -> float:
        """Calculate moving average of loss."""
        recent = self.get_recent_metrics(window)
        if not recent:
            return 0.0
        return np.mean([m.loss for m in recent])
    
    def get_moving_average_grad_norm(self, window: int = 20) -> float:
        """Calculate moving average of gradient norm."""
        recent = self.get_recent_metrics(window)
        if not recent:
            return 0.0
        return np.mean([m.grad_norm for m in recent])
    
    def _calculate_grad_norm(self) -> float:
        """
        Calculate L2 norm of all gradients.
        
        Formula: ||∇||_2 = sqrt(sum(||∇_i||_2^2))
        
        Returns:
            L2 norm of gradients across all parameters
        """
        import math
        
        total_norm_sq = 0.0
        any_grad_found = False
        
        for param in self.model.parameters():
            if param.grad is not None:
                any_grad_found = True
                # Keep computation in tensor space for numerical stability
                # Compute norm, then convert to float, then square
                param_norm = param.grad.data.norm(2).item()
                total_norm_sq += param_norm ** 2
        
        # If no gradients were found, emit debug message
        if not any_grad_found:
            logger.debug(
                "No gradients found during grad_norm calculation; likely zero_grad() was called. "
                "Returning 0.0. This may indicate the model hasn't accumulated gradients yet."
            )
            return 0.0
        
        # Handle edge case of all zero gradients
        if total_norm_sq == 0.0:
            return 0.0
        
        return math.sqrt(total_norm_sq)
    
    def _get_learning_rate(self) -> float:
        """
        Extract current learning rate from optimizer.
        
        Returns:
            Current learning rate
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0
    
    def _calculate_param_stats(self) -> Dict[str, float]:
        """
        Calculate statistics across all model parameters.
        
        Returns:
            Dictionary with mean, std, max of parameters
        """
        all_params = []
        for param in self.model.parameters():
            try:
                all_params.append(param.data.cpu().numpy().flatten())
            except (RuntimeError, TypeError) as e:
                logger.debug(
                    f"Failed to convert parameter to numpy (likely non-copyable CUDA tensor). "
                    f"Skipping this parameter: {e}"
                )
                continue
        
        if not all_params:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0}
        
        all_params = np.concatenate(all_params)
        
        return {
            'mean': float(np.mean(all_params)),
            'std': float(np.std(all_params)),
            'max': float(np.max(np.abs(all_params)))
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Extract model architecture information.
        
        Returns:
            Dictionary with model metadata
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Check for normalization layers
        has_batchnorm = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) 
                           for m in self.model.modules())
        has_layernorm = any(isinstance(m, nn.LayerNorm) for m in self.model.modules())
        
        # Check for saturating activations
        has_sigmoid = any(isinstance(m, nn.Sigmoid) for m in self.model.modules())
        has_tanh = any(isinstance(m, nn.Tanh) for m in self.model.modules())
        
        # Count only leaf modules (layers with no children)
        num_layers = sum(1 for m in self.model.modules() if not list(m.children()))
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'has_batchnorm': has_batchnorm,
            'has_layernorm': has_layernorm,
            'has_normalization': has_batchnorm or has_layernorm,
            'has_sigmoid': has_sigmoid,
            'has_tanh': has_tanh,
            'has_saturating_activations': has_sigmoid or has_tanh,
            'num_layers': num_layers,
            'model_type': type(self.model).__name__
        }
    
    def get_optimizer_info(self) -> Dict[str, Any]:
        """
        Extract optimizer configuration.
        
        Returns:
            Dictionary with optimizer metadata
        """
        optimizer_type = type(self.optimizer).__name__
        
        # Extract key hyperparameters
        config = {
            'optimizer_type': optimizer_type,
            'learning_rate': self._get_learning_rate()
        }
        
        # Extract additional params from first param group
        if self.optimizer.param_groups:
            param_group = self.optimizer.param_groups[0]
            for key in ['momentum', 'weight_decay', 'betas', 'eps']:
                if key in param_group:
                    config[key] = param_group[key]
        
        return config
    
    def reset(self):
        """Reset all collected metrics."""
        self.metrics_buffer.clear()
        self.layer_grad_norms.clear()
        self.current_epoch = 0
        self.current_batch = 0
        self.initial_loss = None
        self.batch_counter = 0
