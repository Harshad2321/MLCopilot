"""
MLCopilot Main Entry Point
Provides convenient wrapper functions for integrating monitoring into training loops.
"""

import logging
from typing import Optional
import torch.nn as nn
import torch.optim as optim

from mlcopilot import (
    TrainingMonitor,
    FailureDetector,
    RootCauseAnalyzer,
    RecommendationEngine,
    CLIReporter
)

logger = logging.getLogger(__name__)


class MLCopilot:
    """
    High-level interface for ML training monitoring and failure detection.
    
    Usage:
        copilot = MLCopilot(model, optimizer)
        copilot.start()
        
        # In training loop:
        for batch in dataloader:
            loss = train_step(batch)
            if copilot.log_and_check(loss):
                break  # Issue detected, stop training
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 optimizer: optim.Optimizer,
                 auto_report: bool = True,
                 use_colors: bool = True):
        """
        Initialize MLCopilot.
        
        Args:
            model: PyTorch model to monitor
            optimizer: PyTorch optimizer to monitor
            auto_report: Automatically print reports when issues detected
            use_colors: Use colors in CLI output
        """
        self.monitor = TrainingMonitor(model, optimizer)
        self.detector = FailureDetector()
        self.analyzer = RootCauseAnalyzer()
        self.recommender = RecommendationEngine()
        self.reporter = CLIReporter(use_colors=use_colors)
        
        self.auto_report = auto_report
        self.issue_detected = False
        
    def start(self):
        """Start monitoring the training process."""
        self.monitor.attach()
        
        # Display monitoring info
        model_info = self.monitor.get_model_info()
        optimizer_info = self.monitor.get_optimizer_info()
        
        if self.auto_report:
            self.reporter.report_monitoring_start(model_info, optimizer_info)
    
    def log_batch(self, loss: float, val_loss: Optional[float] = None):
        """
        Log metrics for a single training batch.
        
        Args:
            loss: Training loss for current batch
            val_loss: Optional validation loss
        """
        self.monitor.log_batch(loss, val_loss)
    
    def log_epoch_end(self):
        """Mark the end of an epoch."""
        self.monitor.log_epoch_end()
    
    def check_health(self) -> bool:
        """
        Check for training failures.
        
        Returns:
            True if issue detected, False otherwise
        """
        if not self.monitor.should_check():
            return False
        
        # Get metrics and detect anomalies
        metrics = self.monitor.get_metrics()
        detections = self.detector.detect_all(metrics)
        
        if detections:
            # Issue detected - analyze and recommend
            detection = detections[0]  # Focus on highest priority issue
            
            model_info = self.monitor.get_model_info()
            optimizer_info = self.monitor.get_optimizer_info()
            
            # Try to analyze the detection
            diagnosis = None
            try:
                diagnosis = self.analyzer.analyze(detection, model_info, optimizer_info)
            except Exception as e:
                logger.error(
                    f"RootCauseAnalyzer.analyze failed: {e}",
                    exc_info=True
                )
            
            # Try to generate recommendations
            recommendations = []
            if diagnosis is not None:
                try:
                    recommendations = self.recommender.generate(diagnosis)
                except Exception as e:
                    logger.error(
                        f"RecommendationEngine.generate failed: {e}",
                        exc_info=True
                    )
            
            # Report if auto-reporting enabled and we have both diagnosis and recommendations
            if self.auto_report:
                if diagnosis is not None and recommendations:
                    self.reporter.report_full(detection, diagnosis, recommendations)
                elif diagnosis is not None:
                    # Report diagnosis without recommendations
                    self.reporter.report_full(detection, diagnosis, [])
                else:
                    # Report detection only (no diagnosis)
                    logger.warning(
                        f"Detection found but analysis failed. Issue type: {detection.anomaly_type}, "
                        f"Confidence: {detection.confidence}"
                    )
            
            self.issue_detected = True
            return True
        
        return False
    
    def log_and_check(self, loss: float, val_loss: Optional[float] = None) -> bool:
        """
        Convenience method: log batch and check for issues.
        
        Args:
            loss: Training loss for current batch
            val_loss: Optional validation loss
        
        Returns:
            True if issue detected, False otherwise
        """
        self.log_batch(loss, val_loss)
        return self.check_health()
    
    def stop(self):
        """Stop monitoring and cleanup."""
        self.monitor.detach()
        
        if self.auto_report and not self.issue_detected:
            self.reporter.report_no_issues()
    
    def get_metrics(self):
        """Get all collected metrics."""
        return self.monitor.get_metrics()
    
    def reset(self):
        """Reset all collected metrics."""
        self.monitor.reset()
        self.issue_detected = False


def main():
    """
    Main entry point for standalone usage.
    This is primarily for demonstration - actual usage should integrate into training loops.
    """
    print("MLCopilot - Real-Time ML Training Failure Detection")
    print("=" * 60)
    print("\nThis is a library meant to be integrated into training loops.")
    print("See examples/failing_training.py for usage demonstration.")
    print("\nQuick start:")
    print("  from mlcopilot import MLCopilot")
    print("  copilot = MLCopilot(model, optimizer)")
    print("  copilot.start()")
    print("  ")
    print("  # In training loop:")
    print("  for batch in dataloader:")
    print("      loss = train_step(batch)")
    print("      if copilot.log_and_check(loss):")
    print("          break  # Issue detected")
    print("  ")
    print("  copilot.stop()")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
