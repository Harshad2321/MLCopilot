"""
MLCopilot Root Cause Analysis
Infers root causes of detected training failures using expert rules.
"""

from typing import Dict, List, Any

from .types import (
    DetectionResult, Diagnosis, AnomalyType, CauseCategory,
    DetectionThresholds
)


class RootCauseAnalyzer:
    """
    Analyzes detected anomalies and infers root causes using rule-based reasoning.
    """
    
    def analyze(self, 
                detection: DetectionResult,
                model_info: Dict[str, Any],
                optimizer_info: Dict[str, Any]) -> Diagnosis:
        """
        Perform root cause analysis on a detected anomaly.
        
        Args:
            detection: DetectionResult from failure detector
            model_info: Model architecture metadata
            optimizer_info: Optimizer configuration
        
        Returns:
            Diagnosis with inferred root cause
        """
        # Route to appropriate analysis function
        analyzer_map = {
            AnomalyType.EXPLODING_GRADIENTS: self._analyze_exploding_gradients,
            AnomalyType.VANISHING_GRADIENTS: self._analyze_vanishing_gradients,
            AnomalyType.LOSS_DIVERGENCE: self._analyze_loss_divergence,
            AnomalyType.LOSS_PLATEAU: self._analyze_loss_plateau,
            AnomalyType.NAN_LOSS: self._analyze_nan_loss,
            AnomalyType.OVERFITTING: self._analyze_overfitting,
            AnomalyType.LR_INSTABILITY: self._analyze_lr_instability,
        }
        
        analyzer = analyzer_map.get(detection.anomaly_type)
        
        if analyzer:
            return analyzer(detection, model_info, optimizer_info)
        else:
            # Fallback for unknown anomaly types
            return self._analyze_unknown(detection, model_info, optimizer_info)
    
    def _analyze_exploding_gradients(self,
                                     detection: DetectionResult,
                                     model_info: Dict[str, Any],
                                     optimizer_info: Dict[str, Any]) -> Diagnosis:
        """Analyze exploding gradients using trigger flags from raw_values."""
        lr = optimizer_info.get('learning_rate', 0.0)
        has_normalization = model_info.get('has_normalization', False)
        num_layers = model_info.get('num_layers', 0)
        
        contributing_factors = []
        
        # Get thresholds from raw_values or use defaults
        high_lr_threshold = detection.raw_values.get('high_lr_threshold', DetectionThresholds.HIGH_LR_THRESHOLD)
        
        # Get trigger flags from raw_values
        absolute_threshold_triggered = detection.raw_values.get('absolute_threshold_triggered', 0.0) == 1.0
        relative_threshold_triggered = detection.raw_values.get('relative_threshold_triggered', 0.0) == 1.0
        log_growth_triggered = detection.raw_values.get('log_growth_triggered', 0.0) == 1.0
        regression_triggered = detection.raw_values.get('regression_triggered', 0.0) == 1.0
        
        # Priority 1: Threshold-driven triggers (absolute or relative snapshot-based)
        if absolute_threshold_triggered or relative_threshold_triggered:
            cause_category = CauseCategory.OPTIMIZATION
            primary_cause = "Gradient norm spike detected"
            
            grad_norm = detection.metric_snapshot.grad_norm
            threshold = detection.raw_values.get('threshold', 0.0)
            
            if absolute_threshold_triggered:
                contributing_factors.append(
                    f"Gradient norm ({grad_norm:.2f}) exceeds absolute threshold ({threshold:.2f})"
                )
            
            if relative_threshold_triggered:
                moving_avg = detection.raw_values.get('moving_average', 0.0)
                relative_threshold = detection.raw_values.get('relative_threshold', 0.0)
                contributing_factors.append(
                    f"Gradient norm spike ({grad_norm:.2f}) vs moving average ({moving_avg:.2f})"
                )
            
            reasoning = (
                "The gradient norm has spiked above its typical behavior, indicating sudden loss landscape "
                "changes or data anomalies. This can occur with outliers in the data or sharp loss surfaces."
            )
        
        # Priority 2: Trend triggers (sustained gradient growth)
        elif log_growth_triggered or regression_triggered:
            cause_category = CauseCategory.OPTIMIZATION
            primary_cause = "Sustained gradient growth trend detected"
            
            if log_growth_triggered:
                avg_log_growth = detection.raw_values.get('average_log_growth', 0.0)
                contributing_factors.append(
                    f"Gradient norm growing exponentially (log-growth rate: {avg_log_growth:.4f})"
                )
            
            if regression_triggered:
                reg_slope = detection.raw_values.get('regression_slope', 0.0)
                reg_r_squared = detection.raw_values.get('regression_r_squared', 0.0)
                contributing_factors.append(
                    f"Strong upward trend in gradient norm (slope: {reg_slope:.4f}, R²: {reg_r_squared:.2f})"
                )
            
            reasoning = (
                "Gradients are exhibiting sustained growth over time, indicating an exponential divergence "
                "pattern. This suggests the optimization dynamics are inherently unstable, possibly amplified "
                "by the learning rate or network initialization."
            )
        
        # Priority 3: High learning rate
        elif lr > high_lr_threshold:
            cause_category = CauseCategory.HYPERPARAMETER
            primary_cause = "Learning rate too high"
            contributing_factors.append(f"Learning rate ({lr}) exceeds recommended threshold ({high_lr_threshold})")
            reasoning = (
                f"The learning rate ({lr}) is very high, causing weight updates to overshoot. "
                "Large learning rates can cause gradients to explode as the optimizer takes "
                "steps that are too large in parameter space."
            )
        
        # Priority 4: Missing normalization
        elif not has_normalization:
            cause_category = CauseCategory.MODEL_ARCHITECTURE
            primary_cause = "Missing normalization layers"
            contributing_factors.append("No BatchNorm or LayerNorm detected in model")
            reasoning = (
                "The model lacks normalization layers (BatchNorm/LayerNorm). "
                "Without normalization, activations can grow unbounded through the network, "
                "leading to exploding gradients during backpropagation."
            )
        
        # Priority 5: Very deep network
        elif num_layers > 10:
            cause_category = CauseCategory.MODEL_ARCHITECTURE
            primary_cause = "Very deep network without proper architecture"
            contributing_factors.append(f"Model has {num_layers} layers")
            reasoning = (
                f"The model is deep ({num_layers} layers) which can cause gradient instability. "
                "Consider using residual connections (ResNet-style) or proper initialization."
            )
        
        # Priority 6: Fallback
        else:
            cause_category = CauseCategory.OPTIMIZATION
            primary_cause = "Optimization instability"
            contributing_factors.append("Gradient clipping may be needed")
            reasoning = (
                "Gradients are exploding during backpropagation. "
                "This could be due to unstable optimization dynamics or numerical issues."
            )
        
        # Additional context
        grad_norm = detection.metric_snapshot.grad_norm
        contributing_factors.append(f"Gradient norm reached {grad_norm:.2f}")
        
        return Diagnosis(
            detection=detection,
            cause_category=cause_category,
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            reasoning=reasoning,
            model_context=model_info,
            optimizer_context=optimizer_info
        )
    
    def _analyze_vanishing_gradients(self,
                                     detection: DetectionResult,
                                     model_info: Dict[str, Any],
                                     optimizer_info: Dict[str, Any]) -> Diagnosis:
        """Analyze vanishing gradients."""
        lr = optimizer_info.get('learning_rate', 0.0)
        num_layers = model_info.get('num_layers', 0)
        has_saturating_activations = model_info.get('has_saturating_activations', False)
        
        contributing_factors = []
        
        if num_layers > 10:
            cause_category = CauseCategory.MODEL_ARCHITECTURE
            primary_cause = "Very deep network causing gradient decay"
            contributing_factors.append(f"Model has {num_layers} layers")
            if has_saturating_activations:
                contributing_factors.append("Model uses saturating activations (sigmoid/tanh) which are prone to vanishing gradients")
            reasoning = (
                f"The model is very deep ({num_layers} layers). "
                "In deep networks, gradients can diminish exponentially as they propagate "
                "backwards through layers, especially with saturating activation functions like sigmoid/tanh."
            )
        elif lr < DetectionThresholds.LOW_LR_THRESHOLD:
            cause_category = CauseCategory.HYPERPARAMETER
            primary_cause = "Learning rate too small"
            contributing_factors.append(f"Learning rate ({lr:.2e}) is extremely small")
            reasoning = (
                f"The learning rate ({lr:.2e}) is extremely small, resulting in nearly zero gradients. "
                "While this won't prevent learning entirely, it will make training extremely slow."
            )
        else:
            cause_category = CauseCategory.MODEL_ARCHITECTURE
            primary_cause = "Architecture causing gradient flow issues"
            contributing_factors.append("Possible saturating activations or initialization issues")
            if has_saturating_activations:
                contributing_factors.append("Model uses saturating activations (sigmoid/tanh) which are prone to vanishing gradients")
            reasoning = (
                "Gradients are vanishing during backpropagation. "
                "This often occurs with sigmoid/tanh activations, poor weight initialization, "
                "or missing residual connections in deep networks."
            )
        
        grad_norm = detection.metric_snapshot.grad_norm
        contributing_factors.append(f"Gradient norm dropped to {grad_norm:.2e}")
        
        return Diagnosis(
            detection=detection,
            cause_category=cause_category,
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            reasoning=reasoning,
            model_context=model_info,
            optimizer_context=optimizer_info
        )
    
    def _analyze_loss_divergence(self,
                                 detection: DetectionResult,
                                 model_info: Dict[str, Any],
                                 optimizer_info: Dict[str, Any]) -> Diagnosis:
        """Analyze loss divergence using trigger flags from raw_values."""
        lr = optimizer_info.get('learning_rate', 0.0)
        
        contributing_factors = []
        
        # Get thresholds from raw_values or use defaults
        high_lr_threshold = detection.raw_values.get('high_lr_threshold', DetectionThresholds.HIGH_LR_THRESHOLD)
        
        # Get trigger flags from raw_values
        old_logic_triggered = detection.raw_values.get('old_logic_triggered', 0.0) == 1.0
        regression_triggered = detection.raw_values.get('regression_triggered', 0.0) == 1.0
        
        # Priority 1: Regression-based sustained upward loss trend
        if regression_triggered:
            cause_category = CauseCategory.OPTIMIZATION
            primary_cause = "Sustained upward loss trend"
            
            relative_slope = detection.raw_values.get('relative_slope', 0.0)
            reg_r_squared = detection.raw_values.get('regression_r_squared', 0.0)
            
            contributing_factors.append(
                f"Loss growing at {relative_slope*100:.1f}% per batch (R²: {reg_r_squared:.2f})"
            )
            contributing_factors.append(
                f"Strong sustained divergence pattern detected"
            )
            
            # Add high LR as amplifying factor if present
            if lr > high_lr_threshold:
                contributing_factors.append(
                    f"High learning rate ({lr}) may amplify the divergence"
                )
            
            reasoning = (
                f"The loss exhibits a strong upward trend with high regression quality (R²={reg_r_squared:.2f}). "
                "This indicates systematic divergence rather than noise, suggesting fundamental issues with "
                "optimization dynamics, learning rate, or data quality."
            )
        
        # Priority 2: Old logic trigger (ratio-based divergence)
        elif old_logic_triggered:
            current_loss = detection.metric_snapshot.loss
            initial_loss = detection.raw_values.get('initial_loss', 0)
            ratio = detection.raw_values.get('ratio', 1.0)
            divergence_multiplier = 2.0
            
            # Determine if high LR is the likely cause or if it's data/initialization
            if lr > high_lr_threshold:
                cause_category = CauseCategory.HYPERPARAMETER
                primary_cause = "Learning rate too high causing instability"
                contributing_factors.append(f"Learning rate ({lr}) exceeds threshold ({high_lr_threshold})")
            else:
                cause_category = CauseCategory.OPTIMIZATION
                primary_cause = "Loss divergence from initialization issue"
                contributing_factors.append("Model initialization or data issues causing divergence")
            
            contributing_factors.append(
                f"Loss increased from {initial_loss:.4f} to {current_loss:.4f} (ratio: {ratio:.2f}x initial)"
            )
            
            reasoning = (
                f"The loss has diverged significantly, exceeding {divergence_multiplier}x the initial loss. "
                "This indicates the optimization is moving away from a valid solution, likely due to "
                "learning rate instability or data quality issues."
            )
        
        # Priority 3: Fallback (no trigger metadata, use LR as heuristic)
        elif lr > high_lr_threshold:
            cause_category = CauseCategory.HYPERPARAMETER
            primary_cause = "Learning rate too high causing instability"
            contributing_factors.append(f"Learning rate ({lr}) is very high (threshold: {high_lr_threshold})")
            reasoning = (
                f"The learning rate ({lr}) is too high for stable training. "
                "High learning rates cause the optimizer to overshoot minima, "
                "leading to diverging loss values."
            )
        
        # Priority 4: Final fallback
        else:
            cause_category = CauseCategory.OPTIMIZATION
            primary_cause = "Training instability"
            contributing_factors.append("Loss diverged from initial value")
            reasoning = (
                "The loss is diverging instead of decreasing. "
                "This indicates fundamental training instability, possibly due to "
                "learning rate, batch size, or data issues."
            )
        
        return Diagnosis(
            detection=detection,
            cause_category=cause_category,
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            reasoning=reasoning,
            model_context=model_info,
            optimizer_context=optimizer_info
        )
    
    def _analyze_loss_plateau(self,
                             detection: DetectionResult,
                             model_info: Dict[str, Any],
                             optimizer_info: Dict[str, Any]) -> Diagnosis:
        """Analyze loss plateau, distinguishing near-zero (converged) vs high-loss (stuck)."""
        lr = optimizer_info.get('learning_rate', 0.0)
        
        # Get plateau metrics from raw_values
        mean_loss = detection.raw_values.get('mean_loss', 0.0)
        cv = detection.raw_values.get('cv', 0.0)
        window_size = detection.raw_values.get('window_size', 0)
        
        # Determine plateau state: converged (near-zero) vs stuck (high-loss)
        # Use same thresholds as detector: initial_loss * 0.1 or absolute 0.01
        initial_loss = detection.raw_values.get('initial_loss', None)
        convergence_threshold_relative = (initial_loss * 0.1) if initial_loss is not None else float('inf')
        convergence_threshold_absolute = 0.01
        
        is_low_loss = mean_loss < convergence_threshold_relative or mean_loss < convergence_threshold_absolute
        
        contributing_factors = []
        
        # Priority 1: Distinguish by plateau loss state, not by LR
        if is_low_loss:
            # Near-zero loss plateau = model may have converged
            # But here it's already filtered out by detector, so this is a fallback path
            cause_category = CauseCategory.OPTIMIZATION
            primary_cause = "Loss converged at low value"
            contributing_factors.append(
                f"Loss plateaued at low value ({mean_loss:.4f}, CV: {cv:.5f})"
            )
            if lr < 1e-5:
                contributing_factors.append(f"Learning rate ({lr:.2e}) is very small")
            else:
                contributing_factors.append(f"Current learning rate: {lr:.6f}")
            
            reasoning = (
                f"The loss has plateaued at a low value ({mean_loss:.4f}) with low variance (CV: {cv:.5f}). "
                "This suggests the model may be approaching convergence. "
                f"Over {window_size} batches, loss has remained stable. "
                "Consider using learning rate scheduling to refine the solution further."
            )
        else:
            # High-loss plateau = model stuck
            cause_category = CauseCategory.OPTIMIZATION
            primary_cause = "Stuck in local minimum or plateau"
            contributing_factors.append(
                f"Loss plateaued at high value ({mean_loss:.4f}, CV: {cv:.5f})"
            )
            
            # Use LR as secondary context (not primary branch)
            if lr < 1e-5:
                contributing_factors.append(
                    f"Learning rate ({lr:.2e}) is very small—may hinder escaping plateau"
                )
            else:
                contributing_factors.append(
                    f"Current learning rate: {lr:.6f}"
                )
            
            contributing_factors.append(f"Loss unchanged over {window_size} batches")
            
            reasoning = (
                f"The loss has plateaued at a high value ({mean_loss:.4f}) over {window_size} batches, "
                f"with low variance (CV: {cv:.5f}). The model appears stuck in a local minimum or flat region. "
                "Consider learning rate scheduling, momentum adjustments, architectural changes, "
                "or data augmentation to escape this plateau."
            )
        
        return Diagnosis(
            detection=detection,
            cause_category=cause_category,
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            reasoning=reasoning,
            model_context=model_info,
            optimizer_context=optimizer_info
        )
    
    def _analyze_nan_loss(self,
                         detection: DetectionResult,
                         model_info: Dict[str, Any],
                         optimizer_info: Dict[str, Any]) -> Diagnosis:
        """Analyze NaN loss."""
        lr = optimizer_info.get('learning_rate', 0.0)
        
        cause_category = CauseCategory.NUMERICAL_INSTABILITY
        primary_cause = "Numerical instability causing NaN"
        
        contributing_factors = [
            "Loss became NaN or Inf",
            "Training has completely diverged"
        ]
        
        if lr > DetectionThresholds.HIGH_LR_THRESHOLD:
            contributing_factors.append(f"Very high learning rate ({lr}) likely contributed")
        
        reasoning = (
            "The loss became NaN (Not a Number) or Inf, indicating severe numerical instability. "
            "This typically results from exploding gradients, division by zero, or log of negative values. "
            "Common causes: extremely high learning rate, missing gradient clipping, or data preprocessing issues."
        )
        
        return Diagnosis(
            detection=detection,
            cause_category=cause_category,
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            reasoning=reasoning,
            model_context=model_info,
            optimizer_context=optimizer_info
        )
    
    def _analyze_overfitting(self,
                            detection: DetectionResult,
                            model_info: Dict[str, Any],
                            optimizer_info: Dict[str, Any]) -> Diagnosis:
        """Analyze overfitting using both gap and ratio from raw_values."""
        total_params = model_info.get('total_params', 0)
        
        cause_category = CauseCategory.MODEL_ARCHITECTURE
        primary_cause = "Model overfitting to training data"
        
        contributing_factors = [
            "Large gap between training and validation loss",
        ]
        
        if total_params > 1_000_000:
            contributing_factors.append(f"Large model ({total_params:,} parameters)")
        
        train_loss = detection.raw_values.get('train_loss', 0)
        val_loss = detection.raw_values.get('val_loss', 0)
        gap = detection.raw_values.get('gap', 0)
        ratio = detection.raw_values.get('ratio', 1.0)
        ratio_threshold = detection.raw_values.get('ratio_threshold', 1.5)
        
        # Add both ratio and gap information
        contributing_factors.append(
            f"Val/train loss ratio: {ratio:.2f}x (threshold: {ratio_threshold:.2f}x)"
        )
        contributing_factors.append(
            f"Absolute gap: {gap:.4f}"
        )
        
        reasoning = (
            f"Validation loss ({val_loss:.4f}) significantly exceeds training loss ({train_loss:.4f}), "
            f"with a val/train ratio of {ratio:.2f}x (gap: {gap:.4f}). "
            "This indicates the model is memorizing the training data rather than learning "
            "generalizable patterns. Consider regularization techniques, data augmentation, "
            "or reducing model capacity."
        )
        
        return Diagnosis(
            detection=detection,
            cause_category=cause_category,
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            reasoning=reasoning,
            model_context=model_info,
            optimizer_context=optimizer_info
        )
    
    def _analyze_lr_instability(self,
                               detection: DetectionResult,
                               model_info: Dict[str, Any],
                               optimizer_info: Dict[str, Any]) -> Diagnosis:
        """Analyze learning rate instability."""
        lr = optimizer_info.get('learning_rate', 0.0)
        
        cause_category = CauseCategory.HYPERPARAMETER
        primary_cause = "Learning rate schedule causing instability"
        
        contributing_factors = [
            f"Current learning rate: {lr:.6f}",
            "Rapid learning rate changes detected in optimization trajectory"
        ]
        
        reasoning = (
            "The optimization is showing signs of learning rate instability, indicating that "
            "rapid changes in the learning rate schedule are destabilizing training. "
            f"Current LR of {lr:.6f} combined with oscillating updates suggests the scheduler "
            "or adaptive learning rate mechanism may be too aggressive. "
            "Consider smoothing the learning rate schedule or using a more conservative adaptation strategy."
        )
        
        return Diagnosis(
            detection=detection,
            cause_category=cause_category,
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            reasoning=reasoning,
            model_context=model_info,
            optimizer_context=optimizer_info
        )
    
    def _analyze_unknown(self,
                        detection: DetectionResult,
                        model_info: Dict[str, Any],
                        optimizer_info: Dict[str, Any]) -> Diagnosis:
        """Fallback analysis for unknown anomaly types."""
        return Diagnosis(
            detection=detection,
            cause_category=CauseCategory.UNKNOWN,
            primary_cause="Unknown training issue",
            contributing_factors=["Unable to determine specific cause"],
            reasoning="An anomaly was detected but root cause analysis is not available for this type.",
            model_context=model_info,
            optimizer_context=optimizer_info
        )
