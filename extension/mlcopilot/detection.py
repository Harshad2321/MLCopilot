"""
MLCopilot Failure Detection
Rule-based anomaly detection for training failures.
"""

import logging
import math
from typing import List, Optional, Tuple
import numpy as np

from .types import (
    MetricSnapshot, DetectionResult, AnomalyType, Severity,
    DetectionThresholds, calculate_confidence, determine_severity
)

logger = logging.getLogger(__name__)


class FailureDetector:
    """
    Orchestrates all anomaly detection functions.
    Runs multiple detectors and returns all detected issues.
    """
    
    def __init__(self, thresholds: Optional[DetectionThresholds] = None):
        self.initial_loss: Optional[float] = None
        self.thresholds = thresholds or DetectionThresholds()
    
    def detect_all(self, metrics: List[MetricSnapshot]) -> List[DetectionResult]:
        """
        Run all detection methods on metrics.
        
        Args:
            metrics: List of metric snapshots
        
        Returns:
            List of detected anomalies (may be empty)
        """
        if not metrics:
            return []
        
        # Set initial loss for divergence detection
        if self.initial_loss is None and len(metrics) > 0:
            self.initial_loss = metrics[0].loss
        
        results = []
        
        # Run all detectors
        detectors = [
            detect_nan_loss,
            detect_exploding_gradients,
            detect_vanishing_gradients,
            detect_loss_divergence,
            detect_loss_plateau,
            detect_overfitting,
        ]
        
        for detector in detectors:
            try:
                result = detector(metrics, self.initial_loss, self.thresholds)
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(
                    f"Detector {detector.__name__} raised an unexpected error and was skipped: {e}",
                    exc_info=True
                )
                continue
        
        # Sort by severity (critical first)
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3
        }
        results.sort(key=lambda r: severity_order[r.severity])
        
        return results


# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

def _linear_regression(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Compute simple linear regression slope and r_squared.

    Uses:
        slope = sum((xi-x̄)(yi-ȳ)) / sum((xi-x̄)^2)
        r_squared = 1 - (SS_res / SS_tot)

    All divisions are guarded with 1e-10.
    """
    eps = 1e-10
    n = min(len(x), len(y))
    if n < 2:
        return 0.0, 0.0

    x_vals = x[:n]
    y_vals = y[:n]

    mean_x = sum(x_vals) / n
    mean_y = sum(y_vals) / n

    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x_vals, y_vals))
    den = sum((xi - mean_x) ** 2 for xi in x_vals)
    slope = num / (den + eps)

    y_hat = [slope * xi + (mean_y - slope * mean_x) for xi in x_vals]
    ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(y_vals, y_hat))
    ss_tot = sum((yi - mean_y) ** 2 for yi in y_vals)
    r_squared = 1.0 - (ss_res / (ss_tot + eps))

    # Keep r_squared within [0, 1]
    r_squared = max(0.0, min(1.0, r_squared))
    return slope, r_squared

def detect_nan_loss(metrics: List[MetricSnapshot], 
                    initial_loss: Optional[float] = None,
                    thresholds: Optional[DetectionThresholds] = None) -> Optional[DetectionResult]:
    """
    Detect NaN or Inf in loss values.
    
    Args:
        metrics: List of metric snapshots
        initial_loss: Initial loss value (unused here)
        thresholds: Detection thresholds (unused here)
    
    Returns:
        DetectionResult if NaN/Inf detected, None otherwise
    """
    if not metrics:
        return None
    
    current = metrics[-1]
    
    if math.isnan(current.loss) or math.isinf(current.loss):
        return DetectionResult(
            anomaly_type=AnomalyType.NAN_LOSS,
            confidence=1.0,
            severity=Severity.CRITICAL,
            detected_at_epoch=current.epoch,
            detected_at_batch=current.batch,
            description="Loss became NaN or Inf - training has completely diverged",
            metric_snapshot=current,
            raw_values={
                'loss': current.loss,
                'grad_norm': current.grad_norm
            }
        )
    
    return None


def detect_exploding_gradients(metrics: List[MetricSnapshot],
                               initial_loss: Optional[float] = None,
                               thresholds: Optional[DetectionThresholds] = None) -> Optional[DetectionResult]:
    """
    Detect exploding gradients.
    
    Criteria:
    1. Gradient norm exceeds absolute threshold
    2. Gradient norm significantly higher than moving average
    
    Args:
        metrics: List of metric snapshots
        initial_loss: Initial loss value (unused here)
        thresholds: Detection thresholds
    
    Returns:
        DetectionResult if exploding gradients detected, None otherwise
    """
    if thresholds is None:
        thresholds = DetectionThresholds()
    eps_log = 1e-8
    if len(metrics) < 5:
        return None
    
    current = metrics[-1]
    recent = metrics[-20:] if len(metrics) >= 20 else metrics
    
    # Calculate moving average of gradient norms
    avg_grad_norm = np.mean([m.grad_norm for m in recent[:-1]]) if len(recent) > 1 else current.grad_norm
    
    # Check absolute threshold
    exceeds_threshold = current.grad_norm > thresholds.exploding_grad_threshold
    
    # Check relative to moving average
    relative_threshold = avg_grad_norm * thresholds.grad_multiplier_threshold
    exceeds_average = avg_grad_norm > 0 and current.grad_norm > relative_threshold

    # ---------------------------------------------------------------------
    # NEW A) Log Growth Rate Detection
    # r_i = log(g_i + 1e-8) - log(g_{i-1} + 1e-8)
    # Trigger if average of last 5 r_i > 0.1
    # ---------------------------------------------------------------------
    average_log_growth = 0.0
    log_growth_positive_count = 0
    if len(metrics) >= 10:
        grad_window_growth = [m.grad_norm for m in metrics[-10:]]
        log_growth_rates: List[float] = []
        for idx in range(1, len(grad_window_growth)):
            prev_g = grad_window_growth[idx - 1]
            curr_g = grad_window_growth[idx]
            r_i = math.log(curr_g + eps_log) - math.log(prev_g + eps_log)
            log_growth_rates.append(r_i)

        # Last 5 r_i values from the 10-point window
        if len(log_growth_rates) >= 5:
            last_5_rates = log_growth_rates[-5:]
            average_log_growth = float(np.mean(last_5_rates))
            # Noise guard: count how many of last 5 are positive
            log_growth_positive_count = sum(1 for r in last_5_rates if r > 0)
        else:
            average_log_growth = 0.0

    # Require average > 0.1 AND at least 4 out of 5 values positive (noise guard)
    log_growth_triggered = average_log_growth > 0.1 and log_growth_positive_count >= 4
    log_growth_confidence = min(1.0, average_log_growth / 0.3) if log_growth_triggered else 0.0

    # ---------------------------------------------------------------------
    # NEW B) Log-Linear Regression Detection
    # x = [0..window-1], y = log(grad_norm + 1e-8)
    # Trigger if slope > 0.1
    # ---------------------------------------------------------------------
    reg_slope, reg_r_squared = 0.0, 0.0
    if len(metrics) >= 10:
        grad_window_reg = [m.grad_norm for m in metrics[-10:]]
        x_reg = [float(i) for i in range(len(grad_window_reg))]
        y_reg = [math.log(g + eps_log) for g in grad_window_reg]

        # Minimum variance check: skip regression on near-constant signal
        y_variance = float(np.var(y_reg))
        if y_variance >= 1e-10:
            reg_slope, reg_r_squared = _linear_regression(x_reg, y_reg)

    reg_triggered = (
        reg_slope > 0.1
        and reg_r_squared > 0.6
    )
    reg_confidence = min(1.0, reg_slope / 0.3) if reg_triggered else 0.0

    # Existing confidence rules
    absolute_confidence = (
        calculate_confidence(current.grad_norm, thresholds.exploding_grad_threshold, inverse=False)
        if exceeds_threshold else 0.0
    )
    relative_confidence = (
        calculate_confidence(current.grad_norm, max(relative_threshold, 1e-10), inverse=False)
        if exceeds_average else 0.0
    )

    exploding_detected = (
        exceeds_threshold
        or exceeds_average
        or log_growth_triggered
        or reg_triggered
    )

    if exploding_detected:
        confidence = max(
            absolute_confidence,
            relative_confidence,
            log_growth_confidence,
            reg_confidence,
        )
        # Confidence floor: if regression or log-growth triggered, ensure >= 0.05
        if (log_growth_triggered or reg_triggered) and confidence < 0.05:
            confidence = 0.05
        confidence = max(0.0, min(1.0, confidence))

        trigger_reasons: List[str] = []
        if exceeds_threshold:
            trigger_reasons.append("absolute_threshold")
        if exceeds_average:
            trigger_reasons.append("relative_threshold")
        if log_growth_triggered:
            trigger_reasons.append("log_growth")
        if reg_triggered:
            trigger_reasons.append("log_linear_regression")

        severity = determine_severity(confidence)
        
        return DetectionResult(
            anomaly_type=AnomalyType.EXPLODING_GRADIENTS,
            confidence=confidence,
            severity=severity,
            detected_at_epoch=current.epoch,
            detected_at_batch=current.batch,
            description=(
                f"Exploding-gradient pattern detected (triggers: {', '.join(trigger_reasons)})"
            ),
            metric_snapshot=current,
            raw_values={
                'grad_norm': current.grad_norm,
                'threshold': thresholds.exploding_grad_threshold,
                'high_lr_threshold': thresholds.high_lr_threshold,
                'moving_average': avg_grad_norm,
                'relative_threshold': relative_threshold,
                'absolute_threshold_triggered': float(exceeds_threshold),
                'relative_threshold_triggered': float(exceeds_average),
                'average_log_growth': average_log_growth,
                'log_growth_triggered': float(log_growth_triggered),
                'regression_slope': reg_slope,
                'regression_r_squared': reg_r_squared,
                'regression_triggered': float(reg_triggered)
            }
        )
    
    return None


def detect_vanishing_gradients(metrics: List[MetricSnapshot],
                               initial_loss: Optional[float] = None,
                               thresholds: Optional[DetectionThresholds] = None) -> Optional[DetectionResult]:
    """
    Detect vanishing gradients.
    
    Criteria:
    1. Gradient norm falls below minimum threshold (absolute snapshot check)
    2. Gradient norm decaying at a fast rate (log-decay trend check)
    3. Gradient norm declining via log-linear regression (regression trend check)
    
    Args:
        metrics: List of metric snapshots
        initial_loss: Initial loss value (unused here)
        thresholds: Detection thresholds
    
    Returns:
        DetectionResult if vanishing gradients detected, None otherwise
    """
    if thresholds is None:
        thresholds = DetectionThresholds()
    
    if len(metrics) < 5:
        return None
    
    current = metrics[-1]
    eps_log = 1e-8
    
    # Check absolute snapshot threshold
    absolute_triggered = current.grad_norm < thresholds.vanishing_grad_threshold
    absolute_confidence = (
        calculate_confidence(
            current.grad_norm,
            thresholds.vanishing_grad_threshold,
            inverse=True
        ) if absolute_triggered else 0.0
    )
    
    # ---------------------------------------------------------------------
    # A) Log-Decay Rate Detection (mirrors log-growth for exploding)
    # r_i = log(g_i + eps) - log(g_{i-1} + eps)
    # Trigger if average of last 5 r_i < -0.1 AND at least 4/5 negative
    # ---------------------------------------------------------------------
    average_log_decay = 0.0
    log_decay_negative_count = 0
    log_decay_triggered = False
    log_decay_confidence = 0.0
    
    if len(metrics) >= 10:
        grad_window_decay = [m.grad_norm for m in metrics[-10:]]
        log_decay_rates: List[float] = []
        
        for idx in range(1, len(grad_window_decay)):
            prev_g = grad_window_decay[idx - 1]
            curr_g = grad_window_decay[idx]
            r_i = math.log(curr_g + eps_log) - math.log(prev_g + eps_log)
            log_decay_rates.append(r_i)
        
        # Last 5 r_i values
        if len(log_decay_rates) >= 5:
            last_5_rates = log_decay_rates[-5:]
            average_log_decay = float(np.mean(last_5_rates))
            log_decay_negative_count = sum(1 for r in last_5_rates if r < 0)
            
            # Trigger: average < -0.1 AND at least 4 out of 5 negative
            log_decay_triggered = average_log_decay < -0.1 and log_decay_negative_count >= 4
            log_decay_confidence = (
                min(1.0, abs(average_log_decay) / 0.3) if log_decay_triggered else 0.0
            )
    
    # ---------------------------------------------------------------------
    # B) Log-Linear Regression Detection
    # x = [0..9], y = log(grad_norm + eps)
    # Trigger if slope < -0.1 AND r_squared > 0.6
    # ---------------------------------------------------------------------
    reg_slope, reg_r_squared = 0.0, 0.0
    reg_triggered = False
    reg_confidence = 0.0
    
    if len(metrics) >= 10:
        grad_window_reg = [m.grad_norm for m in metrics[-10:]]
        x_reg = [float(i) for i in range(len(grad_window_reg))]
        y_reg = [math.log(g + eps_log) for g in grad_window_reg]
        
        # Minimum variance check
        y_variance = float(np.var(y_reg))
        if y_variance >= 1e-10:
            reg_slope, reg_r_squared = _linear_regression(x_reg, y_reg)
            reg_triggered = reg_slope < -0.1 and reg_r_squared > 0.6
            reg_confidence = min(1.0, abs(reg_slope) / 0.3) if reg_triggered else 0.0
    
    # Combine all signals
    vanishing_detected = (
        absolute_triggered
        or log_decay_triggered
        or reg_triggered
    )
    
    if vanishing_detected:
        confidence = max(
            absolute_confidence,
            log_decay_confidence,
            reg_confidence,
        )
        # Confidence floor: if trend-based triggered, ensure >= 0.05
        if (log_decay_triggered or reg_triggered) and confidence < 0.05:
            confidence = 0.05
        confidence = max(0.0, min(1.0, confidence))
        
        trigger_reasons: List[str] = []
        if absolute_triggered:
            trigger_reasons.append("absolute_threshold")
        if log_decay_triggered:
            trigger_reasons.append("log_decay")
        if reg_triggered:
            trigger_reasons.append("log_linear_regression")
        
        severity = determine_severity(confidence)
        
        return DetectionResult(
            anomaly_type=AnomalyType.VANISHING_GRADIENTS,
            confidence=confidence,
            severity=severity,
            detected_at_epoch=current.epoch,
            detected_at_batch=current.batch,
            description=(
                f"Vanishing-gradient pattern detected (triggers: {', '.join(trigger_reasons)})"
            ),
            metric_snapshot=current,
            raw_values={
                'grad_norm': current.grad_norm,
                'threshold': thresholds.vanishing_grad_threshold,
                'log_decay_triggered': float(log_decay_triggered),
                'average_log_decay': average_log_decay,
                'regression_slope': reg_slope,
                'regression_r_squared': reg_r_squared,
                'regression_triggered': float(reg_triggered)
            }
        )
    
    return None


def detect_loss_divergence(metrics: List[MetricSnapshot],
                           initial_loss: Optional[float] = None,
                           thresholds: Optional[DetectionThresholds] = None) -> Optional[DetectionResult]:
    """
    Detect loss divergence (loss increasing dramatically).
    
    Criteria: Current loss significantly higher than initial loss
    
    Args:
        metrics: List of metric snapshots
        initial_loss: Initial loss value for comparison
        thresholds: Detection thresholds
    
    Returns:
        DetectionResult if loss divergence detected, None otherwise
    """
    if thresholds is None:
        thresholds = DetectionThresholds()
    
    if not metrics or initial_loss is None or len(metrics) < 5:
        return None
    
    current = metrics[-1]
    
    # Skip if loss is NaN (handled by separate detector)
    if math.isnan(current.loss) or math.isinf(current.loss):
        return None
    
    # Existing check: divergence from initial loss
    threshold = initial_loss * thresholds.loss_divergence_multiplier
    old_logic_triggered = current.loss > threshold and current.loss > initial_loss
    ratio = current.loss / initial_loss if initial_loss > 1e-8 else float('inf')
    old_confidence = (
        min(1.0, (ratio - thresholds.loss_divergence_multiplier) / 5.0 + 0.5)
        if old_logic_triggered else 0.0
    )

    # ---------------------------------------------------------------------
    # NEW: Regression-based divergence over last 25 losses
    # L_t = a + b t
    # Divergence if:
    #   slope > 0, last_loss > first_loss, relative_slope > 0.05
    # ---------------------------------------------------------------------
    if len(metrics) >= 25:
        recent_window = metrics[-25:]
        losses = [m.loss for m in recent_window]

        # Keep temporal order and require all 25 points finite
        if all(not (math.isnan(l) or math.isinf(l)) for l in losses) and len(losses) == 25:
            # Minimum variance check: skip regression on near-constant signal
            loss_variance = float(np.var(losses))
            if loss_variance >= 1e-10:
                x = [float(i) for i in range(len(losses))]
                slope, reg_r_squared = _linear_regression(x, losses)
                first_loss = losses[0]
                last_loss = losses[-1]
                mean_loss = float(np.mean(losses))
                relative_slope = slope / mean_loss if mean_loss > 1e-8 else 0.0

                regression_triggered = (
                    slope > 0.0
                    and last_loss > first_loss
                    and relative_slope > 0.05
                    and reg_r_squared > 0.5
                )
                regression_confidence = min(1.0, relative_slope / 0.2) if regression_triggered else 0.0
            else:
                slope = 0.0
                reg_r_squared = 0.0
                first_loss = losses[0]
                last_loss = losses[-1]
                mean_loss = float(np.mean(losses))
                relative_slope = 0.0
                regression_triggered = False
                regression_confidence = 0.0
        else:
            slope = 0.0
            reg_r_squared = 0.0
            first_loss = current.loss
            last_loss = current.loss
            mean_loss = current.loss
            relative_slope = 0.0
            regression_triggered = False
            regression_confidence = 0.0
    else:
        slope = 0.0
        reg_r_squared = 0.0
        first_loss = current.loss
        last_loss = current.loss
        mean_loss = current.loss
        relative_slope = 0.0
        regression_triggered = False
        regression_confidence = 0.0

    divergence_detected = old_logic_triggered or regression_triggered

    if divergence_detected:
        confidence = max(old_confidence, regression_confidence)
        # Confidence floor: if regression triggered, ensure >= 0.05
        if regression_triggered and confidence < 0.05:
            confidence = 0.05
        confidence = max(0.0, min(1.0, confidence))

        severity = determine_severity(confidence)
        
        return DetectionResult(
            anomaly_type=AnomalyType.LOSS_DIVERGENCE,
            confidence=confidence,
            severity=severity,
            detected_at_epoch=current.epoch,
            detected_at_batch=current.batch,
            description=f"Loss ({current.loss:.4f}) diverged from initial ({initial_loss:.4f})",
            metric_snapshot=current,
            raw_values={
                'current_loss': current.loss,
                'initial_loss': initial_loss,
                'ratio': ratio,
                'high_lr_threshold': thresholds.high_lr_threshold,
                'old_logic_triggered': float(old_logic_triggered),
                'regression_slope': slope,
                'regression_r_squared': reg_r_squared,
                'first_loss_window': first_loss,
                'last_loss_window': last_loss,
                'mean_loss_window': mean_loss,
                'relative_slope': relative_slope,
                'regression_triggered': float(regression_triggered)
            }
        )
    
    return None


def detect_loss_plateau(metrics: List[MetricSnapshot],
                       initial_loss: Optional[float] = None,
                       thresholds: Optional[DetectionThresholds] = None) -> Optional[DetectionResult]:
    """
    Detect loss plateau (loss not changing) and distinguish between converged vs stuck.
    
    Criteria: Loss changes very little over extended window
    - Converged: flat loss at low value (< 10% of initial or < 0.01 absolute)
    - Stuck: flat loss at high value
    
    Args:
        metrics: List of metric snapshots
        initial_loss: Initial loss value for convergence check
        thresholds: Detection thresholds
    
    Returns:
        DetectionResult if plateau detected, None otherwise
    """
    if thresholds is None:
        thresholds = DetectionThresholds()
    
    window = thresholds.loss_plateau_window
    
    if len(metrics) < window:
        return None
    
    recent = metrics[-window:]
    current = metrics[-1]
    
    # Calculate loss variance and range
    losses = [m.loss for m in recent]
    
    # Skip if any NaN/Inf
    if any(math.isnan(l) or math.isinf(l) for l in losses):
        return None
    
    mean_loss = float(np.mean(losses))
    std_loss = float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0
    loss_range = max(losses) - min(losses)

    plateau_triggered = False
    confidence = 0.0
    cv = 0.0
    rr = 0.0
    is_converged = False

    # Relative tolerance logic
    if mean_loss > 1e-6:
        cv = std_loss / mean_loss
        rr = loss_range / mean_loss

        if cv < 0.01 and rr < 0.01:
            plateau_triggered = True
            confidence = 1.0 - (max(cv, rr) / 0.01)
    else:
        if std_loss < 1e-6 and loss_range < 1e-6:
            plateau_triggered = True
            confidence = 0.8

    confidence = max(0.0, min(1.0, confidence))

    if plateau_triggered:
        # Determine if converged (low loss) or stuck (high loss)
        convergence_threshold_relative = initial_loss * 0.1 if initial_loss is not None else float('inf')
        convergence_threshold_absolute = 0.01
        
        is_converged = mean_loss < convergence_threshold_relative or mean_loss < convergence_threshold_absolute
        
        # Only emit anomaly if stuck (high-loss plateau); converged low-loss plateaus are healthy
        if is_converged:
            return None
        
        severity = determine_severity(confidence)
        
        description = f"Loss is stuck at high value — plateau over {window} batches"
        
        return DetectionResult(
            anomaly_type=AnomalyType.LOSS_PLATEAU,
            confidence=confidence,
            severity=severity,
            detected_at_epoch=current.epoch,
            detected_at_batch=current.batch,
            description=description,
            metric_snapshot=current,
            raw_values={
                'mean_loss': mean_loss,
                'loss_std': std_loss,
                'loss_range': loss_range,
                'cv': cv,
                'rr': rr,
                'window_size': window
            }
        )
    
    return None


def detect_overfitting(metrics: List[MetricSnapshot],
                      initial_loss: Optional[float] = None,
                      thresholds: Optional[DetectionThresholds] = None) -> Optional[DetectionResult]:
    """
    Detect overfitting (train/validation gap).
    
    Criteria: Validation loss significantly higher than training loss
    
    Args:
        metrics: List of metric snapshots
        initial_loss: Initial loss value (unused here)
        thresholds: Detection thresholds
    
    Returns:
        DetectionResult if overfitting detected, None otherwise
    """
    if thresholds is None:
        thresholds = DetectionThresholds()
    
    if len(metrics) < 10:
        return None
    
    # Find recent metrics with validation loss
    recent_with_val = [m for m in metrics[-50:] if m.val_loss is not None]
    
    if not recent_with_val:
        return None
    
    current = recent_with_val[-1]
    
    # Skip if NaN/Inf
    if math.isnan(current.val_loss) or math.isinf(current.val_loss):
        return None
    
    # Use average of last few validation points for robustness (if available)
    if len(recent_with_val) >= 3:
        train_loss = np.mean([m.loss for m in recent_with_val[-3:]])
        val_loss = np.mean([m.val_loss for m in recent_with_val[-3:]])
    else:
        train_loss = current.loss
        val_loss = current.val_loss
    
    # Check absolute gap
    gap = val_loss - train_loss
    
    # Check ratio
    ratio = val_loss / train_loss if train_loss > 0 else float('inf')
    
    if gap > thresholds.overfitting_gap_threshold or \
       ratio > thresholds.overfitting_ratio_threshold:
        
        # Fixed confidence formula based on threshold exceedance:
        # Confidence = 0.0 at gap == threshold, scales to 1.0 at gap == threshold + 2.0
        if gap > thresholds.overfitting_gap_threshold:
            exceedance = gap - thresholds.overfitting_gap_threshold
            gap_confidence = min(1.0, exceedance / 2.0)
        else:
            gap_confidence = 0.0
        
        # Also compute from ratio using standard confidence function
        ratio_confidence = calculate_confidence(ratio, thresholds.overfitting_ratio_threshold)
        
        # Take max of both, clamp to [0, 1]
        confidence = max(gap_confidence, ratio_confidence)
        confidence = max(0.0, min(1.0, confidence))
        
        severity = determine_severity(confidence)
        
        return DetectionResult(
            anomaly_type=AnomalyType.OVERFITTING,
            confidence=confidence,
            severity=severity,
            detected_at_epoch=current.epoch,
            detected_at_batch=current.batch,
            description=f"Validation loss ({val_loss:.4f}) exceeds training loss ({train_loss:.4f})",
            metric_snapshot=current,
            raw_values={
                'train_loss': train_loss,
                'val_loss': val_loss,
                'gap': gap,
                'ratio': ratio,
                'gap_threshold': thresholds.overfitting_gap_threshold,
                'ratio_threshold': thresholds.overfitting_ratio_threshold
            }
        )
    
    return None
