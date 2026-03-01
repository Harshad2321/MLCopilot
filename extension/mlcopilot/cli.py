"""
MLCopilot CLI Reporter
Beautiful terminal output for detection results, diagnosis, and recommendations.
"""

from typing import List
from .types import DetectionResult, Diagnosis, Recommendation, Severity, Priority


class CLIReporter:
    """
    Formats and displays monitoring results in the terminal.
    Uses simple ASCII formatting for maximum compatibility.
    """
    
    # ANSI color codes (fallback to plain if not supported)
    COLORS = {
        'red': '\033[91m',
        'yellow': '\033[93m',
        'green': '\033[92m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'magenta': '\033[95m',
        'bold': '\033[1m',
        'reset': '\033[0m'
    }
    
    def __init__(self, use_colors: bool = True):
        """
        Initialize CLI reporter.
        
        Args:
            use_colors: Whether to use ANSI colors (disable for plain terminals)
        """
        self.use_colors = use_colors
    
    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.use_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"
    
    def _bold(self, text: str) -> str:
        """Make text bold."""
        return self._color(text, 'bold')
    
    def _header(self, text: str, char: str = '=') -> str:
        """Create a header with separator line."""
        line = char * 70
        return f"\n{self._bold(text)}\n{line}"
    
    def _severity_icon(self, severity: Severity) -> str:
        """Get icon for severity level."""
        icons = {
            Severity.CRITICAL: '🚨',
            Severity.HIGH: '⚠️ ',
            Severity.MEDIUM: '⚡',
            Severity.LOW: 'ℹ️ '
        }
        return icons.get(severity, '•')
    
    def _severity_color(self, severity: Severity) -> str:
        """Get color for severity level."""
        colors = {
            Severity.CRITICAL: 'red',
            Severity.HIGH: 'yellow',
            Severity.MEDIUM: 'yellow',
            Severity.LOW: 'blue'
        }
        return colors.get(severity, 'reset')
    
    def _priority_icon(self, priority: Priority) -> str:
        """Get icon for priority level."""
        icons = {
            Priority.CRITICAL: '🔴',
            Priority.HIGH: '🟠',
            Priority.MEDIUM: '🟡',
            Priority.LOW: '🟢'
        }
        return icons.get(priority, '•')
    
    def report_detection(self, detection: DetectionResult):
        """Display detection result."""
        severity_color = self._severity_color(detection.severity)
        severity_text = detection.severity.value.upper()
        
        print(self._header(f"{self._severity_icon(detection.severity)} TRAINING FAILURE DETECTED", '='))
        
        print(f"\n{self._bold('Issue:')} {self._color(detection.description, severity_color)}")
        print(f"{self._bold('Type:')} {detection.anomaly_type.value.replace('_', ' ').title()}")
        print(f"{self._bold('Severity:')} {self._color(severity_text, severity_color)}")
        print(f"{self._bold('Confidence:')} {detection.confidence:.2%}")
        print(f"{self._bold('Detected at:')} Epoch {detection.detected_at_epoch}, Batch {detection.detected_at_batch}")
        
        # Metrics snapshot
        print(self._header("📊 Metrics at Detection", '-'))
        snapshot = detection.metric_snapshot
        print(f"  • Loss: {snapshot.loss:.6f}")
        print(f"  • Gradient Norm: {snapshot.grad_norm:.6f}")
        print(f"  • Learning Rate: {snapshot.learning_rate:.2e}")
        print(f"  • Param Mean: {snapshot.param_mean:.6f}")
        print(f"  • Param Std: {snapshot.param_std:.6f}")
        
        # Raw values (if interesting)
        if detection.raw_values:
            print(f"\n{self._bold('Additional Details:')}")
            for key, value in detection.raw_values.items():
                if isinstance(value, float):
                    print(f"  • {key}: {value:.6f}")
                else:
                    print(f"  • {key}: {value}")
    
    def report_diagnosis(self, diagnosis: Diagnosis):
        """Display diagnosis results."""
        print(self._header("🔍 ROOT CAUSE ANALYSIS", '='))
        
        print(f"\n{self._bold('Category:')} {self._color(diagnosis.cause_category.value.replace('_', ' ').title(), 'cyan')}")
        print(f"{self._bold('Primary Cause:')} {diagnosis.primary_cause}")
        
        print(f"\n{self._bold('Reasoning:')}")
        print(f"{diagnosis.reasoning}")
        
        if diagnosis.contributing_factors:
            print(f"\n{self._bold('Contributing Factors:')}")
            for factor in diagnosis.contributing_factors:
                print(f"  • {factor}")
        
        # Context info
        print(self._header("📋 Context", '-'))
        
        print(f"{self._bold('Model:')}")
        model_ctx = diagnosis.model_context
        if 'model_type' in model_ctx:
            print(f"  • Type: {model_ctx['model_type']}")
        if 'total_params' in model_ctx:
            print(f"  • Parameters: {model_ctx['total_params']:,}")
        if 'has_normalization' in model_ctx:
            norm_status = '✓ Yes' if model_ctx['has_normalization'] else '✗ No'
            print(f"  • Normalization: {norm_status}")
        if 'num_layers' in model_ctx:
            print(f"  • Layers: {model_ctx['num_layers']}")
        
        print(f"\n{self._bold('Optimizer:')}")
        opt_ctx = diagnosis.optimizer_context
        if 'optimizer_type' in opt_ctx:
            print(f"  • Type: {opt_ctx['optimizer_type']}")
        if 'learning_rate' in opt_ctx:
            print(f"  • Learning Rate: {opt_ctx['learning_rate']:.2e}")
        if 'momentum' in opt_ctx:
            print(f"  • Momentum: {opt_ctx['momentum']}")
        if 'weight_decay' in opt_ctx:
            print(f"  • Weight Decay: {opt_ctx['weight_decay']}")
    
    def report_recommendations(self, recommendations: List[Recommendation]):
        """Display recommendations."""
        print(self._header("💡 RECOMMENDATIONS", '='))
        
        if not recommendations:
            print("\nNo specific recommendations available.")
            return
        
        print(f"\n{len(recommendations)} actionable fix(es) suggested:\n")
        
        for i, rec in enumerate(recommendations, 1):
            priority_color = 'red' if rec.priority in [Priority.CRITICAL, Priority.HIGH] else 'yellow'
            
            print(f"{self._bold(f'[{i}]')} {self._priority_icon(rec.priority)} "
                  f"{self._color(rec.priority.value.upper(), priority_color)} - {self._bold(rec.action)}")
            print(f"    Category: {rec.category}")
            
            if rec.current_value:
                print(f"    Current:  {rec.current_value}")
            if rec.suggested_value:
                print(f"    Suggested: {self._color(rec.suggested_value, 'green')}")
            
            print(f"\n    {rec.reasoning}")
            
            if rec.code_example:
                print(f"\n    {self._color('Code Example:', 'cyan')}")
                # Indent code
                for line in rec.code_example.split('\n'):
                    print(f"    {line}")
            
            if rec.expected_impact:
                print(f"\n    {self._bold('Expected Impact:')} {rec.expected_impact}")
            
            print()  # Extra line between recommendations
    
    def report_full(self, detection: DetectionResult, 
                   diagnosis: Diagnosis, 
                   recommendations: List[Recommendation]):
        """Display complete report: detection, diagnosis, and recommendations."""
        print("\n" + "=" * 70)
        print(self._color(self._bold("MLCopilot - Training Failure Analysis Report"), 'magenta'))
        print("=" * 70)
        
        self.report_detection(detection)
        self.report_diagnosis(diagnosis)
        self.report_recommendations(recommendations)
        
        print("\n" + "=" * 70)
        print(self._color("End of Report", 'magenta'))
        print("=" * 70 + "\n")
    
    def report_no_issues(self):
        """Display message when no issues detected."""
        print(f"\n{self._color('✓', 'green')} {self._bold('No training issues detected')} - Training appears healthy!\n")
    
    def report_monitoring_start(self, model_info: dict, optimizer_info: dict):
        """Display monitoring initialization message."""
        print("\n" + "=" * 70)
        print(self._color(self._bold("MLCopilot Monitoring Started"), 'cyan'))
        print("=" * 70)
        
        print(f"\n{self._bold('Model:')} {model_info.get('model_type', 'Unknown')}")
        print(f"  • Parameters: {model_info.get('total_params', 0):,}")
        print(f"  • Normalization: {'Yes' if model_info.get('has_normalization') else 'No'}")
        
        print(f"\n{self._bold('Optimizer:')} {optimizer_info.get('optimizer_type', 'Unknown')}")
        print(f"  • Learning Rate: {optimizer_info.get('learning_rate', 0):.2e}")
        
        print(f"\n{self._color('Monitoring active...', 'green')}")
        print("=" * 70 + "\n")
