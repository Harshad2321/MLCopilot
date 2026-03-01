"""
MLCopilot - Real-Time ML Training Failure Detection & Fix Recommendation Engine

A production-quality ML monitoring system that detects training instability,
performs root cause analysis, and generates actionable recommendations.
"""

__version__ = "0.1.0"

from .types import (
    MetricSnapshot,
    DetectionResult,
    Diagnosis,
    Recommendation,
    AnomalyType,
    Severity,
    CauseCategory,
    Priority
)

from .monitoring import TrainingMonitor
from .detection import FailureDetector
from .analysis import RootCauseAnalyzer
from .recommendation import RecommendationEngine
from .cli import CLIReporter

__all__ = [
    # Core classes
    'TrainingMonitor',
    'FailureDetector',
    'RootCauseAnalyzer',
    'RecommendationEngine',
    'CLIReporter',
    
    # Data types
    'MetricSnapshot',
    'DetectionResult',
    'Diagnosis',
    'Recommendation',
    
    # Enums
    'AnomalyType',
    'Severity',
    'CauseCategory',
    'Priority',
]
