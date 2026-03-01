"""
MLCopilot Backend Server
Self-contained FastAPI server for real-time ML training monitoring.
"""

import asyncio
import json
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# ============================================================================
# Types & Enums
# ============================================================================

class AnomalyType(Enum):
    EXPLODING_GRADIENTS = "exploding_gradients"
    VANISHING_GRADIENTS = "vanishing_gradients"
    LOSS_DIVERGENCE = "loss_divergence"
    LOSS_PLATEAU = "loss_plateau"
    NAN_LOSS = "nan_loss"
    OVERFITTING = "overfitting"


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CauseCategory(Enum):
    DATA_ISSUE = "data_issue"
    MODEL_ARCHITECTURE = "model_architecture"
    HYPERPARAMETER = "hyperparameter"
    OPTIMIZATION = "optimization"
    NUMERICAL_INSTABILITY = "numerical_instability"


@dataclass
class MetricSnapshot:
    epoch: int
    batch: int
    loss: float
    grad_norm: float
    learning_rate: float
    param_mean: float
    param_std: float
    param_max: float
    timestamp: float


@dataclass
class DetectionResult:
    anomaly_type: AnomalyType
    confidence: float
    severity: Severity
    detected_at_epoch: int
    detected_at_batch: int
    description: str
    metric_snapshot: MetricSnapshot
    raw_values: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_type": self.anomaly_type.value,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "detected_at_epoch": self.detected_at_epoch,
            "detected_at_batch": self.detected_at_batch,
            "description": self.description,
            "raw_values": self.raw_values,
        }


@dataclass
class Diagnosis:
    cause_category: CauseCategory
    primary_cause: str
    reasoning: str
    contributing_factors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cause_category": self.cause_category.value,
            "primary_cause": self.primary_cause,
            "reasoning": self.reasoning,
            "contributing_factors": self.contributing_factors,
        }


@dataclass
class Recommendation:
    priority: Priority
    action: str
    reasoning: str
    code_example: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "priority": self.priority.value,
            "action": self.action,
            "reasoning": self.reasoning,
            "code_example": self.code_example,
        }


# ============================================================================
# Thresholds
# ============================================================================

GRAD_NORM_EXPLODING = 100.0
GRAD_NORM_VANISHING = 1e-7
LOSS_DIVERGENCE_FACTOR = 10.0
LOSS_PLATEAU_THRESHOLD = 0.001
HIGH_LR_THRESHOLD = 0.1


# ============================================================================
# Detection
# ============================================================================

class FailureDetector:
    def __init__(self):
        self.initial_loss: Optional[float] = None

    def detect_all(self, metrics: List[MetricSnapshot]) -> List[DetectionResult]:
        if not metrics:
            return []

        if self.initial_loss is None:
            self.initial_loss = metrics[0].loss

        results = []
        current = metrics[-1]

        if self._detect_nan_loss(current):
            results.append(self._detect_nan_loss(current))
        if self._detect_exploding_gradients(current):
            results.append(self._detect_exploding_gradients(current))
        if self._detect_vanishing_gradients(current):
            results.append(self._detect_vanishing_gradients(current))
        if len(metrics) > 10:
            divergence = self._detect_loss_divergence(metrics)
            if divergence:
                results.append(divergence)
            plateau = self._detect_loss_plateau(metrics)
            if plateau:
                results.append(plateau)

        results.sort(key=lambda r: {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3
        }[r.severity])

        return results

    def _detect_nan_loss(self, current: MetricSnapshot) -> Optional[DetectionResult]:
        if math.isnan(current.loss) or math.isinf(current.loss):
            return DetectionResult(
                anomaly_type=AnomalyType.NAN_LOSS,
                confidence=1.0,
                severity=Severity.CRITICAL,
                detected_at_epoch=current.epoch,
                detected_at_batch=current.batch,
                description="Loss became NaN or Inf",
                metric_snapshot=current,
                raw_values={"loss": current.loss, "grad_norm": current.grad_norm},
            )
        return None

    def _detect_exploding_gradients(self, current: MetricSnapshot) -> Optional[DetectionResult]:
        if current.grad_norm > GRAD_NORM_EXPLODING:
            confidence = min(1.0, current.grad_norm / (GRAD_NORM_EXPLODING * 10))
            severity = Severity.CRITICAL if current.grad_norm > 1000 else Severity.HIGH
            return DetectionResult(
                anomaly_type=AnomalyType.EXPLODING_GRADIENTS,
                confidence=confidence,
                severity=severity,
                detected_at_epoch=current.epoch,
                detected_at_batch=current.batch,
                description=f"Gradient norm ({current.grad_norm:.2f}) exceeds threshold",
                metric_snapshot=current,
                raw_values={"grad_norm": current.grad_norm},
            )
        return None

    def _detect_vanishing_gradients(self, current: MetricSnapshot) -> Optional[DetectionResult]:
        if current.grad_norm < GRAD_NORM_VANISHING:
            return DetectionResult(
                anomaly_type=AnomalyType.VANISHING_GRADIENTS,
                confidence=0.9,
                severity=Severity.HIGH,
                detected_at_epoch=current.epoch,
                detected_at_batch=current.batch,
                description=f"Gradient norm ({current.grad_norm:.2e}) is vanishing",
                metric_snapshot=current,
                raw_values={"grad_norm": current.grad_norm},
            )
        return None

    def _detect_loss_divergence(self, metrics: List[MetricSnapshot]) -> Optional[DetectionResult]:
        if self.initial_loss is None or self.initial_loss == 0:
            return None
        current = metrics[-1]
        if current.loss > self.initial_loss * LOSS_DIVERGENCE_FACTOR:
            return DetectionResult(
                anomaly_type=AnomalyType.LOSS_DIVERGENCE,
                confidence=0.85,
                severity=Severity.HIGH,
                detected_at_epoch=current.epoch,
                detected_at_batch=current.batch,
                description=f"Loss diverging: {current.loss:.4f} vs initial {self.initial_loss:.4f}",
                metric_snapshot=current,
                raw_values={"loss": current.loss, "initial_loss": self.initial_loss},
            )
        return None

    def _detect_loss_plateau(self, metrics: List[MetricSnapshot]) -> Optional[DetectionResult]:
        if len(metrics) < 20:
            return None
        recent = [m.loss for m in metrics[-20:]]
        if all(math.isfinite(l) for l in recent):
            variance = np.var(recent)
            if variance < LOSS_PLATEAU_THRESHOLD:
                current = metrics[-1]
                return DetectionResult(
                    anomaly_type=AnomalyType.LOSS_PLATEAU,
                    confidence=0.75,
                    severity=Severity.MEDIUM,
                    detected_at_epoch=current.epoch,
                    detected_at_batch=current.batch,
                    description="Loss has plateaued",
                    metric_snapshot=current,
                    raw_values={"loss_variance": float(variance)},
                )
        return None


# ============================================================================
# Analysis
# ============================================================================

class RootCauseAnalyzer:
    def analyze(
        self,
        detection: DetectionResult,
        model_info: Dict[str, Any],
        optimizer_info: Dict[str, Any],
    ) -> Diagnosis:
        analyzers = {
            AnomalyType.EXPLODING_GRADIENTS: self._analyze_exploding,
            AnomalyType.VANISHING_GRADIENTS: self._analyze_vanishing,
            AnomalyType.LOSS_DIVERGENCE: self._analyze_divergence,
            AnomalyType.LOSS_PLATEAU: self._analyze_plateau,
            AnomalyType.NAN_LOSS: self._analyze_nan,
        }
        analyzer = analyzers.get(detection.anomaly_type, self._analyze_generic)
        return analyzer(detection, model_info, optimizer_info)

    def _analyze_exploding(
        self, detection: DetectionResult, model_info: Dict, optimizer_info: Dict
    ) -> Diagnosis:
        lr = optimizer_info.get("learning_rate", 0.0)
        if lr > HIGH_LR_THRESHOLD:
            return Diagnosis(
                cause_category=CauseCategory.HYPERPARAMETER,
                primary_cause="Learning rate too high",
                reasoning=f"LR={lr} is causing unstable updates",
                contributing_factors=["High learning rate", "Large weight updates"],
            )
        if not model_info.get("has_normalization", False):
            return Diagnosis(
                cause_category=CauseCategory.MODEL_ARCHITECTURE,
                primary_cause="Missing normalization layers",
                reasoning="No BatchNorm/LayerNorm detected",
                contributing_factors=["Unbounded activations", "Deep network"],
            )
        return Diagnosis(
            cause_category=CauseCategory.OPTIMIZATION,
            primary_cause="Optimization instability",
            reasoning="Gradients exploding during backpropagation",
            contributing_factors=["Gradient clipping may help"],
        )

    def _analyze_vanishing(
        self, detection: DetectionResult, model_info: Dict, optimizer_info: Dict
    ) -> Diagnosis:
        return Diagnosis(
            cause_category=CauseCategory.MODEL_ARCHITECTURE,
            primary_cause="Vanishing gradients in deep network",
            reasoning="Gradients shrinking to near-zero",
            contributing_factors=["Deep architecture", "Saturating activations"],
        )

    def _analyze_divergence(
        self, detection: DetectionResult, model_info: Dict, optimizer_info: Dict
    ) -> Diagnosis:
        lr = optimizer_info.get("learning_rate", 0.0)
        return Diagnosis(
            cause_category=CauseCategory.HYPERPARAMETER,
            primary_cause="Learning rate causing divergence",
            reasoning=f"Loss increasing rapidly with LR={lr}",
            contributing_factors=["Reduce learning rate", "Check data normalization"],
        )

    def _analyze_plateau(
        self, detection: DetectionResult, model_info: Dict, optimizer_info: Dict
    ) -> Diagnosis:
        return Diagnosis(
            cause_category=CauseCategory.OPTIMIZATION,
            primary_cause="Training stuck in local minimum",
            reasoning="Loss variance near zero",
            contributing_factors=["Increase learning rate", "Use learning rate scheduler"],
        )

    def _analyze_nan(
        self, detection: DetectionResult, model_info: Dict, optimizer_info: Dict
    ) -> Diagnosis:
        return Diagnosis(
            cause_category=CauseCategory.NUMERICAL_INSTABILITY,
            primary_cause="Numerical overflow or invalid operation",
            reasoning="NaN/Inf in loss indicates catastrophic failure",
            contributing_factors=["Reduce LR", "Add gradient clipping", "Check data"],
        )

    def _analyze_generic(
        self, detection: DetectionResult, model_info: Dict, optimizer_info: Dict
    ) -> Diagnosis:
        return Diagnosis(
            cause_category=CauseCategory.OPTIMIZATION,
            primary_cause="Unknown training issue",
            reasoning="Unable to determine specific cause",
            contributing_factors=["Review training configuration"],
        )


# ============================================================================
# Recommendations
# ============================================================================

class RecommendationEngine:
    def generate(self, diagnosis: Diagnosis, detection: DetectionResult) -> List[Recommendation]:
        generators = {
            AnomalyType.EXPLODING_GRADIENTS: self._rec_exploding,
            AnomalyType.VANISHING_GRADIENTS: self._rec_vanishing,
            AnomalyType.LOSS_DIVERGENCE: self._rec_divergence,
            AnomalyType.LOSS_PLATEAU: self._rec_plateau,
            AnomalyType.NAN_LOSS: self._rec_nan,
        }
        gen = generators.get(detection.anomaly_type, self._rec_generic)
        recs = gen(diagnosis)
        recs.sort(key=lambda r: {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3
        }[r.priority])
        return recs

    def _rec_exploding(self, diagnosis: Diagnosis) -> List[Recommendation]:
        return [
            Recommendation(
                priority=Priority.CRITICAL,
                action="Add Gradient Clipping",
                reasoning="Prevents gradients from growing unbounded",
                code_example="torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
            ),
            Recommendation(
                priority=Priority.HIGH,
                action="Reduce Learning Rate",
                reasoning="Smaller updates prevent explosive growth",
                code_example="optimizer = Adam(model.parameters(), lr=1e-4)",
            ),
        ]

    def _rec_vanishing(self, diagnosis: Diagnosis) -> List[Recommendation]:
        return [
            Recommendation(
                priority=Priority.HIGH,
                action="Use residual connections",
                reasoning="Skip connections help gradient flow",
                code_example="output = layer(x) + x",
            ),
            Recommendation(
                priority=Priority.MEDIUM,
                action="Use different activation",
                reasoning="ReLU variants prevent saturation",
                code_example="nn.LeakyReLU(0.1) or nn.GELU()",
            ),
        ]

    def _rec_divergence(self, diagnosis: Diagnosis) -> List[Recommendation]:
        return [
            Recommendation(
                priority=Priority.CRITICAL,
                action="Reduce Learning Rate",
                reasoning="High LR causes overshooting",
                code_example="optimizer = Adam(model.parameters(), lr=1e-5)",
            ),
        ]

    def _rec_plateau(self, diagnosis: Diagnosis) -> List[Recommendation]:
        return [
            Recommendation(
                priority=Priority.HIGH,
                action="Use Learning Rate Scheduler",
                reasoning="Dynamic LR helps escape plateaus",
                code_example="scheduler = ReduceLROnPlateau(optimizer)",
            ),
            Recommendation(
                priority=Priority.MEDIUM,
                action="Increase Learning Rate",
                reasoning="Higher LR may escape local minimum",
            ),
        ]

    def _rec_nan(self, diagnosis: Diagnosis) -> List[Recommendation]:
        return [
            Recommendation(
                priority=Priority.CRITICAL,
                action="Reduce Learning Rate by 10x",
                reasoning="NaN usually means catastrophic instability",
                code_example="optimizer = Adam(model.parameters(), lr=1e-5)",
            ),
            Recommendation(
                priority=Priority.CRITICAL,
                action="Add Gradient Clipping",
                reasoning="Prevents numerical overflow",
                code_example="torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
            ),
        ]

    def _rec_generic(self, diagnosis: Diagnosis) -> List[Recommendation]:
        return [
            Recommendation(
                priority=Priority.MEDIUM,
                action="Review training configuration",
                reasoning="Check hyperparameters and data pipeline",
            ),
        ]


# ============================================================================
# Demo Model
# ============================================================================

class DemoMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=2, num_layers=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def create_dataset(n_samples=500, input_dim=10):
    X = torch.randn(n_samples, input_dim)
    y = (X.sum(dim=1) > 0).long()
    return X, y


def compute_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)


def get_param_stats(model: nn.Module) -> Dict[str, float]:
    all_p = []
    for p in model.parameters():
        all_p.append(p.data.cpu().numpy().flatten())
    if not all_p:
        return {"mean": 0.0, "std": 0.0, "max": 0.0}
    cat = np.concatenate(all_p)
    return {"mean": float(np.mean(cat)), "std": float(np.std(cat)), "max": float(np.max(np.abs(cat)))}


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    return {
        "total_params": sum(p.numel() for p in model.parameters()),
        "has_normalization": any(
            isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm))
            for m in model.modules()
        ),
        "num_layers": len(list(model.modules())),
    }


def get_optimizer_info(optimizer: optim.Optimizer, lr: float) -> Dict[str, Any]:
    return {"optimizer_type": type(optimizer).__name__, "learning_rate": lr}


# ============================================================================
# Application State
# ============================================================================

class TrainingState:
    def __init__(self):
        self.running = False
        self.stop_requested = False
        self.thread: Optional[threading.Thread] = None
        self.metrics: List[Dict[str, Any]] = []
        self.status = "idle"
        self.learning_rate = 0.01
        self.epoch = 0
        self.batch = 0
        self.ws_clients: List[WebSocket] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def reset(self):
        self.metrics.clear()
        self.status = "idle"
        self.stop_requested = False
        self.epoch = 0
        self.batch = 0


state = TrainingState()
detector = FailureDetector()
analyzer = RootCauseAnalyzer()
recommender = RecommendationEngine()


# ============================================================================
# Broadcasting
# ============================================================================

async def broadcast(message: dict):
    data = json.dumps(message)
    disconnected = []
    for ws in state.ws_clients:
        try:
            await ws.send_text(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        state.ws_clients.remove(ws)


def broadcast_from_thread(message: dict):
    if state.loop:
        asyncio.run_coroutine_threadsafe(broadcast(message), state.loop)


# ============================================================================
# Training Loop
# ============================================================================

def run_training():
    global detector
    detector = FailureDetector()
    state.status = "running"
    state.running = True
    broadcast_from_thread({"type": "status", "status": "running"})

    torch.manual_seed(int(time.time()) % 10000)
    model = DemoMLP()
    lr = state.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    X_train, y_train = create_dataset()

    model_info = get_model_info(model)
    batch_size = 32
    max_epochs = 50
    metric_buffer: List[MetricSnapshot] = []
    global_batch = 0

    try:
        for epoch in range(max_epochs):
            if state.stop_requested:
                break

            state.epoch = epoch
            indices = torch.randperm(len(X_train))
            X_s, y_s = X_train[indices], y_train[indices]

            for i in range(0, len(X_train), batch_size):
                if state.stop_requested:
                    break

                current_lr = state.learning_rate
                if abs(current_lr - lr) > 1e-10:
                    lr = current_lr
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

                xb = X_s[i: i + batch_size]
                yb = y_s[i: i + batch_size]

                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                grad_norm = compute_grad_norm(model)
                ps = get_param_stats(model)
                batch_in_epoch = i // batch_size

                snapshot = MetricSnapshot(
                    epoch=epoch,
                    batch=batch_in_epoch,
                    loss=loss_val,
                    grad_norm=grad_norm,
                    learning_rate=lr,
                    param_mean=ps["mean"],
                    param_std=ps["std"],
                    param_max=ps["max"],
                    timestamp=time.time(),
                )
                metric_buffer.append(snapshot)
                if len(metric_buffer) > 1000:
                    metric_buffer.pop(0)
                global_batch += 1

                issue_data = None
                if global_batch >= 5 and global_batch % 10 == 0:
                    detections = detector.detect_all(metric_buffer)
                    if detections:
                        det = detections[0]
                        opt_info = get_optimizer_info(optimizer, lr)
                        diagnosis = analyzer.analyze(det, model_info, opt_info)
                        recs = recommender.generate(diagnosis, det)

                        issue_data = {
                            "anomaly": det.anomaly_type.value,
                            "severity": det.severity.value,
                            "description": det.description,
                            "confidence": det.confidence,
                            "cause": diagnosis.primary_cause,
                            "reasoning": diagnosis.reasoning,
                            "recommendations": [r.to_dict() for r in recs],
                        }

                metric_msg = {
                    "type": "metric",
                    "epoch": epoch,
                    "batch": batch_in_epoch,
                    "global_batch": global_batch,
                    "loss": loss_val if math.isfinite(loss_val) else None,
                    "grad_norm": grad_norm if math.isfinite(grad_norm) else None,
                    "lr": lr,
                    "issue": issue_data,
                }
                state.metrics.append(metric_msg)
                broadcast_from_thread(metric_msg)

                time.sleep(0.08)

    except Exception:
        state.status = "error"
        broadcast_from_thread({"type": "status", "status": "error"})
        return

    state.running = False
    state.status = "stopped"
    broadcast_from_thread({"type": "status", "status": "stopped"})


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="MLCopilot Backend")

WEB_DIR = Path(__file__).parent.parent / "web"
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.post("/start")
async def start_training():
    if state.running:
        return {"ok": False, "message": "Already running"}
    state.reset()
    state.thread = threading.Thread(target=run_training, daemon=True)
    state.thread.start()
    return {"ok": True}


@app.post("/stop")
async def stop_training():
    if not state.running:
        return {"ok": False, "message": "Not running"}
    state.stop_requested = True
    return {"ok": True}


@app.post("/set_lr")
async def set_learning_rate(body: dict):
    lr = float(body.get("lr", state.learning_rate))
    if lr <= 0:
        return {"ok": False, "message": "LR must be positive"}
    state.learning_rate = lr
    broadcast_from_thread({"type": "lr_changed", "lr": lr})
    return {"ok": True, "lr": lr}


@app.get("/status")
async def get_status():
    return {
        "status": state.status,
        "epoch": state.epoch,
        "batch": state.batch,
        "lr": state.learning_rate,
        "metrics_count": len(state.metrics),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state.ws_clients.append(ws)
    state.loop = asyncio.get_event_loop()
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in state.ws_clients:
            state.ws_clients.remove(ws)


@app.websocket("/ws/client")
async def client_websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state.loop = asyncio.get_event_loop()
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                msg_type = msg.get("type", "")
                if msg_type == "status":
                    state.status = msg.get("status", "idle")
                    state.running = state.status == "running"
                    await broadcast(msg)
                elif msg_type == "metric":
                    state.epoch = msg.get("epoch", 0)
                    state.batch = msg.get("batch", 0)
                    state.metrics.append(msg)
                    await broadcast(msg)
                else:
                    await broadcast(msg)
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        state.running = False
        state.status = "stopped"
        await broadcast({"type": "status", "status": "stopped"})


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050, log_level="warning")
