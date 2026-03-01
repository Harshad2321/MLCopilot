"""
MLCopilot FastAPI Backend
Real-time ML training monitor with WebSocket streaming.
"""

import sys
import asyncio
import threading
import math
import time
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from pydantic import BaseModel, field_validator

from mlcopilot.types import MetricSnapshot, DetectionResult, Severity
from mlcopilot.detection import FailureDetector
from mlcopilot.analysis import RootCauseAnalyzer
from mlcopilot.recommendation import RecommendationEngine

# ============================================================================
# App Setup
# ============================================================================

app = FastAPI(title="MLCopilot Backend")

# Serve frontend
WEBVIEW_DIR = Path(__file__).parent.parent / "webview"
app.mount("/static", StaticFiles(directory=str(WEBVIEW_DIR)), name="static")

# ============================================================================
# Request Models
# ============================================================================

class SetLRRequest(BaseModel):
    """Request body for setting learning rate."""
    lr: float
    
    @field_validator('lr', mode='before')
    @classmethod
    def validate_lr(cls, v):
        """Ensure learning rate is positive and finite."""
        lr_val = float(v)
        if not math.isfinite(lr_val):
            raise ValueError('Learning rate must be a finite number')
        if lr_val <= 0:
            raise ValueError('Learning rate must be positive')
        return lr_val

# ============================================================================
# Global State
# ============================================================================

class TrainingState:
    def __init__(self):
        self.running = False
        self.stop_requested = False
        self.thread: Optional[threading.Thread] = None
        self.metrics: List[Dict[str, Any]] = []
        self.detections: List[Dict[str, Any]] = []
        self.diagnoses: List[Dict[str, Any]] = []
        self.recommendations: List[Dict[str, Any]] = []
        self.status = "idle"  # idle | running | stopped | error
        self.learning_rate = 0.01
        self.epoch = 0
        self.batch = 0
        self.ws_clients: List[WebSocket] = []
        self.ws_lock: threading.Lock = threading.Lock()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def reset(self):
        self.metrics.clear()
        self.detections.clear()
        self.diagnoses.clear()
        self.recommendations.clear()
        self.status = "idle"
        self.stop_requested = False
        self.epoch = 0
        self.batch = 0

state = TrainingState()
detector = FailureDetector()
analyzer = RootCauseAnalyzer()
recommender = RecommendationEngine()

# ============================================================================
# Demo Model & Training
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
    # Count only leaf modules (modules with no children)
    num_layers = sum(1 for m in model.modules() if not list(m.children()))
    has_sigmoid = any(isinstance(m, nn.Sigmoid) for m in model.modules())
    has_tanh = any(isinstance(m, nn.Tanh) for m in model.modules())
    
    return {
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "has_batchnorm": any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules()),
        "has_layernorm": any(isinstance(m, nn.LayerNorm) for m in model.modules()),
        "has_normalization": any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)) for m in model.modules()),
        "has_sigmoid": has_sigmoid,
        "has_tanh": has_tanh,
        "has_saturating_activations": has_sigmoid or has_tanh,
        "num_layers": num_layers,
        "model_type": type(model).__name__,
    }


def get_optimizer_info(optimizer: optim.Optimizer, lr: float) -> Dict[str, Any]:
    info = {"optimizer_type": type(optimizer).__name__, "learning_rate": lr}
    if optimizer.param_groups:
        pg = optimizer.param_groups[0]
        for key in ["momentum", "weight_decay", "betas", "eps"]:
            if key in pg:
                info[key] = pg[key]
    return info


async def broadcast(message: dict):
    """Send message to all connected WebSocket clients."""
    data = json.dumps(message)
    
    # Take a snapshot of clients under lock to avoid concurrent mutation
    with state.ws_lock:
        clients_snapshot = list(state.ws_clients)
    
    disconnected = []
    for ws in clients_snapshot:
        try:
            await ws.send_text(data)
        except Exception:
            disconnected.append(ws)
    
    # Remove disconnected clients under lock
    for ws in disconnected:
        with state.ws_lock:
            if ws in state.ws_clients:
                state.ws_clients.remove(ws)


def broadcast_from_thread(message: dict):
    """Thread-safe broadcast wrapper."""
    if state.loop:
        asyncio.run_coroutine_threadsafe(broadcast(message), state.loop)


def run_training():
    """Background training loop."""
    global detector
    detector = FailureDetector()
    state.status = "running"
    state.running = True
    broadcast_from_thread({"type": "status", "status": "running"})

    torch.manual_seed(int(time.time()) % 10000)
    model = DemoMLP(input_dim=10, hidden_dim=64, output_dim=2, num_layers=4)
    lr = state.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    X_train, y_train = create_dataset(500, 10)

    model_info = get_model_info(model)
    batch_size = 32
    max_epochs = 50
    metric_buffer: List[MetricSnapshot] = []
    initial_loss = None
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

                # Check if LR was changed from UI
                current_lr = state.learning_rate
                if abs(current_lr - lr) > 1e-10:
                    lr = current_lr
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

                xb = X_s[i : i + batch_size]
                yb = y_s[i : i + batch_size]

                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                if initial_loss is None:
                    initial_loss = loss_val

                grad_norm = compute_grad_norm(model)
                ps = get_param_stats(model)
                batch_in_epoch = i // batch_size

                snapshot = MetricSnapshot(
                    epoch=epoch, batch=batch_in_epoch, loss=loss_val,
                    grad_norm=grad_norm, learning_rate=lr,
                    param_mean=ps["mean"], param_std=ps["std"], param_max=ps["max"],
                    timestamp=time.time(),
                )
                metric_buffer.append(snapshot)
                if len(metric_buffer) > 1000:
                    metric_buffer.pop(0)
                global_batch += 1

                # Detection every 10 batches after warmup
                issue_data = None
                if global_batch >= 5 and global_batch % 10 == 0:
                    detections = detector.detect_all(metric_buffer)
                    if detections:
                        det = detections[0]
                        opt_info = get_optimizer_info(optimizer, lr)
                        diagnosis = analyzer.analyze(det, model_info, opt_info)
                        recs = recommender.generate(diagnosis)

                        det_dict = det.to_dict()
                        diag_dict = diagnosis.to_dict()
                        rec_dicts = [r.to_dict() for r in recs]

                        state.detections.append(det_dict)
                        state.diagnoses.append(diag_dict)
                        state.recommendations = rec_dicts

                        severity = det.severity.value
                        issue_data = {
                            "anomaly": det.anomaly_type.value,
                            "severity": severity,
                            "description": det.description,
                            "confidence": det.confidence,
                            "cause": diagnosis.primary_cause,
                            "reasoning": diagnosis.reasoning,
                            "recommendations": rec_dicts,
                        }

                # Build metric message
                metric_msg = {
                    "type": "metric",
                    "epoch": epoch,
                    "batch": batch_in_epoch,
                    "global_batch": global_batch,
                    "loss": loss_val if not (math.isnan(loss_val) or math.isinf(loss_val)) else None,
                    "grad_norm": grad_norm if not (math.isnan(grad_norm) or math.isinf(grad_norm)) else None,
                    "lr": lr,
                    "issue": issue_data,
                }
                state.metrics.append(metric_msg)
                broadcast_from_thread(metric_msg)

                time.sleep(0.08)  # ~12 updates/sec for smooth charting

    except Exception as e:
        state.status = "error"
        broadcast_from_thread({"type": "status", "status": "error", "message": str(e)})
    
    finally:
        state.running = False
        # Only set status to stopped if training completed normally (not error)
        if state.status != "error":
            state.status = "stopped"
            broadcast_from_thread({"type": "status", "status": "stopped"})


# ============================================================================
# Routes
# ============================================================================

@app.get("/")
async def index():
    return FileResponse(str(WEBVIEW_DIR / "index.html"))


@app.post("/start")
async def start_training():
    if state.running:
        return {"ok": False, "message": "Training already running"}
    state.reset()
    state.thread = threading.Thread(target=run_training, daemon=True)
    state.thread.start()
    return {"ok": True}


@app.post("/stop")
async def stop_training():
    if not state.running:
        return {"ok": False, "message": "No training running"}
    state.stop_requested = True
    return {"ok": True}


@app.post("/set_lr")
async def set_learning_rate(body: SetLRRequest):
    state.learning_rate = body.lr
    broadcast_from_thread({"type": "lr_changed", "lr": body.lr})
    return {"ok": True, "lr": body.lr}


@app.get("/status")
async def get_status():
    return {
        "status": state.status,
        "epoch": state.epoch,
        "batch": state.batch,
        "lr": state.learning_rate,
        "metrics_count": len(state.metrics),
        "detections_count": len(state.detections),
    }


# ============================================================================
# WebSocket
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    
    # Store the currently running event loop for thread-safe broadcasting
    state.loop = asyncio.get_running_loop()
    
    # Add client to list under lock
    with state.ws_lock:
        state.ws_clients.append(ws)
    
    try:
        while True:
            # Keep connection alive; client may send pings
            data = await ws.receive_text()
            # Could handle client messages here if needed
    except WebSocketDisconnect:
        # Remove client from list under lock
        with state.ws_lock:
            if ws in state.ws_clients:
                state.ws_clients.remove(ws)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  MLCopilot backend running on http://localhost:5050")
    print("=" * 55)
    uvicorn.run(app, host="0.0.0.0", port=5050, log_level="warning")
