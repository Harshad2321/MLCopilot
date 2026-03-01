# ⚡ MLCopilot – Real-Time ML Training Monitor

> **Detect training failures. Get root cause analysis. Fix your model — live.**

MLCopilot is a real-time monitoring system for PyTorch training that detects anomalies (exploding gradients, loss divergence, overfitting), identifies root causes, and recommends fixes with code examples—all from a beautiful web dashboard.

---

## 🎯 Features

| Feature | What It Does |
|---------|-------------|
| **Live Metrics** | Real-time loss & gradient norm charts |
| **Anomaly Detection** | Detects exploding/vanishing gradients, loss divergence, overfitting, NaN losses |
| **Root Cause Analysis** | Identifies why training failed (hyperparameter, architecture, optimization) |
| **Fix Recommendations** | Actionable suggestions with Python code examples |
| **Live LR Control** | Adjust learning rate mid-training from the UI |
| **Status Badges** | Healthy / Warning / Critical at a glance |
| **Dark Web UI** | Clean, minimal dashboard — no React bloat |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install fastapi uvicorn[standard] websockets torch numpy
```

### 2. Run the backend server
```bash
python python-backend/server.py
```

### 3. Open dashboard
Dashboard auto-opens at **http://localhost:5050**

### 4. Start training
Click **[▶ Start Training]** to run a demo. Watch metrics stream live.

---

## 📁 Project Structure

```
MLCopilot/
├── python-backend/
│   ├── server.py              # FastAPI backend with WebSocket
│   └── requirements.txt        # Dependencies
├── webview/
│   ├── index.html             # Dashboard HTML
│   ├── style.css              # Dark theme CSS
│   └── app.js                 # WebSocket client + Chart.js
└── mlcopilot/
    ├── types.py               # Data structures
    ├── detection.py           # Anomaly detection rules
    ├── analysis.py            # Root cause inference
    ├── recommendation.py       # Fix suggestions
    ├── monitoring.py          # PyTorch metric collection
    ├── cli.py                 # CLI reporting
    └── __init__.py            # Package exports
```

---

## 🔧 How It Works

### Backend Flow
1. **Server** spawns FastAPI on port 5050
2. **Training thread** runs demo MLP with synthetic dataset
3. **Every batch**:
   - Compute metrics (loss, grad norm, learning rate)
   - Run detection (FailureDetector)
   - If issue: analyze root cause + generate recommendations
   - Stream via WebSocket to dashboard (80ms updates)

### Dashboard Flow
1. **Connect** via WebSocket to `/ws`
2. **Display** live charts (Chart.js)
3. **Highlight** anomaly points in red
4. **Show** detection panels + recommendations
5. **Control** learning rate from UI

### Detection Rules
- **Exploding Gradients**: `grad_norm > 10.0` or rapid growth
- **Vanishing Gradients**: `grad_norm < 1e-7`
- **Loss Divergence**: `loss > 2x initial loss`
- **Loss Plateau**: `loss unchanged for 50+ batches`
- **NaN Loss**: `math.isnan(loss) or math.isinf(loss)`
- **Overfitting**: `val_loss > train_loss * 1.5`

---

## 💻 Integration with Your Training

Import the core detection + analysis into your code:

```python
import torch.nn as nn
import torch.optim as optim
from mlcopilot import FailureDetector, RootCauseAnalyzer, RecommendationEngine
from mlcopilot import MetricSnapshot

# Your model and optimizer
model = nn.Linear(10, 2)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize detectors
detector = FailureDetector()
analyzer = RootCauseAnalyzer()
recommender = RecommendationEngine()

# Training loop
for epoch in range(10):
    for batch in range(5):
        # Your training step
        loss = torch.rand(1).item()
        
        # Get grad norm
        grad_norm = sum(p.grad.norm(2).item()**2 for p in model.parameters() if p.grad)**0.5
        
        # Create metric snapshot
        snapshot = MetricSnapshot(
            epoch=epoch, batch=batch, loss=loss,
            grad_norm=grad_norm, learning_rate=0.001,
            param_mean=0.0, param_std=0.1, param_max=0.5,
            timestamp=time.time()
        )
        
        # Store metrics
        metrics.append(snapshot)
        
        # Check for issues every 10 batches
        if batch % 10 == 0 and len(metrics) >= 5:
            detections = detector.detect_all(metrics)
            if detections:
                det = detections[0]
                diagnosis = analyzer.analyze(det, {}, {})
                recs = recommender.generate(diagnosis)
                print(f"Issue: {det.description}")
                for rec in recs:
                    print(f"  → {rec.action}: {rec.reasoning}")
```

---

## 🔬 Detectable Issues & Examples

### Exploding Gradients
- **Sign**: Grad norm suddenly spikes 10x+
- **Cause**: Learning rate too high, poor initialization
- **Fix**: Reduce LR, add gradient clipping, use batch norm

### Vanishing Gradients
- **Sign**: Grad norm drops to near-zero
- **Cause**: Network too deep, saturating activations (sigmoid/tanh)
- **Fix**: Use ReLU, add residual connections, better init

### Loss Divergence
- **Sign**: Loss increases instead of decreasing
- **Cause**: LR too high, bad batch, numerical instability
- **Fix**: Reduce LR, check data, normalize inputs

### Loss Plateau
- **Sign**: Loss unchanged for many batches
- **Cause**: Bad hyperparameters, stuck in local minimum
- **Fix**: Increase LR decay, use different optimizer

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│  Browser: http://localhost:5050         │
│  ├─ Dashboard (HTML/CSS/JS)             │
│  ├─ Live Charts (Chart.js)              │
│  └─ WebSocket client                    │
└──────────────────┬──────────────────────┘
                   │ WebSocket /ws
                   ↓
┌─────────────────────────────────────────┐
│  FastAPI Server (python-backend/server.py)
│  ├─ POST /start       → spawn training  │
│  ├─ POST /stop        → stop training   │
│  ├─ POST /set_lr      → update LR       │
│  ├─ GET  /status      → current state   │
│  └─ WS   /ws          → stream metrics  │
└──────────────────┬──────────────────────┘
                   │
                   ↓
      ┌────────────────────────┐
      │  Training Thread       │
      │  ├─ Demo MLP model     │
      │  ├─ Synthetic data     │
      │  └─ Metrics logging    │
      └────────────────────────┘
                   │
                   ↓
      ┌────────────────────────┐
      │  MLCopilot Core        │
      │  ├─ Detection          │
      │  ├─ Analysis           │
      │  └─ Recommendations    │
      └────────────────────────┘
```

---

## 📊 Tech Stack

| Layer | Tech |
|-------|------|
| **Backend** | Python, FastAPI, WebSocket, PyTorch |
| **Frontend** | Vanilla HTML/CSS/JS, Chart.js (no frameworks) |
| **ML Core** | PyTorch, NumPy, Rule-based detection/analysis |
| **Extension** | TypeScript, VS Code API *(optional)* |

---

## 🎓 Files Matter

| File | Purpose |
|------|---------|
| `detection.py` | Rule-based anomaly detection (~300 lines) |
| `analysis.py` | Root cause reasoning (~200 lines) |
| `recommendation.py` | Fix suggestions with code (~200 lines) |
| `types.py` | Data structures (MetricSnapshot, Diagnosis, etc.) |
| `monitoring.py` | PyTorch integration (hooks, metric collection) |
| `server.py` | FastAPI backend + board serving (~240 lines) |
| `app.js` | WebSocket client + Chart.js UI (~200 lines) |

**Total production code: ~1500 lines** — clean, minimal, powerful.

---

## 🚦 Demo Scenarios

### Scenario 1: Normal Training
- Model converges smoothly
- No alerts
- Dashboard shows steady loss decrease

### Scenario 2: High Learning Rate (will detect)
- Gradient norms spike after a few batches
- **CRITICAL** badge
- Recommendations: Reduce LR by 10x

### Scenario 3: Deep Network (will detect)
- Vanishing gradients after many layers
- **MEDIUM** warning
- Recommendations: Add residual connections, use ReLU

---

## 📝 License

MIT

---

**MLCopilot Labs** — For the hackathon. Built clean. Build fast. 🚀


---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd MLCopilot

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import torch.nn as nn
import torch.optim as optim
from main import MLCopilot

# Your model and optimizer
model = YourModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize MLCopilot
copilot = MLCopilot(model, optimizer)
copilot.start()

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(batch)
        
        # Log and check for issues
        if copilot.log_and_check(loss):
            break  # Issue detected - recommendations displayed
    
    copilot.log_epoch_end()

copilot.stop()
```

### Run Demo

```bash
# See MLCopilot detect a failing training scenario
python examples/failing_training.py

# See MLCopilot monitor a healthy training scenario
python examples/normal_training.py
```

---

## 📁 Project Structure

```
mlcopilot/
├── mlcopilot/
│   ├── __init__.py           # Package exports
│   ├── types.py              # Data structures and enums
│   ├── monitoring.py         # Training metric collection
│   ├── detection.py          # Anomaly detection
│   ├── analysis.py           # Root cause analysis
│   ├── recommendation.py     # Fix suggestions
│   └── cli.py                # Terminal output formatting
├── examples/
│   ├── failing_training.py   # Demo: intentional failure
│   └── normal_training.py    # Demo: healthy training
├── main.py                   # High-level API
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

---

## 🔧 How It Works

### 1. **Monitoring Layer**
- Attaches PyTorch hooks to capture gradients
- Logs loss, gradient norms, learning rate, parameter statistics
- Stores metrics in circular buffer (last 1000 batches)

### 2. **Detection Layer**
- Runs rule-based detectors on collected metrics
- Each detector returns confidence score + severity
- Detectors include:
  - `detect_exploding_gradients()` - grad_norm > threshold
  - `detect_vanishing_gradients()` - grad_norm < 1e-7
  - `detect_loss_divergence()` - loss > 2x initial loss
  - `detect_loss_plateau()` - loss unchanged for 50+ batches
  - `detect_nan_loss()` - NaN or Inf in loss
  - `detect_overfitting()` - val_loss >> train_loss

### 3. **Analysis Layer**
- Infers root cause using expert rules
- Considers:
  - Model architecture (depth, normalization)
  - Optimizer config (learning rate, momentum)
  - Training context (batch number, loss history)
- Categories:
  - Hyperparameter issues
  - Model architecture issues
  - Optimization issues
  - Data issues
  - Numerical instability

### 4. **Recommendation Layer**
- Generates specific, actionable fixes
- Includes:
  - Current vs suggested values
  - Code examples
  - Reasoning
  - Expected impact
- Prioritized by severity (Critical → High → Medium → Low)

### 5. **CLI Reporter**
- Formats output with colors and emojis
- Displays:
  - Detection results with metrics
  - Diagnosis with reasoning
  - Recommendations with code
- Clean, demo-ready output

---

## 🎓 Example Output

When training with a learning rate that's too high:

```
======================================================================
🚨 TRAINING FAILURE DETECTED
======================================================================

Issue: Gradient norm (157.30) exceeds threshold (10.00)
Type: Exploding Gradients
Severity: CRITICAL
Confidence: 95%
Detected at: Epoch 0, Batch 3

📊 Metrics at Detection
----------------------------------------------------------------------
  • Loss: 245873728.000000
  • Gradient Norm: 157.301453
  • Learning Rate: 5.00e-01
  • Param Mean: 0.124523
  • Param Std: 0.453621

🔍 ROOT CAUSE ANALYSIS
======================================================================

Category: Hyperparameter
Primary Cause: Learning rate too high

Reasoning:
The learning rate (0.5) is very high, causing weight updates to 
overshoot. Large learning rates can cause gradients to explode as 
the optimizer takes steps that are too large in parameter space.

Contributing Factors:
  • Learning rate (0.5) exceeds recommended threshold
  • Gradient norm reached 157.30

💡 RECOMMENDATIONS
======================================================================

3 actionable fix(es) suggested:

[1] 🔴 CRITICAL - Reduce Learning Rate
    Category: Hyperparameter
    Current:  0.5
    Suggested: 0.05

    High learning rate is causing unstable weight updates. Reducing 
    it by 10x will help stabilize training.

    Code Example:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    Expected Impact: Should immediately stabilize gradient norms and 
    prevent divergence

[2] 🟠 HIGH - Enable Gradient Clipping
    Category: Optimization
    Current:  None
    Suggested: max_norm=1.0

    Gradient clipping prevents gradients from growing too large...
```

---

## 🧪 Detection Thresholds

Configurable thresholds in `types.py`:

```python
class DetectionThresholds:
    EXPLODING_GRAD_THRESHOLD = 10.0
    VANISHING_GRAD_THRESHOLD = 1e-7
    LOSS_DIVERGENCE_MULTIPLIER = 2.0
    LOSS_PLATEAU_THRESHOLD = 0.001
    LOSS_PLATEAU_WINDOW = 50
    HIGH_LR_THRESHOLD = 0.1
    OVERFITTING_GAP_THRESHOLD = 0.5
```

---

## 🎯 Use Cases

### Hackathon / Demo
- Impressive visual output for presentations
- Fast setup (3 lines of code integration)
- Educational explanations

### Research / Experimentation
- Catch training issues early
- Learn from recommendations
- Save time debugging

### Production Training
- Monitor long-running jobs
- Get alerts on instability
- Automatic diagnosis

---

## 🔮 Future Extensions

- [ ] Support for TensorFlow/JAX
- [ ] Web dashboard (FastAPI + React)
- [ ] Statistical anomaly detection (Isolation Forest)
- [ ] LLM integration for complex reasoning
- [ ] Distributed training support
- [ ] Integration with Weights & Biases / MLflow
- [ ] Auto-hyperparameter tuning suggestions
- [ ] Email/Slack alerting

---

## 🏗️ Architecture Design

**Core Philosophy**: Simple, modular, hackathon-ready

- **No overengineering**: Each file has ONE clear purpose
- **No cloud dependencies**: Fully local
- **Minimal dependencies**: torch + numpy (+ optional rich)
- **Production patterns**: Clean separation of concerns

**Pipeline Flow**:
```
Monitor → Detect → Analyze → Recommend → Report
```

---

## 🤝 Contributing

This is a hackathon prototype. For production use:
1. Add comprehensive testing
2. Implement configuration system
3. Add data persistence
4. Expand detector library
5. Improve confidence scoring

---

## 📄 License

MIT License - Feel free to use in your projects!

---

## 👥 Authors

Built for ML engineers who want intelligent training monitoring without cloud overhead.

---

## 🙏 Acknowledgments

Inspired by best practices from:
- PyTorch Lightning
- TensorBoard
- Weights & Biases
- Deep learning community wisdom

---

**MLCopilot** - Because training shouldn't be trial and error. 🚀
