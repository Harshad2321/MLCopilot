# MLCopilot – Real-Time ML Training Monitor

<p align="center">
  <img src="icon.png" alt="MLCopilot Logo" width="128" />
</p>

**Detect training failures. Get root cause analysis. Fix your model — live.**

MLCopilot is a VS Code extension that monitors your PyTorch training in real time, detects issues like exploding gradients, loss divergence, and vanishing gradients, then tells you exactly why it's happening and how to fix it.

---

## Features

| Feature | Description |
|---|---|
| **Live Loss Curve** | Real-time loss chart with anomaly highlighting |
| **Gradient Monitoring** | Track gradient norms, detect explosions and vanishing |
| **Root Cause Analysis** | Rule-based reasoning explains *why* training failed |
| **Fix Recommendations** | Actionable code suggestions to fix detected issues |
| **Live LR Control** | Change learning rate mid-training from the UI |
| **Status Badges** | Healthy / Warning / Critical at a glance |

## Quick Start

1. Open any ML project in VS Code
2. Run command: **`MLCopilot: Start Training Monitor`**
3. The dashboard opens automatically at `http://localhost:5050`
4. Click **Start Training** to begin a demo run
5. Watch live metrics, detections, and recommendations stream in

## Commands

| Command | Description |
|---|---|
| `MLCopilot: Start Training Monitor` | Starts the backend server and opens the dashboard |
| `MLCopilot: Stop Training Monitor` | Stops the backend server |
| `MLCopilot: Open Dashboard` | Opens the web dashboard in your browser |

## Sidebar

MLCopilot adds an **Activity Bar** icon. Click it to access quick controls:
- ▶ Start Monitor
- ⏹ Stop Monitor
- 🌐 Open Dashboard

## Requirements

- **Python 3.9+** with `pip`
- **PyTorch** installed
- Python packages: `fastapi`, `uvicorn`, `websockets`, `numpy`

Install backend dependencies:
```bash
pip install fastapi uvicorn[standard] websockets numpy torch
```

## How It Works

```
VS Code Extension
    │
    ├── Spawns Python FastAPI server (localhost:5050)
    ├── Opens browser dashboard
    │
    └── Dashboard ← WebSocket → Server
                                   │
                                   ├── Runs training in background thread
                                   ├── Streams metrics every batch
                                   ├── Detects anomalies (FailureDetector)
                                   ├── Analyzes root cause (RootCauseAnalyzer)
                                   └── Generates recommendations (RecommendationEngine)
```

## Detectable Issues

- 💥 **Exploding Gradients** – gradient norms spiking
- 📉 **Vanishing Gradients** – gradient norms near zero
- 📈 **Loss Divergence** – loss increasing instead of decreasing
- 🔁 **Loss Plateau** – training stuck, no improvement
- ❌ **NaN Loss** – training completely collapsed
- 📊 **Overfitting** – train/val gap growing

## Tech Stack

- **Backend:** Python, FastAPI, WebSocket
- **Frontend:** Vanilla HTML/CSS/JS, Chart.js
- **Extension:** TypeScript, VS Code API
- **ML Core:** PyTorch, NumPy

## License

MIT

---

**Built by MLCopilot Labs**
