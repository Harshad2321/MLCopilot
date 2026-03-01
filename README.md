# MLCopilot 🤖

**Real-Time ML Training Failure Detection & Fix Recommendation Engine**

MLCopilot is an intelligent monitoring system that integrates into PyTorch training loops to detect training instability early, performs root cause analysis, and generates actionable ML-specific recommendations.

---

## 🎯 Features

- **🔍 Real-Time Monitoring**: Non-invasive hooks into PyTorch training
- **🚨 Intelligent Detection**: Rule-based anomaly detection for common training failures
- **🧠 Root Cause Analysis**: Expert-system reasoning to identify why failures occur
- **💡 Actionable Recommendations**: Specific fixes with code examples
- **📊 Beautiful CLI Output**: Clear, colorful terminal reports

### Detects:
- ⚠️ Exploding gradients
- � Vanishing gradients
- 💥 Loss divergence
- 📊 Loss plateau
- ❌ NaN/Inf losses
- 📈 Overfitting (train/val gap)

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
