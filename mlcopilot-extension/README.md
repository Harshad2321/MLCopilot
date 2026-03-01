# MLCopilot

Real-Time ML Training Monitor for VS Code.

## Features

- **Automatic Detection**: Automatically starts monitoring when you run Python ML scripts
- **Live Dashboard**: Real-time loss curves and gradient norm visualization
- **Failure Detection**: Detects exploding/vanishing gradients, loss divergence, NaN losses
- **Root Cause Analysis**: Explains why issues occur
- **Actionable Recommendations**: Code-level fixes for detected problems

## Requirements

- Python 3.8+
- Python extension for VS Code
- Required Python packages:
  ```
  pip install fastapi uvicorn torch numpy websockets
  ```

## Usage

1. Install the extension
2. Open any ML project
3. Run your Python training script
4. The dashboard opens automatically and streams live metrics

## How It Works

When you execute a Python file that appears to be an ML training script:
1. The backend server starts automatically
2. Your browser opens to the live dashboard
3. Metrics stream in real-time via WebSocket
4. Issues are detected and recommendations provided

## Commands

- `MLCopilot: Open Dashboard` - Manually open the monitoring dashboard

## Extension Settings

This extension uses the Python extension's configured interpreter.

## License

MIT
