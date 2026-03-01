"""
MLCopilot Recommendation Engine
Generates actionable recommendations to fix detected training issues.
"""

from typing import List
import math

from .types import Diagnosis, Recommendation, Priority, AnomalyType, CauseCategory


class RecommendationEngine:
    """
    Generates structured, actionable recommendations based on diagnosis.
    """
    
    def generate(self, diagnosis: Diagnosis) -> List[Recommendation]:
        """
        Generate recommendations based on diagnosis.
        
        Args:
            diagnosis: Diagnosis from root cause analyzer
        
        Returns:
            List of recommendations, sorted by priority
        """
        anomaly_type = diagnosis.detection.anomaly_type
        
        # Route to appropriate recommendation generator
        generator_map = {
            AnomalyType.EXPLODING_GRADIENTS: self._recommend_exploding_gradients,
            AnomalyType.VANISHING_GRADIENTS: self._recommend_vanishing_gradients,
            AnomalyType.LOSS_DIVERGENCE: self._recommend_loss_divergence,
            AnomalyType.LOSS_PLATEAU: self._recommend_loss_plateau,
            AnomalyType.NAN_LOSS: self._recommend_nan_loss,
            AnomalyType.OVERFITTING: self._recommend_overfitting,
        }
        
        generator = generator_map.get(anomaly_type)
        
        if generator:
            recommendations = generator(diagnosis)
        else:
            recommendations = self._recommend_generic(diagnosis)
        
        # Sort by priority
        priority_order = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3
        }
        recommendations.sort(key=lambda r: priority_order[r.priority])
        
        return recommendations
    
    def _recommend_exploding_gradients(self, diagnosis: Diagnosis) -> List[Recommendation]:
        """Generate recommendations for exploding gradients."""
        recommendations = []
        
        lr = diagnosis.optimizer_context.get('learning_rate', 0.0)
        has_normalization = diagnosis.model_context.get('has_normalization', False)
        
        # Recommendation 1: Reduce learning rate (if high)
        if lr > 0.01:
            suggested_lr = lr / 10
            recommendations.append(Recommendation(
                priority=Priority.CRITICAL,
                category="Hyperparameter",
                action="Reduce Learning Rate",
                current_value=f"{lr}",
                suggested_value=f"{suggested_lr}",
                reasoning=(
                    "High learning rate is causing unstable weight updates. "
                    "Reducing it by 10x will help stabilize training."
                ),
                code_example=f"optimizer = torch.optim.Adam(model.parameters(), lr={suggested_lr})",
                expected_impact="Should immediately stabilize gradient norms and prevent divergence"
            ))
        
        # Recommendation 2: Add gradient clipping
        recommendations.append(Recommendation(
            priority=Priority.HIGH,
            category="Optimization",
            action="Enable Gradient Clipping",
            current_value="None",
            suggested_value="max_norm=1.0",
            reasoning=(
                "Gradient clipping prevents gradients from growing too large, "
                "protecting against exploding gradients while preserving training dynamics."
            ),
            code_example=(
                "# Add before optimizer.step():\n"
                "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)"
            ),
            expected_impact="Will cap gradient norms at 1.0, preventing explosions"
        ))
        
        # Recommendation 3: Add normalization (if missing)
        if not has_normalization:
            recommendations.append(Recommendation(
                priority=Priority.HIGH,
                category="Architecture",
                action="Add Batch Normalization",
                current_value="No normalization layers",
                suggested_value="Add BatchNorm after each layer",
                reasoning=(
                    "Batch normalization stabilizes activations throughout the network, "
                    "preventing them from growing unbounded and causing gradient explosions."
                ),
                code_example=(
                    "# Example: Add between linear layers\n"
                    "self.fc1 = nn.Linear(input_dim, hidden_dim)\n"
                    "self.bn1 = nn.BatchNorm1d(hidden_dim)\n"
                    "# Forward: x = self.bn1(F.relu(self.fc1(x)))"
                ),
                expected_impact="Will normalize activations and stabilize gradient flow"
            ))
        
        # Recommendation 4: Check initialization
        recommendations.append(Recommendation(
            priority=Priority.MEDIUM,
            category="Architecture",
            action="Use Proper Weight Initialization",
            current_value="Unknown",
            suggested_value="Xavier/He initialization",
            reasoning=(
                "Proper initialization prevents initial activations from being too large or small."
            ),
            code_example=(
                "# Apply to your model:\n"
                "def init_weights(m):\n"
                "    if isinstance(m, nn.Linear):\n"
                "        torch.nn.init.xavier_uniform_(m.weight)\n"
                "        m.bias.data.fill_(0.01)\n"
                "model.apply(init_weights)"
            ),
            expected_impact="Better initial gradient magnitudes"
        ))
        
        return recommendations
    
    def _recommend_vanishing_gradients(self, diagnosis: Diagnosis) -> List[Recommendation]:
        """Generate recommendations for vanishing gradients."""
        recommendations = []
        
        num_layers = diagnosis.model_context.get('num_layers', 0)
        
        # Recommendation 1: Use ReLU activation
        recommendations.append(Recommendation(
            priority=Priority.HIGH,
            category="Architecture",
            action="Switch to ReLU or LeakyReLU Activation",
            current_value="Possibly sigmoid/tanh",
            suggested_value="ReLU",
            reasoning=(
                "Sigmoid and tanh activations saturate and cause vanishing gradients. "
                "ReLU doesn't saturate for positive values, maintaining gradient flow."
            ),
            code_example=(
                "# Replace sigmoid/tanh with:\n"
                "activation = nn.ReLU()\n"
                "# Or use LeakyReLU for negative values:\n"
                "activation = nn.LeakyReLU(0.01)"
            ),
            expected_impact="Will prevent gradient saturation in deep layers"
        ))
        
        # Recommendation 2: Add residual connections (if deep)
        if num_layers > 10:
            recommendations.append(Recommendation(
                priority=Priority.HIGH,
                category="Architecture",
                action="Add Residual Connections",
                current_value="Sequential architecture",
                suggested_value="ResNet-style skip connections",
                reasoning=(
                    f"Your network has {num_layers} layers. Residual connections create "
                    "gradient highways that bypass deep layers, preventing vanishing gradients."
                ),
                code_example=(
                    "# Add skip connections:\n"
                    "class ResidualBlock(nn.Module):\n"
                    "    def forward(self, x):\n"
                    "        residual = x\n"
                    "        out = self.layers(x)\n"
                    "        return out + residual  # Skip connection"
                ),
                expected_impact="Gradients can flow directly through skip connections"
            ))
        
        # Recommendation 3: Better initialization
        recommendations.append(Recommendation(
            priority=Priority.MEDIUM,
            category="Architecture",
            action="Use He Initialization",
            current_value="Unknown",
            suggested_value="He/Kaiming initialization for ReLU",
            reasoning=(
                "He initialization is designed for ReLU activations and helps maintain "
                "appropriate gradient magnitudes."
            ),
            code_example=(
                "def init_weights(m):\n"
                "    if isinstance(m, nn.Linear):\n"
                "        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')\n"
                "model.apply(init_weights)"
            ),
            expected_impact="Better initial gradient flow"
        ))
        
        return recommendations
    
    def _recommend_loss_divergence(self, diagnosis: Diagnosis) -> List[Recommendation]:
        """Generate recommendations for loss divergence."""
        recommendations = []
        
        lr = diagnosis.optimizer_context.get('learning_rate', 0.0)
        
        # Recommendation 1: Reduce learning rate
        if lr > 0.01:
            suggested_lr = lr / 10
        else:
            suggested_lr = 0.001
        
        recommendations.append(Recommendation(
            priority=Priority.CRITICAL,
            category="Hyperparameter",
            action="Reduce Learning Rate",
            current_value=f"{lr}",
            suggested_value=f"{suggested_lr}",
            reasoning=(
                "Loss divergence indicates training instability. "
                "A lower learning rate will help the optimizer converge more reliably."
            ),
            code_example=f"optimizer = torch.optim.Adam(model.parameters(), lr={suggested_lr})",
            expected_impact="Should stop loss from diverging and enable stable convergence"
        ))
        
        # Recommendation 2: Check data preprocessing
        recommendations.append(Recommendation(
            priority=Priority.HIGH,
            category="Data",
            action="Normalize Input Data",
            current_value="Unknown",
            suggested_value="Standardize to mean=0, std=1",
            reasoning=(
                "Unnormalized data can cause loss divergence. "
                "Standardizing inputs helps optimization."
            ),
            code_example=(
                "# Normalize your data:\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "scaler = StandardScaler()\n"
                "X_train = scaler.fit_transform(X_train)\n"
                "X_val = scaler.transform(X_val)"
            ),
            expected_impact="More stable loss landscape"
        ))
        
        # Recommendation 3: Use learning rate scheduling
        recommendations.append(Recommendation(
            priority=Priority.MEDIUM,
            category="Optimization",
            action="Add Learning Rate Scheduler",
            current_value="Fixed LR",
            suggested_value="ReduceLROnPlateau or CosineAnnealing",
            reasoning=(
                "Learning rate scheduling can help navigate difficult parts of the loss landscape."
            ),
            code_example=(
                "scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(\n"
                "    optimizer, mode='min', factor=0.5, patience=5\n"
                ")\n"
                "# After validation:\n"
                "scheduler.step(val_loss)"
            ),
            expected_impact="Adaptive learning rate for better convergence"
        ))
        
        return recommendations
    
    def _recommend_loss_plateau(self, diagnosis: Diagnosis) -> List[Recommendation]:
        """Generate recommendations for loss plateau."""
        recommendations = []
        
        lr = diagnosis.optimizer_context.get('learning_rate', 0.0)
        
        # Recommendation 1: Increase learning rate (if too small)
        if lr < 1e-5:
            suggested_lr = lr * 100
            recommendations.append(Recommendation(
                priority=Priority.HIGH,
                category="Hyperparameter",
                action="Increase Learning Rate",
                current_value=f"{lr:.2e}",
                suggested_value=f"{suggested_lr:.2e}",
                reasoning=(
                    "Learning rate is too small for meaningful progress. "
                    "Increasing it will allow larger parameter updates."
                ),
                code_example=f"optimizer = torch.optim.Adam(model.parameters(), lr={suggested_lr})",
                expected_impact="Faster convergence and escape from plateau"
            ))
        else:
            # Use learning rate warmup/cycling
            recommendations.append(Recommendation(
                priority=Priority.HIGH,
                category="Optimization",
                action="Use Learning Rate Warm Restart",
                current_value="Fixed LR",
                suggested_value="Cyclic LR or Warm Restarts",
                reasoning=(
                    "Temporary learning rate increases can help escape local minima."
                ),
                code_example=(
                    "scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(\n"
                    "    optimizer, T_0=10, T_mult=2\n"
                    ")\n"
                    "# Call after each batch:\n"
                    "scheduler.step()"
                ),
                expected_impact="May help escape local minima"
            ))
        
        # Recommendation 2: Add momentum (if not using)
        optimizer_type = diagnosis.optimizer_context.get('optimizer_type', '')
        if optimizer_type == 'SGD':
            recommendations.append(Recommendation(
                priority=Priority.MEDIUM,
                category="Optimization",
                action="Add Momentum to SGD",
                current_value="SGD without momentum",
                suggested_value="momentum=0.9",
                reasoning=(
                    "Momentum helps escape plateaus by accumulating velocity."
                ),
                code_example="optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)",
                expected_impact="Better navigation of flat regions"
            ))
        
        # Recommendation 3: Check if model has enough capacity
        recommendations.append(Recommendation(
            priority=Priority.MEDIUM,
            category="Architecture",
            action="Verify Model Capacity",
            current_value="Current architecture",
            suggested_value="May need more layers/neurons",
            reasoning=(
                "If loss plateaus at high values, the model may not have enough capacity "
                "to fit the data."
            ),
            code_example=(
                "# Consider increasing hidden dimensions or adding layers\n"
                "# Example: 128 -> 256 neurons"
            ),
            expected_impact="Better representational power if underfitting"
        ))
        
        return recommendations
    
    def _recommend_nan_loss(self, diagnosis: Diagnosis) -> List[Recommendation]:
        """Generate recommendations for NaN loss."""
        recommendations = []
        
        # Recommendation 1: Reduce learning rate drastically
        lr = diagnosis.optimizer_context.get('learning_rate', 0.0)
        suggested_lr = min(0.0001, lr / 100) if lr > 0 else 0.0001
        
        recommendations.append(Recommendation(
            priority=Priority.CRITICAL,
            category="Hyperparameter",
            action="Drastically Reduce Learning Rate",
            current_value=f"{lr}",
            suggested_value=f"{suggested_lr}",
            reasoning=(
                "NaN loss indicates severe numerical instability. "
                "Starting fresh with a much lower learning rate is essential."
            ),
            code_example=f"optimizer = torch.optim.Adam(model.parameters(), lr={suggested_lr})",
            expected_impact="Prevent numerical overflow"
        ))
        
        # Recommendation 2: Add gradient clipping
        recommendations.append(Recommendation(
            priority=Priority.CRITICAL,
            category="Optimization",
            action="Add Aggressive Gradient Clipping",
            current_value="None",
            suggested_value="max_norm=0.5",
            reasoning=(
                "Gradient clipping is essential to prevent NaN propagation."
            ),
            code_example="torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)",
            expected_impact="Prevents extreme gradient values"
        ))
        
        # Recommendation 3: Check data for NaN/Inf
        recommendations.append(Recommendation(
            priority=Priority.HIGH,
            category="Data",
            action="Validate Input Data",
            current_value="Unknown",
            suggested_value="Check for NaN/Inf in data",
            reasoning=(
                "NaN in training data will propagate through the network."
            ),
            code_example=(
                "# Check data:\n"
                "assert not torch.isnan(inputs).any(), 'NaN in inputs'\n"
                "assert not torch.isinf(inputs).any(), 'Inf in inputs'"
            ),
            expected_impact="Ensure clean data pipeline"
        ))
        
        return recommendations
    
    def _recommend_overfitting(self, diagnosis: Diagnosis) -> List[Recommendation]:
        """Generate recommendations for overfitting."""
        recommendations = []
        
        # Recommendation 1: Add dropout
        recommendations.append(Recommendation(
            priority=Priority.HIGH,
            category="Regularization",
            action="Add Dropout Layers",
            current_value="No dropout",
            suggested_value="dropout=0.3-0.5",
            reasoning=(
                "Dropout randomly disables neurons during training, preventing co-adaptation "
                "and reducing overfitting."
            ),
            code_example=(
                "# Add dropout between layers:\n"
                "self.dropout = nn.Dropout(0.3)\n"
                "# In forward:\n"
                "x = self.dropout(F.relu(self.fc1(x)))"
            ),
            expected_impact="Reduces train/val gap by preventing overfitting"
        ))
        
        # Recommendation 2: Add weight decay
        recommendations.append(Recommendation(
            priority=Priority.HIGH,
            category="Regularization",
            action="Add Weight Decay (L2 Regularization)",
            current_value="No weight decay",
            suggested_value="weight_decay=1e-4",
            reasoning=(
                "Weight decay penalizes large weights, encouraging simpler models."
            ),
            code_example="optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)",
            expected_impact="Encourages simpler, more generalizable models"
        ))
        
        # Recommendation 3: Data augmentation
        recommendations.append(Recommendation(
            priority=Priority.MEDIUM,
            category="Data",
            action="Use Data Augmentation",
            current_value="No augmentation",
            suggested_value="Augment training data",
            reasoning=(
                "Data augmentation artificially increases dataset diversity, "
                "helping the model generalize better."
            ),
            code_example=(
                "# For images:\n"
                "from torchvision import transforms\n"
                "transform = transforms.Compose([\n"
                "    transforms.RandomHorizontalFlip(),\n"
                "    transforms.RandomRotation(10),\n"
                "])"
            ),
            expected_impact="Better generalization to validation data"
        ))
        
        # Recommendation 4: Early stopping
        recommendations.append(Recommendation(
            priority=Priority.MEDIUM,
            category="Training Strategy",
            action="Implement Early Stopping",
            current_value="Training until completion",
            suggested_value="Stop when val_loss stops improving",
            reasoning=(
                "Early stopping prevents training beyond the point of best generalization."
            ),
            code_example=(
                "# Track best validation loss:\n"
                "if val_loss < best_val_loss:\n"
                "    best_val_loss = val_loss\n"
                "    patience_counter = 0\n"
                "    torch.save(model.state_dict(), 'best_model.pt')\n"
                "else:\n"
                "    patience_counter += 1\n"
                "    if patience_counter >= patience:\n"
                "        break  # Stop training"
            ),
            expected_impact="Prevents overfitting by stopping at optimal point"
        ))
        
        return recommendations
    
    def _recommend_generic(self, diagnosis: Diagnosis) -> List[Recommendation]:
        """Generic recommendations for unknown issues."""
        return [
            Recommendation(
                priority=Priority.MEDIUM,
                category="General",
                action="Review Training Configuration",
                current_value="Unknown",
                suggested_value="Check all hyperparameters",
                reasoning="An issue was detected but specific recommendations are unavailable.",
                code_example="# Review: learning rate, batch size, regularization",
                expected_impact="Unknown"
            )
        ]
