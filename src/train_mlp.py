"""
FightPredict - MLP Neural Network Training
============================================
Improved MLP with LeakyReLU/GELU activation to prevent dying neurons.

Features:
- LeakyReLU/ELU/GELU activation (prevents dying ReLU problem)
- Optional residual connections for deeper networks
- Dead neuron monitoring during training
- Optuna Bayesian hyperparameter optimization
- Class imbalance handling via weighted loss
- Threshold tuning for optimal classification

Usage:
    python src/train_mlp.py
    python src/train_mlp.py --quick  # Skip Optuna, use good defaults

Output:
    - models/mlp_model.pt
    - models/mlp_model_with_odds.pt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Literal

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# Sklearn imports
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    precision_score, recall_score, balanced_accuracy_score
)

# Optuna for hyperparameter tuning
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not installed. Run: pip install optuna")

import warnings
warnings.filterwarnings('ignore')

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# =============================================================================
# MLP MODEL ARCHITECTURE
# =============================================================================

class ImprovedMLPClassifier(nn.Module):
    """
    Improved MLP for UFC Fight Prediction

    Features:
    - LeakyReLU/ELU/GELU instead of ReLU (prevents dying neurons)
    - Optional residual connections for deeper networks
    - Layer normalization option (more stable than BatchNorm for small batches)
    - Configurable activation function
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        activation: Literal["leaky_relu", "elu",
                            "selu", "gelu"] = "leaky_relu",
        leaky_slope: float = 0.01,
        use_residual: bool = False
    ):
        super(ImprovedMLPClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation_name = activation
        self.leaky_slope = leaky_slope
        self.use_residual = use_residual

        # Build layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Residual projections (if sizes don't match)
        self.residual_projections = nn.ModuleList()

        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            self.layers.append(nn.Linear(prev_size, hidden_size))

            # Normalization
            if use_batch_norm:
                self.norms.append(nn.BatchNorm1d(hidden_size))
            else:
                self.norms.append(nn.Identity())

            # Dropout
            self.dropouts.append(nn.Dropout(dropout_rate))

            # Residual projection if needed
            if use_residual and prev_size != hidden_size:
                self.residual_projections.append(
                    nn.Linear(prev_size, hidden_size))
            else:
                self.residual_projections.append(None)

            prev_size = hidden_size

        # Output layer
        self.output_layer = nn.Linear(prev_size, 1)

        # Activation function
        self.activation = self._get_activation(activation, leaky_slope)

        # Initialize weights properly
        self._init_weights()

    def _get_activation(self, name: str, leaky_slope: float) -> nn.Module:
        """Get activation function by name."""
        if name == "leaky_relu":
            return nn.LeakyReLU(negative_slope=leaky_slope)
        elif name == "elu":
            return nn.ELU(alpha=1.0)
        elif name == "selu":
            return nn.SELU()
        elif name == "gelu":
            return nn.GELU()
        else:
            return nn.LeakyReLU(negative_slope=0.01)

    def _init_weights(self):
        """Initialize weights using Kaiming initialization for LeakyReLU."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, a=self.leaky_slope, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Output layer - Xavier for sigmoid
        nn.init.xavier_normal_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (layer, norm, dropout) in enumerate(zip(self.layers, self.norms, self.dropouts)):
            identity = x

            x = layer(x)
            x = norm(x)
            x = self.activation(x)
            x = dropout(x)

            # Residual connection
            if self.use_residual and i > 0:
                proj = self.residual_projections[i]
                if proj is not None:
                    identity = proj(identity)
                if identity.shape == x.shape:
                    x = x + identity

        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Get probability predictions (sklearn-compatible)"""
        self.eval()
        with torch.no_grad():
            probs = self.forward(x).cpu().numpy().flatten()
        return np.column_stack([1 - probs, probs])

    def count_dead_neurons(self, x: torch.Tensor) -> Dict[str, float]:
        """Count dead neurons (outputting 0) for each layer."""
        self.eval()
        dead_counts = {}

        with torch.no_grad():
            for i, (layer, norm, dropout) in enumerate(zip(self.layers, self.norms, self.dropouts)):
                x = layer(x)
                x = norm(x)
                x = self.activation(x)

                # Count neurons that are always 0
                dead_pct = (x == 0).float().mean().item() * 100
                dead_counts[f'layer_{i}'] = dead_pct

                x = dropout(x)

        return dead_counts


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Alias for backward compatibility with old model files
UFCMLPClassifier = ImprovedMLPClassifier


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class EarlyStopping:
    """Early stopping with best model restoration."""

    def __init__(self, patience: int = 20, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_model_state = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu().clone()
                                     for k, v in model.state_dict().items()}
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu().clone()
                                     for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def restore_best(self, model: nn.Module):
        """Restore best model weights."""
        if self.best_model_state is not None:
            model.load_state_dict({k: v.to(DEVICE)
                                  for k, v in self.best_model_state.items()})


# =============================================================================
# MLP TRAINER
# =============================================================================

class MLPTrainer:
    """Trains MLP models with dead neuron monitoring."""

    def __init__(self, data_path: str = "data/processed", models_path: str = "models"):
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.scaler = None
        self.feature_names = None
        self.optimal_threshold = 0.5
        self.results = {}

    def load_data(self, include_odds: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """Load model-ready data."""
        if include_odds:
            filepath = self.data_path / "features_model_ready_with_odds.csv"
        else:
            filepath = self.data_path / "features_model_ready.csv"

        df = pd.read_csv(filepath)
        X = df.drop('target', axis=1)
        y = df['target']

        self.feature_names = X.columns.tolist()
        print(f"Loaded: {len(X):,} samples, {len(X.columns)} features")
        print(f"Class distribution: {y.value_counts().to_dict()}")

        return X, y

    def create_data_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 64,
        use_weighted_sampler: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """Create DataLoaders with optional weighted sampling."""

        X_train_t = torch.FloatTensor(X_train).to(DEVICE)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
        X_val_t = torch.FloatTensor(X_val).to(DEVICE)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(DEVICE)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)

        if use_weighted_sampler:
            class_counts = np.bincount(y_train.astype(int))
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[y_train.astype(int)]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        hidden_sizes: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3,
        activation: str = "leaky_relu",
        leaky_slope: float = 0.01,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        batch_size: int = 64,
        epochs: int = 200,
        patience: int = 30,
        use_weighted_sampler: bool = True,
        verbose: bool = True
    ) -> Tuple[nn.Module, Dict]:
        """Train MLP model."""

        n_features = X_train.shape[1]

        # Create model
        model = ImprovedMLPClassifier(
            input_size=n_features,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
            activation=activation,
            leaky_slope=leaky_slope,
            use_residual=len(hidden_sizes) > 2
        ).to(DEVICE)

        if verbose:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\nModel architecture:")
            print(f"  Hidden sizes: {hidden_sizes}")
            print(f"  Activation: {activation}")
            print(f"  Dropout: {dropout_rate}")
            print(f"  Total parameters: {total_params:,}")

        # Data loaders
        train_loader, val_loader = self.create_data_loaders(
            X_train, y_train, X_val, y_val,
            batch_size=batch_size,
            use_weighted_sampler=use_weighted_sampler
        )

        # Loss with class weighting
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        pos_weight = torch.tensor(
            [n_neg / n_pos], dtype=torch.float32).to(DEVICE)

        # Use BCELoss since our model outputs sigmoid
        criterion = nn.BCELoss(reduction='none')

        # Optimizer with weight decay (L2 regularization)
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=learning_rate/100
        )

        # Early stopping
        early_stopping = EarlyStopping(patience=patience)

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_auc': [],
            'learning_rate': [],
            'dead_neurons': []
        }

        if verbose:
            print(
                f"\nTraining for up to {epochs} epochs (patience={patience})...")

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)

                # Weighted BCE loss
                bce = criterion(outputs, y_batch)
                weights = torch.where(
                    y_batch == 1, pos_weight, torch.ones_like(y_batch))
                loss = (bce * weights).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            all_probs = []
            all_labels = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    bce = criterion(outputs, y_batch)
                    val_loss += bce.mean().item()

                    all_probs.extend(outputs.cpu().numpy().flatten())
                    all_labels.extend(y_batch.cpu().numpy().flatten())

            val_loss /= len(val_loader)

            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)
            val_preds = (all_probs >= 0.5).astype(int)
            val_accuracy = accuracy_score(all_labels, val_preds)
            val_auc = roc_auc_score(all_labels, all_probs)

            # Check dead neurons periodically
            if epoch % 10 == 0:
                sample_batch = torch.FloatTensor(X_train[:100]).to(DEVICE)
                dead_counts = model.count_dead_neurons(sample_batch)
                avg_dead = np.mean(list(dead_counts.values()))
                history['dead_neurons'].append(
                    {'epoch': epoch, 'avg_dead_pct': avg_dead, 'by_layer': dead_counts})

            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['val_auc'].append(val_auc)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # Learning rate scheduling
            scheduler.step()

            # Early stopping check
            if early_stopping(val_loss, model):
                if verbose:
                    print(f"\n   Early stopping at epoch {epoch+1}")
                break

            if verbose and (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}, val_auc={val_auc:.4f}")

        # Restore best model
        early_stopping.restore_best(model)

        history['final_epoch'] = epoch + 1
        history['best_val_loss'] = early_stopping.best_loss

        # Final dead neuron check
        sample_batch = torch.FloatTensor(X_train[:500]).to(DEVICE)
        final_dead = model.count_dead_neurons(sample_batch)
        history['final_dead_neurons'] = final_dead

        if verbose:
            print(f"\n   Final dead neuron rates: {final_dead}")

        return model, history

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50
    ) -> Dict:
        """Tune hyperparameters with Optuna, optimizing for low dead neurons too."""

        if not OPTUNA_AVAILABLE:
            print("⚠️ Optuna not available, using defaults")
            return self.get_default_params()

        print("\n" + "="*60)
        print("Hyperparameter Tuning with Optuna")
        print("="*60)

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            # Architecture
            n_layers = trial.suggest_int('n_layers', 2, 4)
            hidden_sizes = []
            prev_size = 256
            for i in range(n_layers):
                size = trial.suggest_categorical(
                    f'hidden_{i}', [64, 128, 256, 512])
                size = min(size, prev_size)
                hidden_sizes.append(size)
                prev_size = size

            # Training params
            dropout_rate = trial.suggest_float('dropout', 0.1, 0.4)
            learning_rate = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float(
                'weight_decay', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

            # Activation
            activation = trial.suggest_categorical(
                'activation', ['leaky_relu', 'elu', 'gelu'])
            leaky_slope = 0.01
            if activation == 'leaky_relu':
                leaky_slope = trial.suggest_float('leaky_slope', 0.01, 0.2)

            try:
                model, history = self.train_model(
                    X_train, y_train, X_val, y_val,
                    hidden_sizes=hidden_sizes,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    leaky_slope=leaky_slope,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    batch_size=batch_size,
                    epochs=100,
                    patience=20,
                    verbose=False
                )

                # Evaluate
                model.eval()
                X_val_t = torch.FloatTensor(X_val).to(DEVICE)
                with torch.no_grad():
                    y_prob = model(X_val_t).cpu().numpy().flatten()

                auc = roc_auc_score(y_val, y_prob)

                # Penalize high dead neuron rates
                dead_neurons = history.get('final_dead_neurons', {})
                avg_dead = np.mean(list(dead_neurons.values())
                                   ) if dead_neurons else 0

                # Combined score: AUC - penalty for dead neurons
                penalty = max(0, (avg_dead - 20) / 100)
                score = auc - penalty

                trial.set_user_attr('auc', auc)
                trial.set_user_attr('dead_neurons', avg_dead)

                return score

            except Exception as e:
                print(f"Trial failed: {e}")
                return 0.0

        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)

        def callback(study, trial):
            if trial.number % 10 == 0:
                print(f"   Trial {trial.number}: score={trial.value:.4f}, "
                      f"AUC={trial.user_attrs.get('auc', 0):.4f}, "
                      f"dead={trial.user_attrs.get('dead_neurons', 0):.1f}%")

        study.optimize(objective, n_trials=n_trials, callbacks=[
                       callback], show_progress_bar=True)

        print(f"\n✓ Best score: {study.best_value:.4f}")
        print(f"✓ Best params: {study.best_params}")

        # Convert to usable format
        best = study.best_params
        n_layers = best['n_layers']
        hidden_sizes = [best[f'hidden_{i}'] for i in range(n_layers)]

        return {
            'hidden_sizes': hidden_sizes,
            'dropout_rate': best['dropout'],
            'learning_rate': best['lr'],
            'weight_decay': best['weight_decay'],
            'batch_size': best['batch_size'],
            'activation': best['activation'],
            'leaky_slope': best.get('leaky_slope', 0.01)
        }

    def get_default_params(self) -> Dict:
        """Good default parameters based on common findings."""
        return {
            'hidden_sizes': [256, 128, 64],
            'dropout_rate': 0.25,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'batch_size': 64,
            'activation': 'leaky_relu',
            'leaky_slope': 0.05
        }

    def find_optimal_threshold(
        self,
        model: nn.Module,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = 'f1_macro'
    ) -> Tuple[float, pd.DataFrame]:
        """Find optimal classification threshold."""

        model.eval()
        X_val_t = torch.FloatTensor(X_val).to(DEVICE)

        with torch.no_grad():
            y_prob = model(X_val_t).cpu().numpy().flatten()

        thresholds = np.arange(0.30, 0.70, 0.01)
        results = []

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            if len(np.unique(y_pred)) < 2:
                continue

            results.append({
                'threshold': thresh,
                'accuracy': accuracy_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred),
                'f1_macro': f1_score(y_val, y_pred, average='macro'),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_val, y_pred),
            })

        df = pd.DataFrame(results)
        best_idx = df[metric].idxmax()
        optimal_threshold = df.iloc[best_idx]['threshold']

        print(f"\n   Optimal threshold ({metric}): {optimal_threshold:.3f}")
        print(f"   Metrics at optimal: acc={df.iloc[best_idx]['accuracy']:.4f}, "
              f"f1_macro={df.iloc[best_idx]['f1_macro']:.4f}")

        return optimal_threshold, df

    def evaluate_model(
        self,
        model: nn.Module,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """Evaluate model performance."""

        model.eval()
        X_test_t = torch.FloatTensor(X_test).to(DEVICE)

        with torch.no_grad():
            y_prob = model(X_test_t).cpu().numpy().flatten()

        y_pred = (y_prob >= threshold).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
        }

        print("\n" + "="*50)
        print(f"Test Set Evaluation (threshold={threshold:.3f})")
        print("="*50)
        print(f"  Accuracy:      {metrics['accuracy']:.1%}")
        print(f"  F1 Score:      {metrics['f1_score']:.4f}")
        print(f"  F1 Macro:      {metrics['f1_macro']:.4f}")
        print(f"  Balanced Acc:  {metrics['balanced_accuracy']:.4f}")
        print(f"  AUC-ROC:       {metrics['roc_auc']:.4f}")

        return metrics

    def save_model(
        self,
        model: nn.Module,
        scaler: StandardScaler,
        filename: str,
        threshold: float,
        params: Dict,
        history: Dict = None
    ):
        """Save model with all metadata."""
        filepath = self.models_path / filename

        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_size': model.input_size,
                'hidden_sizes': model.hidden_sizes,
                'dropout_rate': model.dropout_rate,
                'use_batch_norm': model.use_batch_norm,
                'activation': model.activation_name,
                'leaky_slope': model.leaky_slope,
                'use_residual': model.use_residual
            },
            'scaler': scaler,
            'optimal_threshold': threshold,
            'best_params': params,
            'training_history': history,
            'model_version': 'v2_improved',
            'timestamp': datetime.now().isoformat()
        }

        torch.save(save_dict, filepath)
        print(f"\n✓ Model saved to: {filepath}")

    def run_full_pipeline(
        self,
        tune: bool = True,
        n_trials: int = 50,
        include_odds: bool = False
    ) -> Dict:
        """Run complete training pipeline."""

        print("\n" + "="*70)
        print("    MLP Training Pipeline")
        print("="*70)

        # Load data
        X, y = self.load_data(include_odds=include_odds)
        X_np = X.values
        y_np = y.values

        # Split: 60% train, 20% val, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )

        print(f"\nData splits:")
        print(f"  Train: {len(X_train):,} ({len(X_train)/len(X_np)*100:.0f}%)")
        print(f"  Val:   {len(X_val):,} ({len(X_val)/len(X_np)*100:.0f}%)")
        print(f"  Test:  {len(X_test):,} ({len(X_test)/len(X_np)*100:.0f}%)")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Tune or use defaults
        if tune and OPTUNA_AVAILABLE:
            best_params = self.tune_hyperparameters(
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                n_trials=n_trials
            )
        else:
            best_params = self.get_default_params()
            print(f"\nUsing default parameters: {best_params}")

        # Train final model
        print("\n" + "="*60)
        print("Training Final Model")
        print("="*60)

        model, history = self.train_model(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            hidden_sizes=best_params['hidden_sizes'],
            dropout_rate=best_params['dropout_rate'],
            activation=best_params['activation'],
            leaky_slope=best_params.get('leaky_slope', 0.01),
            learning_rate=best_params['learning_rate'],
            weight_decay=best_params.get('weight_decay', 0.01),
            batch_size=best_params['batch_size'],
            epochs=300,
            patience=40,
            verbose=True
        )

        # Find optimal threshold
        threshold, _ = self.find_optimal_threshold(
            model, X_val_scaled, y_val, metric='f1_macro'
        )

        # Evaluate on test set
        metrics = self.evaluate_model(model, X_test_scaled, y_test, threshold)

        # Save model
        suffix = "_with_odds" if include_odds else ""
        self.save_model(
            model, scaler, f"mlp_model{suffix}.pt",
            threshold, best_params, history
        )

        self.model = model
        self.scaler = scaler
        self.optimal_threshold = threshold

        return {
            'metrics': metrics,
            'threshold': threshold,
            'params': best_params,
            'history': history,
            'dead_neurons': history.get('final_dead_neurons', {})
        }


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Trainer alias
ImprovedMLPTrainer = MLPTrainer


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train MLP")
    parser.add_argument("--quick", action="store_true",
                        help="Skip Optuna tuning")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials")
    parser.add_argument("--with-odds", action="store_true",
                        help="Include odds features")

    args = parser.parse_args()

    trainer = MLPTrainer()

    # Train without odds
    print("\n" + "🥊"*35)
    print("         Training Model WITHOUT Odds")
    print("🥊"*35)

    results = trainer.run_full_pipeline(
        tune=not args.quick,
        n_trials=args.trials,
        include_odds=False
    )

    # Train with odds
    if args.with_odds:
        print("\n" + "🥊"*35)
        print("         Training Model WITH Odds")
        print("🥊"*35)

        results_odds = trainer.run_full_pipeline(
            tune=not args.quick,
            n_trials=args.trials,
            include_odds=True
        )

    print("\n" + "="*70)
    print("✓ Training Complete!")
    print("="*70)

    return trainer, results


if __name__ == "__main__":
    trainer, results = main()
