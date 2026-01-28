"""
FightPredict - XGBoost Model Training (Full Tuning Edition)
============================================================
Trains XGBoost classifier for UFC fight outcome prediction.

Features:
- Stratified train/test split
- Class imbalance handling
- Optuna Bayesian hyperparameter optimization
- Threshold tuning for class imbalance
- Threshold stability check across CV folds
- Multiple evaluation metrics
- SHAP explainability
- Model saving

Usage:
    python src/train_xgboost.py

Output:
    - models/xgboost_model.pkl
    - models/xgboost_model_with_odds.pkl
    - models/training_results.json
    - models/shap_analysis.json
    - models/threshold_tuning*.png
    - models/threshold_stability*.png
    - models/optuna_optimization*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib
from datetime import datetime

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    precision_score, recall_score, balanced_accuracy_score
)
import xgboost as xgb

# Optuna for hyperparameter tuning
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not installed. Run: pip install optuna")

# Explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Run: pip install shap")

# Import our enhanced SHAP explainer
try:
    from shap_explainer import SHAPExplainer
    SHAP_EXPLAINER_AVAILABLE = True
except ImportError:
    SHAP_EXPLAINER_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')


class UFCXGBoostTrainer:
    """Trains and evaluates XGBoost models for UFC fight prediction"""

    def __init__(self, data_path: str = "data/processed", models_path: str = "models"):
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.model_with_odds = None
        self.shap_explainer = None
        self.shap_explainer_odds = None
        self.feature_names = None
        self.feature_names_odds = None
        self.optimal_threshold = 0.5
        self.optimal_threshold_odds = 0.5
        self.best_params = None
        self.best_params_odds = None
        self.results = {}

    def load_data(self, include_odds: bool = False) -> tuple:
        """Load model-ready data"""
        if include_odds:
            filepath = self.data_path / "features_model_ready_with_odds.csv"
        else:
            filepath = self.data_path / "features_model_ready.csv"

        df = pd.read_csv(filepath)

        X = df.drop('target', axis=1)
        y = df['target']

        print(f"Loaded: {len(X):,} samples, {len(X.columns)} features")
        print(f"Class distribution: {y.value_counts().to_dict()}")

        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple:
        """Stratified train/test split"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        print(f"\nTrain set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        print(
            f"Train class ratio: {y_train.mean():.1%} / {1-y_train.mean():.1%}")
        print(f"Test class ratio: {y_test.mean():.1%} / {1-y_test.mean():.1%}")

        return X_train, X_test, y_train, y_test

    def calculate_class_weight(self, y: pd.Series) -> float:
        """Calculate scale_pos_weight for imbalanced classes"""
        n_positive = (y == 1).sum()
        n_negative = (y == 0).sum()
        scale = n_negative / n_positive
        print(f"\nClass weight (scale_pos_weight): {scale:.3f}")
        return scale

    def get_base_model(self, scale_pos_weight: float = 1.0) -> xgb.XGBClassifier:
        """Get base XGBoost model with good defaults"""
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    def train_basic(self, X_train, y_train, X_test, y_test,
                    use_class_weight: bool = True) -> xgb.XGBClassifier:
        """Train basic XGBoost model"""
        print("\n" + "=" * 60)
        print("Training Basic XGBoost Model")
        print("=" * 60)

        scale = self.calculate_class_weight(
            y_train) if use_class_weight else 1.0
        model = self.get_base_model(scale_pos_weight=scale)

        print("\nTraining...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        print("✓ Training complete")
        return model

    def train_with_tuning(self, X_train, y_train, X_test, y_test,
                          use_class_weight: bool = True) -> xgb.XGBClassifier:
        """Train XGBoost with GridSearchCV hyperparameter tuning"""
        print("\n" + "=" * 60)
        print("Training XGBoost with Hyperparameter Tuning (GridSearchCV)")
        print("=" * 60)

        scale = self.calculate_class_weight(
            y_train) if use_class_weight else 1.0

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
        }

        base_model = xgb.XGBClassifier(
            scale_pos_weight=scale,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        print("\nRunning GridSearchCV (this may take a few minutes)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"\n✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV F1 score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    # =========================================================================
    # OPTUNA HYPERPARAMETER TUNING
    # =========================================================================

    def tune_hyperparameters_optuna(self, X_train, y_train, X_val, y_val,
                                    n_trials: int = 100,
                                    optimize_metric: str = 'roc_auc') -> dict:
        """
        Advanced hyperparameter tuning using Optuna with Bayesian optimization

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data for evaluation
            n_trials: Number of optimization trials
            optimize_metric: 'roc_auc', 'f1', or 'f1_macro'

        Returns:
            Best hyperparameters dict
        """
        if not OPTUNA_AVAILABLE:
            print("⚠️ Optuna not installed. Using default parameters.")
            return self.get_base_model().get_params()

        print("\n" + "=" * 60)
        print("Hyperparameter Tuning with Optuna (Bayesian Optimization)")
        print("=" * 60)
        print(f"Optimization metric: {optimize_metric}")
        print(f"Running {n_trials} trials...")

        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                # Core parameters
                'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),

                # Sampling parameters
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),

                # Regularization
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),

                # Class imbalance
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.3, 1.0),
            }

            model = xgb.XGBClassifier(
                **params,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                early_stopping_rounds=30,
                n_jobs=-1
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)

            if optimize_metric == 'roc_auc':
                score = roc_auc_score(y_val, y_prob)
            elif optimize_metric == 'f1':
                score = f1_score(y_val, y_pred)
            elif optimize_metric == 'f1_macro':
                score = f1_score(y_val, y_pred, average='macro')
            else:
                score = roc_auc_score(y_val, y_prob)

            return score

        # Create study with TPE sampler (Bayesian)
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)

        # Progress callback
        def print_progress(study, trial):
            if (trial.number + 1) % 20 == 0:
                print(f"   Trial {trial.number + 1}/{n_trials}: "
                      f"Best {optimize_metric} = {study.best_value:.4f}")

        study.optimize(objective, n_trials=n_trials, callbacks=[print_progress],
                       show_progress_bar=True)

        print(f"\n✓ Optimization complete!")
        print(f"✓ Best {optimize_metric}: {study.best_value:.4f}")
        print(f"✓ Best parameters:")
        for k, v in study.best_params.items():
            if isinstance(v, float):
                print(f"   {k}: {v:.6f}")
            else:
                print(f"   {k}: {v}")

        # Plot optimization history
        self._plot_optuna_history(study)

        return study.best_params

    def _plot_optuna_history(self, study, suffix: str = ""):
        """Plot Optuna optimization history"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Optimization history
        trials = [t.number for t in study.trials]
        values = [t.value for t in study.trials]
        best_values = [max(values[:i+1]) for i in range(len(values))]

        axes[0].scatter(trials, values, alpha=0.5, label='Trial score')
        axes[0].plot(trials, best_values, color='red',
                     linewidth=2, label='Best so far')
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Optuna Optimization History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Parameter importances (top 10)
        try:
            importances = optuna.importance.get_param_importances(study)
            params = list(importances.keys())[:10]
            imp_values = [importances[p] for p in params]

            axes[1].barh(range(len(params)), imp_values, color='steelblue')
            axes[1].set_yticks(range(len(params)))
            axes[1].set_yticklabels(params)
            axes[1].invert_yaxis()
            axes[1].set_xlabel('Importance')
            axes[1].set_title('Hyperparameter Importance')
            axes[1].grid(True, alpha=0.3, axis='x')
        except Exception:
            axes[1].text(0.5, 0.5, 'Not enough trials\nfor importance analysis',
                         ha='center', va='center', fontsize=12)
            axes[1].set_title('Hyperparameter Importance')

        plt.tight_layout()
        plt.savefig(self.models_path /
                    f"optuna_optimization{suffix}.png", dpi=150)
        plt.show()

    # =========================================================================
    # THRESHOLD TUNING
    # =========================================================================

    def find_optimal_threshold(self, model, X_val, y_val,
                               metric: str = 'f1_macro',
                               model_suffix: str = "") -> tuple:
        """
        Find optimal classification threshold for the given metric

        Args:
            model: Trained model
            X_val, y_val: Validation data
            metric: 'f1_macro', 'f1', 'accuracy', 'balanced_accuracy', or 'youden'
            model_suffix: For plot filenames

        Returns:
            (optimal_threshold, results_dataframe)
        """
        print("\n" + "-" * 50)
        print(f"Finding Optimal Threshold (optimizing: {metric})")
        print("-" * 50)

        y_prob = model.predict_proba(X_val)[:, 1]

        # Test thresholds from 0.25 to 0.75
        thresholds = np.arange(0.25, 0.76, 0.01)
        results = []

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)

            # Skip if all predictions are same class
            if len(np.unique(y_pred)) < 2:
                continue

            metrics_dict = {
                'threshold': thresh,
                'accuracy': accuracy_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred),
                'f1_macro': f1_score(y_val, y_pred, average='macro'),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_val, y_pred),
            }

            # Youden's J statistic = Sensitivity + Specificity - 1
            cm = confusion_matrix(y_val, y_pred)
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics_dict['youden'] = sensitivity + specificity - 1
            metrics_dict['sensitivity'] = sensitivity
            metrics_dict['specificity'] = specificity

            results.append(metrics_dict)

        df = pd.DataFrame(results)

        # Find best threshold for target metric
        best_idx = df[metric].idxmax()
        best_row = df.iloc[best_idx]
        optimal_threshold = best_row['threshold']

        # Get default threshold row
        default_mask = df['threshold'].round(2) == 0.50
        if default_mask.any():
            default_row = df[default_mask].iloc[0]
        else:
            default_row = df.iloc[len(df)//2]

        print(
            f"\n   {'Metric':<20} {'Default (0.50)':<15} {'Optimal ('+f'{optimal_threshold:.2f})':>15}")
        print(f"   {'-'*50}")
        print(
            f"   {'Accuracy':<20} {default_row['accuracy']:<15.4f} {best_row['accuracy']:<15.4f}")
        print(
            f"   {'F1 Score':<20} {default_row['f1']:<15.4f} {best_row['f1']:<15.4f}")
        print(
            f"   {'F1 Macro':<20} {default_row['f1_macro']:<15.4f} {best_row['f1_macro']:<15.4f}")
        print(
            f"   {'Balanced Acc':<20} {default_row['balanced_accuracy']:<15.4f} {best_row['balanced_accuracy']:<15.4f}")
        print(
            f"   {'Precision':<20} {default_row['precision']:<15.4f} {best_row['precision']:<15.4f}")
        print(
            f"   {'Recall':<20} {default_row['recall']:<15.4f} {best_row['recall']:<15.4f}")

        # Calculate improvements
        acc_improvement = (best_row['accuracy'] -
                           default_row['accuracy']) * 100
        f1m_improvement = (best_row['f1_macro'] -
                           default_row['f1_macro']) * 100

        print(f"\n   📈 Threshold tuning improvement:")
        print(f"      Accuracy:  {acc_improvement:+.2f}%")
        print(f"      F1 Macro:  {f1m_improvement:+.2f}%")

        # Plot threshold curves
        self._plot_threshold_curves(
            df, optimal_threshold, metric, model_suffix)

        return optimal_threshold, df

    def _plot_threshold_curves(self, df, optimal_threshold, metric, model_suffix=""):
        """Plot threshold tuning curves"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Main metrics vs Threshold
        ax1 = axes[0]
        ax1.plot(df['threshold'], df['accuracy'],
                 label='Accuracy', linewidth=2)
        ax1.plot(df['threshold'], df['f1'], label='F1 (class 1)', linewidth=2)
        ax1.plot(df['threshold'], df['f1_macro'],
                 label='F1 Macro', linewidth=2, linestyle='--')
        ax1.plot(df['threshold'], df['balanced_accuracy'],
                 label='Balanced Acc', linewidth=2)
        ax1.axvline(x=0.5, color='gray', linestyle='--',
                    alpha=0.5, label='Default (0.5)')
        ax1.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2,
                    label=f'Optimal ({optimal_threshold:.2f})')
        ax1.set_xlabel('Threshold', fontsize=11)
        ax1.set_ylabel('Score', fontsize=11)
        ax1.set_title('Metrics vs Classification Threshold',
                      fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0.25, 0.75)

        # Plot 2: Precision-Recall Tradeoff
        ax2 = axes[1]
        ax2.plot(df['threshold'], df['precision'],
                 label='Precision', linewidth=2, color='blue')
        ax2.plot(df['threshold'], df['recall'],
                 label='Recall', linewidth=2, color='orange')
        ax2.plot(df['threshold'], df['sensitivity'], label='Sensitivity', linewidth=2,
                 linestyle=':', color='green')
        ax2.plot(df['threshold'], df['specificity'], label='Specificity', linewidth=2,
                 linestyle=':', color='red')
        ax2.axvline(x=optimal_threshold, color='red',
                    linestyle='--', linewidth=2)
        ax2.set_xlabel('Threshold', fontsize=11)
        ax2.set_ylabel('Score', fontsize=11)
        ax2.set_title('Precision/Recall & Sensitivity/Specificity',
                      fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0.25, 0.75)

        # Plot 3: Target metric zoomed
        ax3 = axes[2]
        ax3.plot(df['threshold'], df[metric], linewidth=2, color='purple')
        ax3.axvline(x=optimal_threshold, color='red',
                    linestyle='--', linewidth=2)
        ax3.scatter([optimal_threshold], [df.loc[df[metric].idxmax(), metric]],
                    color='red', s=100, zorder=5, label=f'Best: {df[metric].max():.4f}')
        ax3.set_xlabel('Threshold', fontsize=11)
        ax3.set_ylabel(metric, fontsize=11)
        ax3.set_title(
            f'Optimized Metric: {metric}', fontsize=12, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0.25, 0.75)

        plt.tight_layout()
        filename = f"threshold_tuning{model_suffix}.png"
        plt.savefig(self.models_path / filename, dpi=150)
        plt.show()
        print(f"✓ Saved: {self.models_path / filename}")

    # =========================================================================
    # THRESHOLD STABILITY CHECK
    # =========================================================================

    def check_threshold_stability(self, X, y, best_params: dict,
                                  metric: str = 'f1_macro',
                                  n_splits: int = 5,
                                  model_suffix: str = "") -> dict:
        """
        Check if optimal threshold is stable across CV folds

        Args:
            X, y: Full dataset
            best_params: Tuned hyperparameters
            metric: Metric to optimize threshold for
            n_splits: Number of CV folds
            model_suffix: For plot filenames

        Returns:
            Dict with threshold statistics
        """
        print("\n" + "=" * 60)
        print(f"Threshold Stability Check ({n_splits}-Fold CV)")
        print("=" * 60)

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_thresholds = []
        fold_metrics_default = []
        fold_metrics_tuned = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            # Train model with best params
            model = xgb.XGBClassifier(
                **best_params,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            model.fit(X_train_fold, y_train_fold, verbose=False)

            # Find optimal threshold for this fold
            y_prob = model.predict_proba(X_val_fold)[:, 1]

            best_thresh = 0.5
            best_score = 0

            for thresh in np.arange(0.25, 0.76, 0.01):
                y_pred = (y_prob >= thresh).astype(int)
                if len(np.unique(y_pred)) < 2:
                    continue

                if metric == 'f1_macro':
                    score = f1_score(y_val_fold, y_pred, average='macro')
                elif metric == 'f1':
                    score = f1_score(y_val_fold, y_pred)
                else:
                    score = accuracy_score(y_val_fold, y_pred)

                if score > best_score:
                    best_score = score
                    best_thresh = thresh

            # Get default threshold metrics
            y_pred_default = (y_prob >= 0.5).astype(int)
            default_score = f1_score(
                y_val_fold, y_pred_default, average='macro')

            fold_thresholds.append(best_thresh)
            fold_metrics_default.append(default_score)
            fold_metrics_tuned.append(best_score)

            print(f"   Fold {fold}: optimal_threshold={best_thresh:.2f}, "
                  f"F1_macro: {default_score:.4f} → {best_score:.4f}")

        # Calculate statistics
        mean_thresh = np.mean(fold_thresholds)
        std_thresh = np.std(fold_thresholds)
        min_thresh = np.min(fold_thresholds)
        max_thresh = np.max(fold_thresholds)

        print(f"\n   {'─'*50}")
        print(f"   Threshold Statistics:")
        print(f"      Mean:  {mean_thresh:.3f}")
        print(f"      Std:   {std_thresh:.3f}")
        print(f"      Range: [{min_thresh:.2f}, {max_thresh:.2f}]")

        # Stability assessment
        if std_thresh < 0.05:
            stability = "✅ STABLE (std < 0.05)"
        elif std_thresh < 0.10:
            stability = "⚠️ MODERATE (std < 0.10)"
        else:
            stability = "❌ UNSTABLE (std >= 0.10)"

        print(f"      Assessment: {stability}")

        # Recommendation
        recommended = mean_thresh
        conservative = min(mean_thresh + std_thresh, 0.5)

        print(f"\n   Recommendation:")
        print(f"      Use threshold: {recommended:.3f} (mean across folds)")
        print(
            f"      Conservative:  {conservative:.3f} (mean + 1 std, capped at 0.5)")

        # Average improvement
        avg_improvement = np.mean(fold_metrics_tuned) - \
            np.mean(fold_metrics_default)
        print(
            f"\n   Avg F1_macro improvement from threshold tuning: {avg_improvement:+.4f}")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot 1: Thresholds per fold
        ax1 = axes[0]
        folds = range(1, n_splits + 1)
        ax1.bar(folds, fold_thresholds, color='steelblue', alpha=0.7)
        ax1.axhline(y=mean_thresh, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_thresh:.3f}')
        ax1.axhline(y=0.5, color='gray', linestyle=':', linewidth=2,
                    label='Default: 0.50')
        ax1.fill_between([0.5, n_splits + 0.5],
                         mean_thresh - std_thresh, mean_thresh + std_thresh,
                         color='red', alpha=0.2, label=f'±1 std: {std_thresh:.3f}')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Optimal Threshold')
        ax1.set_title('Optimal Threshold per CV Fold')
        ax1.set_xticks(folds)
        ax1.legend(loc='best')
        ax1.set_ylim(0.2, 0.7)
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: F1 Macro improvement
        ax2 = axes[1]
        x = np.arange(n_splits)
        width = 0.35
        ax2.bar(x - width/2, fold_metrics_default, width,
                label='Default (0.50)', color='gray', alpha=0.7)
        ax2.bar(x + width/2, fold_metrics_tuned, width,
                label='Tuned', color='green', alpha=0.7)
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('F1 Macro')
        ax2.set_title('F1 Macro: Default vs Tuned Threshold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Fold {i+1}' for i in x])
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = f"threshold_stability{model_suffix}.png"
        plt.savefig(self.models_path / filename, dpi=150)
        plt.show()
        print(f"✓ Saved: {self.models_path / filename}")

        return {
            'fold_thresholds': fold_thresholds,
            'mean_threshold': float(mean_thresh),
            'std_threshold': float(std_thresh),
            'range': (float(min_thresh), float(max_thresh)),
            'stability': stability,
            'recommended_threshold': float(recommended),
            'conservative_threshold': float(conservative),
            'avg_improvement': float(avg_improvement),
        }

    # =========================================================================
    # FULL TUNING PIPELINE
    # =========================================================================

    def train_with_full_tuning(self, X, y, n_trials: int = 100,
                               threshold_metric: str = 'f1_macro',
                               optimize_metric: str = 'roc_auc',
                               model_suffix: str = "") -> tuple:
        """
        Complete training pipeline with:
        1. Train/val/test split (60/20/20)
        2. Optuna hyperparameter tuning
        3. Threshold optimization
        4. Final evaluation

        Args:
            X, y: Full dataset
            n_trials: Number of Optuna trials
            threshold_metric: Metric to optimize threshold for
            optimize_metric: Metric for Optuna optimization
            model_suffix: For file naming

        Returns:
            (model, best_params, optimal_threshold, X_test, y_test, results_comparison)
        """
        print("\n" + "=" * 70)
        print("    Full Tuning Pipeline: Optuna + Threshold Optimization")
        print("=" * 70)

        # Split: 60% train, 20% val (for tuning), 20% test (final eval)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )

        print(f"\nData splits:")
        print(
            f"   Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.0f}%)")
        print(
            f"   Val:   {len(X_val):,} samples ({len(X_val)/len(X)*100:.0f}%) - for tuning")
        print(
            f"   Test:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.0f}%) - final eval")

        # Step 1: Hyperparameter tuning with Optuna
        best_params = self.tune_hyperparameters_optuna(
            X_train, y_train, X_val, y_val,
            n_trials=n_trials,
            optimize_metric=optimize_metric
        )

        # Step 2: Train model with best params for threshold tuning
        print("\n" + "=" * 60)
        print("Training Model for Threshold Tuning")
        print("=" * 60)

        temp_model = xgb.XGBClassifier(
            **best_params,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        temp_model.fit(X_train, y_train, verbose=False)

        # Step 3: Threshold tuning on validation set
        optimal_threshold, threshold_df = self.find_optimal_threshold(
            temp_model, X_val, y_val,
            metric=threshold_metric,
            model_suffix=model_suffix
        )

        # Step 4: Train final model on train+val
        print("\n" + "=" * 60)
        print("Training Final Model on Train+Val Data")
        print("=" * 60)

        X_train_full = pd.concat([X_train, X_val])
        y_train_full = pd.concat([y_train, y_val])

        final_model = xgb.XGBClassifier(
            **best_params,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        final_model.fit(X_train_full, y_train_full, verbose=False)
        print(f"✓ Final model trained on {len(X_train_full):,} samples")

        # Step 5: Final evaluation on test set
        print("\n" + "=" * 60)
        print("Final Evaluation on Held-Out Test Set")
        print("=" * 60)

        y_prob = final_model.predict_proba(X_test)[:, 1]

        print(f"\n{'─'*60}")
        print(
            f"{'Metric':<20} {'Default (0.50)':<18} {'Tuned ('+f'{optimal_threshold:.2f})':>18}")
        print(f"{'─'*60}")

        results_comparison = {}
        for thresh_name, thresh in [('default', 0.5), ('tuned', optimal_threshold)]:
            y_pred = (y_prob >= thresh).astype(int)

            results_comparison[thresh_name] = {
                'threshold': thresh,
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_prob),
            }

        for metric_name in ['accuracy', 'f1', 'f1_macro', 'balanced_accuracy', 'precision', 'recall', 'roc_auc']:
            default_val = results_comparison['default'][metric_name]
            tuned_val = results_comparison['tuned'][metric_name]
            diff = tuned_val - default_val
            arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
            print(
                f"{metric_name:<20} {default_val:<18.4f} {tuned_val:.4f} {arrow} ({diff:+.4f})")

        print(f"{'─'*60}")

        # Confusion matrices comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for idx, (thresh_name, thresh) in enumerate([('Default (0.50)', 0.5),
                                                     (f'Tuned ({optimal_threshold:.2f})', optimal_threshold)]):
            y_pred = (y_prob >= thresh).astype(int)
            cm = confusion_matrix(y_test, y_pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                        xticklabels=['F2 Wins', 'F1 Wins'],
                        yticklabels=['F2 Wins', 'F1 Wins'])
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average='macro')
            axes[idx].set_title(f'{thresh_name}\nAcc: {acc:.3f}, F1-Macro: {f1m:.3f}',
                                fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.models_path /
                    f"confusion_matrix_comparison{model_suffix}.png", dpi=150)
        plt.show()

        return final_model, best_params, optimal_threshold, X_test, y_test, results_comparison

    # =========================================================================
    # EVALUATION & PLOTTING
    # =========================================================================

    def evaluate_model(self, model, X_test, y_test, model_name: str = "Model",
                       threshold: float = 0.5) -> dict:
        """Comprehensive model evaluation with custom threshold"""
        print("\n" + "=" * 60)
        print(f"Evaluating: {model_name} (threshold={threshold:.2f})")
        print("=" * 60)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'threshold': threshold,
        }

        print(f"\n📊 Results:")
        print(f"   Accuracy:      {metrics['accuracy']:.1%}")
        print(f"   F1 Score:      {metrics['f1_score']:.4f}")
        print(f"   F1 Macro:      {metrics['f1_macro']:.4f}")
        print(f"   Balanced Acc:  {metrics['balanced_accuracy']:.4f}")
        print(f"   Precision:     {metrics['precision']:.4f}")
        print(f"   Recall:        {metrics['recall']:.4f}")
        print(f"   AUC-ROC:       {metrics['roc_auc']:.4f}")

        baseline_acc = y_test.mean()
        print(f"\n📈 Baseline (always predict F1): {baseline_acc:.1%}")
        print(
            f"   Improvement over baseline: +{(metrics['accuracy'] - baseline_acc)*100:.1f}%")

        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred,
              target_names=['F2 Wins', 'F1 Wins']))

        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        return metrics

    def plot_confusion_matrix(self, model, X_test, y_test, title: str = "Confusion Matrix",
                              threshold: float = 0.5):
        """Plot confusion matrix with custom threshold"""
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['F2 Wins', 'F1 Wins'],
                    yticklabels=['F2 Wins', 'F1 Wins'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f"{title}\n(threshold={threshold:.2f})",
                     fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.models_path /
                    f"{title.lower().replace(' ', '_')}.png", dpi=150)
        plt.close()

        return cm

    def plot_feature_importance(self, model, feature_names, top_n: int = 20,
                                title: str = "Feature Importance"):
        """Plot XGBoost feature importance"""
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = importance.head(top_n)

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
        ax.barh(range(len(top_features)),
                top_features['importance'], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance (Gain)')
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.models_path /
                    f"{title.lower().replace(' ', '_')}.png", dpi=150)
        plt.close()

        return importance

    def cross_validate(self, model, X, y, cv: int = 5) -> dict:
        """Perform cross-validation"""
        print(f"\n🔄 Running {cv}-fold Cross-Validation...")

        cv_strat = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        scores = {
            'accuracy': cross_val_score(model, X, y, cv=cv_strat, scoring='accuracy'),
            'f1': cross_val_score(model, X, y, cv=cv_strat, scoring='f1'),
            'f1_macro': cross_val_score(model, X, y, cv=cv_strat, scoring='f1_macro'),
            'roc_auc': cross_val_score(model, X, y, cv=cv_strat, scoring='roc_auc'),
        }

        print(f"\nCross-Validation Results ({cv} folds):")
        for metric, values in scores.items():
            print(f"   {metric}: {values.mean():.4f} (+/- {values.std()*2:.4f})")

        return {k: {'mean': v.mean(), 'std': v.std()} for k, v in scores.items()}

    # =========================================================================
    # SHAP ANALYSIS
    # =========================================================================

    def run_shap_deep_dive(self, model, X_test, feature_names, model_suffix: str = ""):
        """Run enhanced SHAP analysis"""
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP not available. Skipping deep dive.")
            return None, None

        print("\n" + "=" * 70)
        print(
            f"        SHAP Deep Dive Analysis{model_suffix.replace('_', ' ').title()}")
        print("=" * 70)

        if SHAP_EXPLAINER_AVAILABLE:
            explainer = SHAPExplainer(model, output_path=str(self.models_path))
            results = explainer.analyze(
                X_test, feature_names=feature_names, max_samples=1000)

            # Rename output files with suffix (Windows-safe)
            if model_suffix:
                for old_name in ['shap_summary_detailed.png', 'shap_bar_importance.png',
                                 'shap_dependence_plots.png', 'shap_analysis.json']:
                    old_path = self.models_path / old_name
                    if old_path.exists():
                        new_name = old_name.replace('.', f'{model_suffix}.')
                        new_path = self.models_path / new_name
                        if new_path.exists():
                            new_path.unlink()
                        old_path.rename(new_path)

            # Demo single prediction
            print("\n" + "-" * 50)
            print("Example: Single Prediction Explanation")
            print("-" * 50)

            sample = X_test.iloc[0]
            explanation = explainer.explain_single_prediction(
                sample, fighter_1_name="Fighter 1", fighter_2_name="Fighter 2"
            )
            print(explanation['summary'])

            return explainer, results
        else:
            # Fallback to basic SHAP
            return self.explain_with_shap(model, X_test, feature_names)

    def explain_with_shap(self, model, X_test, feature_names, max_samples: int = 1000):
        """Basic SHAP explanations (fallback)"""
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP not available.")
            return None, None

        print("\nCalculating SHAP values...")

        if len(X_test) > max_samples:
            X_sample = X_test.sample(max_samples, random_state=42)
        else:
            X_sample = X_test

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample,
                          feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(self.models_path / "shap_summary.png",
                    dpi=150, bbox_inches='tight')
        plt.close()

        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('shap_importance', ascending=False)

        print("\nTop 15 Features by SHAP:")
        print(shap_importance.head(15).to_string(index=False))

        return shap_values, shap_importance

    # =========================================================================
    # SAVE/LOAD
    # =========================================================================

    def save_model(self, model, filename: str, threshold: float = 0.5, params: dict = None):
        """Save trained model with metadata"""
        filepath = self.models_path / filename

        save_data = {
            'model': model,
            'optimal_threshold': threshold,
            'best_params': params,
        }

        joblib.dump(save_data, filepath)
        print(f"\n💾 Model saved: {filepath}")
        print(
            f"   Includes: model, optimal_threshold={threshold:.3f}, best_params")

    def save_results(self, results: dict, filename: str = "training_results.json"):
        """Save training results to JSON"""
        filepath = self.models_path / filename

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj

        results_clean = json.loads(json.dumps(results, default=convert))

        with open(filepath, 'w') as f:
            json.dump(results_clean, f, indent=2)

        print(f"📄 Results saved: {filepath}")

    # =========================================================================
    # PREDICTION
    # =========================================================================

    def predict_fight(self, features: pd.DataFrame, include_odds: bool = False,
                      fighter_1_name: str = "Fighter 1",
                      fighter_2_name: str = "Fighter 2") -> dict:
        """
        Predict outcome for a new fight with explanation
        """
        model = self.model_with_odds if include_odds else self.model
        threshold = self.optimal_threshold_odds if include_odds else self.optimal_threshold
        explainer = self.shap_explainer_odds if include_odds else self.shap_explainer

        if model is None:
            raise ValueError("Model not trained. Run training first.")

        pred_proba = model.predict_proba(features)[0]
        pred_class = int(pred_proba[1] >= threshold)

        result = {
            'prediction': {
                'winner': fighter_1_name if pred_class == 1 else fighter_2_name,
                'loser': fighter_2_name if pred_class == 1 else fighter_1_name,
                'f1_win_probability': float(pred_proba[1]),
                'f2_win_probability': float(pred_proba[0]),
                'confidence': float(max(pred_proba)),
                'threshold_used': threshold,
            }
        }

        if explainer is not None:
            explanation = explainer.explain_single_prediction(
                features.iloc[0],
                fighter_1_name=fighter_1_name,
                fighter_2_name=fighter_2_name
            )
            result['explanation'] = explanation

        return result

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================

    def run_full_pipeline(self, tune_hyperparameters: bool = False,
                          full_tuning: bool = False,
                          n_trials: int = 100,
                          threshold_metric: str = 'f1_macro',
                          run_shap_deep_dive: bool = True,
                          check_threshold_stability: bool = True):
        """
        Run complete training pipeline

        Args:
            tune_hyperparameters: Use GridSearchCV (legacy)
            full_tuning: Use Optuna + threshold tuning (recommended)
            n_trials: Number of Optuna trials
            threshold_metric: Metric to optimize threshold for
            run_shap_deep_dive: Run enhanced SHAP analysis
            check_threshold_stability: Run CV-based threshold stability check
        """
        print("=" * 70)
        print("          FightPredict - XGBoost Training Pipeline")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(
            f"Mode: {'Full Tuning (Optuna + Threshold)' if full_tuning else 'Basic Training'}")

        all_results = {}

        # =============================================
        # MODEL 1: Without Betting Odds
        # =============================================
        print("\n\n" + "🥊" * 35)
        print("MODEL 1: Without Betting Odds")
        print("🥊" * 35)

        X, y = self.load_data(include_odds=False)
        self.feature_names = list(X.columns)

        if full_tuning:
            model, best_params, optimal_threshold, X_test, y_test, eval_results = \
                self.train_with_full_tuning(
                    X, y,
                    n_trials=n_trials,
                    threshold_metric=threshold_metric,
                    model_suffix=""
                )
            self.model = model
            self.best_params = best_params
            self.optimal_threshold = optimal_threshold
        elif tune_hyperparameters:
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            model = self.train_with_tuning(X_train, y_train, X_test, y_test)
            self.model = model
            self.optimal_threshold = 0.5
            best_params = None
        else:
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            model = self.train_basic(X_train, y_train, X_test, y_test)
            self.model = model
            self.optimal_threshold = 0.5
            best_params = None

        # Evaluate with optimal threshold
        metrics = self.evaluate_model(model, X_test, y_test, "XGBoost (No Odds)",
                                      threshold=self.optimal_threshold)

        # Cross-validation
        cv_scores = self.cross_validate(model, X, y)

        # Feature importance
        importance = self.plot_feature_importance(model, self.feature_names,
                                                  title="Feature Importance - No Odds")

        # Threshold stability check
        stability_results = None
        if full_tuning and check_threshold_stability and best_params is not None:
            stability_results = self.check_threshold_stability(
                X, y, best_params, metric=threshold_metric, model_suffix=""
            )

            # Update threshold to recommended if stable
            if stability_results['std_threshold'] < 0.10:
                print(
                    f"\n   Using recommended threshold: {stability_results['recommended_threshold']:.3f}")
                self.optimal_threshold = stability_results['recommended_threshold']

        # SHAP Analysis
        if run_shap_deep_dive and SHAP_AVAILABLE:
            self.shap_explainer, shap_results = self.run_shap_deep_dive(
                model, X_test, self.feature_names, model_suffix=""
            )

        # Save model with threshold
        self.save_model(model, "xgboost_model.pkl",
                        threshold=self.optimal_threshold,
                        params=best_params)

        all_results['model_no_odds'] = {
            'metrics': metrics,
            'cv_scores': cv_scores,
            'top_features': importance.head(20).to_dict('records'),
            'optimal_threshold': self.optimal_threshold,
            'best_params': best_params,
            'threshold_stability': stability_results,
            'n_features': len(X.columns),
            'n_samples': len(X),
        }

        # =============================================
        # MODEL 2: With Betting Odds
        # =============================================
        print("\n\n" + "💰" * 35)
        print("MODEL 2: With Betting Odds")
        print("💰" * 35)

        X_odds, y_odds = self.load_data(include_odds=True)
        self.feature_names_odds = list(X_odds.columns)

        if full_tuning:
            model_odds, best_params_odds, optimal_threshold_odds, X_test_o, y_test_o, eval_results_odds = \
                self.train_with_full_tuning(
                    X_odds, y_odds,
                    n_trials=n_trials,
                    threshold_metric=threshold_metric,
                    model_suffix="_with_odds"
                )
            self.model_with_odds = model_odds
            self.best_params_odds = best_params_odds
            self.optimal_threshold_odds = optimal_threshold_odds
        elif tune_hyperparameters:
            X_train_o, X_test_o, y_train_o, y_test_o = self.split_data(
                X_odds, y_odds)
            model_odds = self.train_with_tuning(
                X_train_o, y_train_o, X_test_o, y_test_o)
            self.model_with_odds = model_odds
            self.optimal_threshold_odds = 0.5
            best_params_odds = None
        else:
            X_train_o, X_test_o, y_train_o, y_test_o = self.split_data(
                X_odds, y_odds)
            model_odds = self.train_basic(
                X_train_o, y_train_o, X_test_o, y_test_o)
            self.model_with_odds = model_odds
            self.optimal_threshold_odds = 0.5
            best_params_odds = None

        # Evaluate
        metrics_odds = self.evaluate_model(model_odds, X_test_o, y_test_o,
                                           "XGBoost (With Odds)",
                                           threshold=self.optimal_threshold_odds)

        cv_scores_odds = self.cross_validate(model_odds, X_odds, y_odds)

        importance_odds = self.plot_feature_importance(model_odds, self.feature_names_odds,
                                                       title="Feature Importance - With Odds")

        # Threshold stability check for odds model
        stability_results_odds = None
        if full_tuning and check_threshold_stability and best_params_odds is not None:
            stability_results_odds = self.check_threshold_stability(
                X_odds, y_odds, best_params_odds, metric=threshold_metric, model_suffix="_with_odds"
            )

            # Update threshold to recommended if stable
            if stability_results_odds['std_threshold'] < 0.10:
                print(
                    f"\n   Using recommended threshold: {stability_results_odds['recommended_threshold']:.3f}")
                self.optimal_threshold_odds = stability_results_odds['recommended_threshold']

        # SHAP
        if run_shap_deep_dive and SHAP_AVAILABLE:
            self.shap_explainer_odds, shap_results_odds = self.run_shap_deep_dive(
                model_odds, X_test_o, self.feature_names_odds, model_suffix="_with_odds"
            )

        # Save
        self.save_model(model_odds, "xgboost_model_with_odds.pkl",
                        threshold=self.optimal_threshold_odds,
                        params=best_params_odds)

        all_results['model_with_odds'] = {
            'metrics': metrics_odds,
            'cv_scores': cv_scores_odds,
            'top_features': importance_odds.head(20).to_dict('records'),
            'optimal_threshold': self.optimal_threshold_odds,
            'best_params': best_params_odds,
            'threshold_stability': stability_results_odds,
            'n_features': len(X_odds.columns),
            'n_samples': len(X_odds),
        }

        # =============================================
        # FINAL COMPARISON
        # =============================================
        print("\n\n" + "=" * 70)
        print("                    FINAL COMPARISON")
        print("=" * 70)

        print("\n┌─────────────────────┬──────────────┬──────────────┐")
        print("│ Metric              │   No Odds    │  With Odds   │")
        print("├─────────────────────┼──────────────┼──────────────┤")
        print(
            f"│ Accuracy            │    {metrics['accuracy']:.1%}     │    {metrics_odds['accuracy']:.1%}     │")
        print(
            f"│ F1 Score            │    {metrics['f1_score']:.4f}     │    {metrics_odds['f1_score']:.4f}     │")
        print(
            f"│ F1 Macro            │    {metrics['f1_macro']:.4f}     │    {metrics_odds['f1_macro']:.4f}     │")
        print(
            f"│ Balanced Accuracy   │    {metrics['balanced_accuracy']:.4f}     │    {metrics_odds['balanced_accuracy']:.4f}     │")
        print(
            f"│ AUC-ROC             │    {metrics['roc_auc']:.4f}     │    {metrics_odds['roc_auc']:.4f}     │")
        print(
            f"│ Optimal Threshold   │    {self.optimal_threshold:.3f}      │    {self.optimal_threshold_odds:.3f}      │")
        print("└─────────────────────┴──────────────┴──────────────┘")

        # Threshold stability summary
        if stability_results and stability_results_odds:
            print("\n📊 Threshold Stability:")
            print(f"   No Odds:    {stability_results['stability']}")
            print(f"   With Odds:  {stability_results_odds['stability']}")

        improvement = metrics_odds['accuracy'] - metrics['accuracy']
        print(f"\n📈 Adding odds improved accuracy by: {improvement*100:+.1f}%")

        baseline = 0.646
        print(f"\n📊 Baseline (always predict F1): {baseline:.1%}")
        print(
            f"   Model (no odds) vs baseline: {(metrics['accuracy']-baseline)*100:+.1f}%")
        print(
            f"   Model (with odds) vs baseline: {(metrics_odds['accuracy']-baseline)*100:+.1f}%")

        all_results['comparison'] = {
            'baseline_accuracy': baseline,
            'improvement_from_odds': improvement,
            'timestamp': datetime.now().isoformat(),
        }

        self.save_results(all_results)
        self.results = all_results

        print("\n" + "=" * 70)
        print("                    ✓ Training Complete!")
        print("=" * 70)

        return all_results


def main():
    """Run XGBoost training with full tuning"""
    trainer = UFCXGBoostTrainer(
        data_path="data/processed",
        models_path="models"
    )

    results = trainer.run_full_pipeline(
        tune_hyperparameters=False,
        full_tuning=True,                    # Enable Optuna + threshold tuning
        n_trials=100,                         # Optuna trials
        threshold_metric='f1_macro',          # Optimize threshold for F1 Macro
        run_shap_deep_dive=True,
        check_threshold_stability=True        # Verify threshold is stable across folds
    )

    return trainer, results


if __name__ == "__main__":
    trainer, results = main()
