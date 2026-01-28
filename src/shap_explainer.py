"""
FightPredict - SHAP Explainability Deep Dive
=============================================
Enhanced SHAP analysis for UFC fight predictions.

Features:
- Direction of impact analysis (how features influence predictions)
- Individual prediction explanations (for Streamlit UI)
- Feature interaction detection
- Dependence plots for top features
- Force plots for single predictions
- Exportable explanation data for the app

Usage:
    from src.shap_explainer import SHAPExplainer
    
    explainer = SHAPExplainer(model, X_train)
    explainer.analyze(X_test, feature_names)
    explanation = explainer.explain_single_prediction(X_test.iloc[0])
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib
from typing import Dict, List, Tuple, Optional, Union

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Run: pip install shap")

import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """Enhanced SHAP explainability for UFC fight predictions"""

    def __init__(self, model, X_background: pd.DataFrame = None, output_path: str = "models"):
        """
        Initialize SHAP explainer

        Args:
            model: Trained XGBoost model
            X_background: Background data for SHAP (uses training data sample)
            output_path: Directory to save plots and results
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Run: pip install shap")

        self.model = model
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Create explainer
        print("Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(model)

        # Store computed values
        self.shap_values = None
        self.X_explained = None
        self.feature_names = None
        self.expected_value = self.explainer.expected_value

        ev = float(np.array(self.expected_value).flatten()[0])
        print(f"✓ Explainer ready. Base prediction (expected value): {ev:.4f}")

    def compute_shap_values(self, X: pd.DataFrame, max_samples: int = 1000) -> np.ndarray:
        """Compute SHAP values for dataset"""
        # Sample if too large
        if len(X) > max_samples:
            print(f"Sampling {max_samples} from {len(X)} samples for SHAP...")
            X_sample = X.sample(max_samples, random_state=42)
        else:
            X_sample = X.copy()

        print(f"Computing SHAP values for {len(X_sample)} samples...")
        self.shap_values = self.explainer.shap_values(X_sample)
        self.X_explained = X_sample
        self.feature_names = list(X.columns)

        print(f"✓ SHAP values computed. Shape: {self.shap_values.shape}")
        return self.shap_values

    def analyze(self, X: pd.DataFrame, feature_names: List[str] = None,
                max_samples: int = 1000) -> Dict:
        """
        Run full SHAP analysis

        Returns dict with all analysis results
        """
        print("\n" + "=" * 70)
        print("           SHAP Deep Dive Analysis")
        print("=" * 70)

        if feature_names:
            self.feature_names = feature_names

        # Compute SHAP values
        self.compute_shap_values(X, max_samples)

        results = {}

        # 1. Direction of Impact Analysis
        results['direction_analysis'] = self.analyze_direction_of_impact()

        # 2. Feature Importance (mean |SHAP|)
        results['importance'] = self.get_feature_importance()

        # 3. Generate plots
        self.plot_summary()
        self.plot_bar_importance()
        self.plot_top_dependence(top_n=6)

        # 4. Feature interactions
        results['interactions'] = self.analyze_interactions(top_n=5)

        # 5. Save results
        self.save_results(results)

        return results

    def analyze_direction_of_impact(self) -> pd.DataFrame:
        """
        Analyze how each feature impacts predictions

        Returns DataFrame with:
        - mean_shap: Average SHAP value (direction)
        - mean_abs_shap: Average |SHAP| (importance)
        - positive_pct: % of time feature pushes toward F1 win
        - correlation: Correlation between feature value and SHAP value
        """
        print("\n📊 Direction of Impact Analysis")
        print("-" * 50)

        direction_data = []

        for i, feature in enumerate(self.feature_names):
            shap_col = self.shap_values[:, i]
            feature_col = self.X_explained[feature].values

            # Calculate metrics
            mean_shap = shap_col.mean()
            mean_abs_shap = np.abs(shap_col).mean()
            positive_pct = (shap_col > 0).mean() * 100

            # Correlation between feature value and SHAP value
            # High positive = higher feature value → more likely F1 wins
            if np.std(feature_col) > 0 and np.std(shap_col) > 0:
                correlation = np.corrcoef(feature_col, shap_col)[0, 1]
            else:
                correlation = 0

            direction_data.append({
                'feature': feature,
                'mean_shap': mean_shap,
                'mean_abs_shap': mean_abs_shap,
                'positive_pct': positive_pct,
                'correlation': correlation,
                'direction': 'F1 ↑' if correlation > 0.1 else ('F2 ↑' if correlation < -0.1 else 'Mixed')
            })

        df = pd.DataFrame(direction_data).sort_values(
            'mean_abs_shap', ascending=False)

        # Print top features with direction
        print("\nTop 15 Features with Direction of Impact:")
        print(
            f"{'Feature':<30} {'Importance':>10} {'Direction':>10} {'Correlation':>12}")
        print("-" * 65)

        for _, row in df.head(15).iterrows():
            print(
                f"{row['feature']:<30} {row['mean_abs_shap']:>10.4f} {row['direction']:>10} {row['correlation']:>12.3f}")

        return df

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance ranked by mean |SHAP|"""
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(self.shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)

        return importance

    def explain_single_prediction(self, x: Union[pd.Series, pd.DataFrame, np.ndarray],
                                  fighter_1_name: str = "Fighter 1",
                                  fighter_2_name: str = "Fighter 2") -> Dict:
        """
        Explain a single prediction - for Streamlit UI

        Args:
            x: Single sample features
            fighter_1_name: Name of fighter 1
            fighter_2_name: Name of fighter 2

        Returns:
            Dict with prediction explanation
        """
        # Handle input format
        if isinstance(x, pd.DataFrame):
            x_arr = x.values[0] if len(x) == 1 else x.values
            x_df = x.iloc[0] if len(x) == 1 else x
        elif isinstance(x, pd.Series):
            x_arr = x.values
            x_df = x
        else:
            x_arr = x
            x_df = pd.Series(x, index=self.feature_names)

        # Get SHAP values for this prediction
        shap_vals = self.explainer.shap_values(x_arr.reshape(1, -1))[0]

        # Get prediction
        pred_proba = self.model.predict_proba(x_arr.reshape(1, -1))[0]
        pred_class = self.model.predict(x_arr.reshape(1, -1))[0]

        # Build explanation
        contributions = []
        for i, (feat, shap_val) in enumerate(zip(self.feature_names, shap_vals)):
            contributions.append({
                'feature': feat,
                'value': float(x_df.iloc[i]) if isinstance(x_df, pd.Series) else float(x_df[feat]),
                'shap_value': float(shap_val),
                'impact': 'Favors F1' if shap_val > 0 else 'Favors F2',
                'abs_impact': abs(float(shap_val))
            })

        # Sort by absolute impact
        contributions = sorted(
            contributions, key=lambda x: x['abs_impact'], reverse=True)

        explanation = {
            'prediction': {
                'winner': fighter_1_name if pred_class == 1 else fighter_2_name,
                'f1_win_probability': float(pred_proba[1]),
                'f2_win_probability': float(pred_proba[0]),
                'confidence': float(max(pred_proba))
            },
            'base_value': float(self.expected_value),
            'top_factors': contributions[:10],  # Top 10 factors
            'all_contributions': contributions,
            'summary': self._generate_explanation_summary(
                contributions[:5], pred_class, fighter_1_name, fighter_2_name, pred_proba[1]
            )
        }

        return explanation

    def _generate_explanation_summary(self, top_factors: List[Dict], pred_class: int,
                                      f1_name: str, f2_name: str, f1_prob: float) -> str:
        """Generate human-readable explanation summary"""
        winner = f1_name if pred_class == 1 else f2_name
        prob = f1_prob if pred_class == 1 else (1 - f1_prob)

        summary = f"Prediction: {winner} wins ({prob:.1%} confidence)\n\nKey factors:\n"

        for i, factor in enumerate(top_factors, 1):
            feat_name = factor['feature'].replace(
                '_', ' ').replace('diff', 'difference')
            direction = "+" if factor['shap_value'] > 0 else "-"
            impact = "favoring F1" if factor['shap_value'] > 0 else "favoring F2"
            summary += f"  {i}. {feat_name}: {direction}{abs(factor['shap_value']):.3f} ({impact})\n"

        return summary

    def plot_single_prediction(self, x: Union[pd.Series, pd.DataFrame],
                               save_path: str = None, show: bool = True):
        """Generate force plot for single prediction"""
        if isinstance(x, pd.DataFrame):
            x_arr = x.values[0] if len(x) == 1 else x.values
        elif isinstance(x, pd.Series):
            x_arr = x.values
        else:
            x_arr = x

        shap_vals = self.explainer.shap_values(x_arr.reshape(1, -1))

        # Create force plot
        shap.force_plot(
            self.expected_value,
            shap_vals[0],
            x_arr,
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_summary(self):
        """Plot SHAP summary (beeswarm) plot"""
        print("\n📊 Generating SHAP Summary Plot...")

        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            self.shap_values,
            self.X_explained,
            feature_names=self.feature_names,
            show=False,
            max_display=20
        )
        plt.title("SHAP Summary: Feature Impact on Predictions",
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_path / "shap_summary_detailed.png",
                    dpi=150, bbox_inches='tight')
        plt.show()

        print(f"✓ Saved: {self.output_path / 'shap_summary_detailed.png'}")

    def plot_bar_importance(self):
        """Plot SHAP bar importance"""
        print("\n📊 Generating SHAP Bar Plot...")

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_explained,
            feature_names=self.feature_names,
            plot_type="bar",
            show=False,
            max_display=20
        )
        plt.title("SHAP Feature Importance (Mean |SHAP|)",
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_path / "shap_bar_importance.png",
                    dpi=150, bbox_inches='tight')
        plt.show()

        print(f"✓ Saved: {self.output_path / 'shap_bar_importance.png'}")

    def plot_top_dependence(self, top_n: int = 6):
        """Plot dependence plots for top N features"""
        print(f"\n📊 Generating Dependence Plots for Top {top_n} Features...")

        importance = self.get_feature_importance()
        top_features = importance.head(top_n)['feature'].tolist()

        # Create subplot grid
        n_cols = 3
        n_rows = (top_n + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if top_n > 1 else [axes]

        for idx, feature in enumerate(top_features):
            ax = axes[idx]
            feat_idx = self.feature_names.index(feature)

            shap.dependence_plot(
                feat_idx,
                self.shap_values,
                self.X_explained,
                feature_names=self.feature_names,
                ax=ax,
                show=False
            )
            ax.set_title(f"{feature}", fontsize=11)

        # Hide empty subplots
        for idx in range(top_n, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("SHAP Dependence Plots: How Feature Values Affect Predictions",
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_path / "shap_dependence_plots.png",
                    dpi=150, bbox_inches='tight')
        plt.show()

        print(f"✓ Saved: {self.output_path / 'shap_dependence_plots.png'}")

    def analyze_interactions(self, top_n: int = 5) -> pd.DataFrame:
        """
        Analyze feature interactions using SHAP interaction values
        (Simplified version using correlation of SHAP values)
        """
        print(f"\n📊 Analyzing Feature Interactions...")

        importance = self.get_feature_importance()
        top_features = importance.head(top_n)['feature'].tolist()

        interactions = []

        for i, feat1 in enumerate(top_features):
            idx1 = self.feature_names.index(feat1)
            for feat2 in top_features[i+1:]:
                idx2 = self.feature_names.index(feat2)

                # Check if feature values interact in affecting SHAP
                # Simple approach: correlation between product of features and sum of SHAP
                feat1_vals = self.X_explained[feat1].values
                feat2_vals = self.X_explained[feat2].values

                shap1 = self.shap_values[:, idx1]
                shap2 = self.shap_values[:, idx2]

                # Interaction strength: how much does one feature's SHAP depend on other feature's value
                if np.std(feat2_vals) > 0 and np.std(shap1) > 0:
                    interaction_strength = abs(
                        np.corrcoef(feat2_vals, shap1)[0, 1])
                else:
                    interaction_strength = 0

                interactions.append({
                    'feature_1': feat1,
                    'feature_2': feat2,
                    'interaction_strength': interaction_strength
                })

        df = pd.DataFrame(interactions).sort_values(
            'interaction_strength', ascending=False)

        print("\nTop Feature Interactions:")
        print(df.head(10).to_string(index=False))

        return df

    def save_results(self, results: Dict, filename: str = "shap_analysis.json"):
        """Save analysis results to JSON"""
        # Convert DataFrames to dicts for JSON
        save_data = {}

        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                save_data[key] = value.to_dict('records')
            else:
                save_data[key] = value

        filepath = self.output_path / filename
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"\n💾 Results saved: {filepath}")

    def save_explainer(self, filename: str = "shap_explainer.pkl"):
        """Save explainer for later use in Streamlit"""
        filepath = self.output_path / filename

        save_data = {
            'expected_value': self.expected_value,
            'feature_names': self.feature_names,
        }

        joblib.dump(save_data, filepath)
        print(f"💾 Explainer metadata saved: {filepath}")


def run_shap_analysis(model_path: str = "models/xgboost_model.pkl",
                      data_path: str = "data/processed/features_model_ready.csv",
                      output_path: str = "models",
                      include_odds: bool = False):
    """
    Run full SHAP analysis on trained model

    Args:
        model_path: Path to trained XGBoost model
        data_path: Path to feature data
        output_path: Directory for outputs
        include_odds: Whether using odds model
    """
    print("=" * 70)
    print("        FightPredict - SHAP Explainability Analysis")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from {model_path}...")
    model_data = joblib.load(model_path)

    # Extract the actual model from the dictionary
    if isinstance(model_data, dict):
        model = model_data['model']
        threshold = model_data.get('optimal_threshold', 0.5)
        print(f"  Threshold: {threshold:.4f}")
    else:
        model = model_data

    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']

    # Split to get test set (same split as training)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Test set: {len(X_test)} samples, {len(X_test.columns)} features")

    # Create explainer and run analysis
    explainer = SHAPExplainer(model, output_path=output_path)
    results = explainer.analyze(X_test, feature_names=list(X.columns))

    # Demo: Explain a single prediction
    print("\n" + "=" * 70)
    print("        Example: Single Prediction Explanation")
    print("=" * 70)

    sample_idx = 0
    sample = X_test.iloc[sample_idx]
    actual = y_test.iloc[sample_idx]

    explanation = explainer.explain_single_prediction(
        sample,
        fighter_1_name="Fighter A",
        fighter_2_name="Fighter B"
    )

    print(
        f"\nActual outcome: {'Fighter A wins' if actual == 1 else 'Fighter B wins'}")
    print(explanation['summary'])

    # Save explainer for Streamlit
    explainer.save_explainer()

    print("\n" + "=" * 70)
    print("        ✓ SHAP Analysis Complete!")
    print("=" * 70)

    return explainer, results


if __name__ == "__main__":
    # Run analysis on model without odds
    explainer, results = run_shap_analysis(
        model_path="models/xgboost_model.pkl",
        data_path="data/processed/features_model_ready.csv",
        output_path="models"
    )
