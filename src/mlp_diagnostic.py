"""
FightPredict - MLP Diagnostic Tool
===================================
Investigates why MLP outputs low/extreme probabilities.

Checks:
1. Feature distribution comparison (training vs inference)
2. Scaler statistics
3. MLP internal activations
4. Probability distribution on training data

Usage:
    python src/mlp_diagnostic.py
    python src/mlp_diagnostic.py --fighter1 "Brandon Royval" --fighter2 "Manel Kape"
    python src/mlp_diagnostic.py --plot
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_training_data(data_path: str = "data/processed") -> Tuple[pd.DataFrame, pd.Series]:
    """Load the training data to compare distributions."""
    filepath = Path(data_path) / "features_model_ready.csv"
    df = pd.read_csv(filepath)
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y


def load_mlp_model(models_path: str = "models", with_odds: bool = False):
    """Load MLP model and scaler."""
    from train_mlp import ImprovedMLPClassifier

    models_path = Path(models_path)

    # Select model file
    if with_odds:
        filename = "mlp_model_with_odds.pt"
    else:
        filename = "mlp_model.pt"

    filepath = models_path / filename
    if not filepath.exists():
        raise FileNotFoundError(f"MLP model not found: {filepath}")

    data = torch.load(filepath, map_location=DEVICE, weights_only=False)
    config = data['model_config']

    model = ImprovedMLPClassifier(**config)
    model.load_state_dict(data['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    scaler = data['scaler']
    threshold = data.get('optimal_threshold', 0.5)

    print(f"  ✓ Loaded MLP from {filename}")
    print(f"    Architecture: {config['hidden_sizes']}")
    print(f"    Activation: {config.get('activation', 'leaky_relu')}")

    return model, scaler, threshold, config


def analyze_scaler(scaler: StandardScaler, feature_names: list) -> pd.DataFrame:
    """Analyze the scaler's learned statistics."""
    stats = pd.DataFrame({
        'feature': feature_names,
        'mean': scaler.mean_,
        'std': scaler.scale_,
        'var': scaler.var_
    })
    return stats


def analyze_feature_distribution(
    X_train: pd.DataFrame,
    X_inference: np.ndarray,
    scaler: StandardScaler,
    feature_names: list
) -> pd.DataFrame:
    """Compare training vs inference feature distributions."""

    # Scale both
    X_train_scaled = scaler.transform(X_train.values)
    X_inference_scaled = scaler.transform(X_inference.reshape(1, -1))

    results = {
        'feature': [],
        'train_mean': [],
        'train_std': [],
        'train_min': [],
        'train_max': [],
        'inference_value': [],
        'inference_scaled': [],
        'z_score': [],
        'percentile': [],
        'is_outlier': []
    }

    for i, fname in enumerate(feature_names):
        train_vals = X_train_scaled[:, i]
        inf_val = X_inference_scaled[0, i]

        # Calculate z-score (how many stds from mean)
        z_score = inf_val  # Already scaled, so this IS the z-score

        # Calculate percentile
        percentile = (train_vals < inf_val).mean() * 100

        # Flag outliers (|z| > 3)
        is_outlier = abs(z_score) > 3

        results['feature'].append(fname)
        results['train_mean'].append(train_vals.mean())
        results['train_std'].append(train_vals.std())
        results['train_min'].append(train_vals.min())
        results['train_max'].append(train_vals.max())
        results['inference_value'].append(X_inference[i])
        results['inference_scaled'].append(inf_val)
        results['z_score'].append(z_score)
        results['percentile'].append(percentile)
        results['is_outlier'].append(is_outlier)

    return pd.DataFrame(results)


def analyze_mlp_activations(
    model,
    X_scaled: np.ndarray,
    verbose: bool = True
) -> Dict:
    """Analyze internal MLP activations to find saturation or dead neurons."""

    X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)

    activations = {}

    # Hook to capture activations
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu().numpy()
        return hook

    # Register hooks for each layer (ImprovedMLPClassifier uses model.layers)
    hooks = []

    if hasattr(model, 'layers'):
        for i, layer in enumerate(model.layers):
            hook = layer.register_forward_hook(get_activation(f'layer_{i}'))
            hooks.append(hook)

        if hasattr(model, 'output_layer'):
            hook = model.output_layer.register_forward_hook(
                get_activation('output_layer'))
            hooks.append(hook)
    else:
        print("Warning: Unknown model architecture, skipping activation analysis")
        return {}

    # Forward pass
    with torch.no_grad():
        output = model(X_tensor)

    # Add final output
    activations['final_output'] = output.detach().cpu().numpy()

    # Remove hooks
    for hook in hooks:
        hook.remove()

    if verbose:
        print("\n" + "="*60)
        print("MLP Activation Analysis")
        print("="*60)

        for name, act in activations.items():
            act_flat = act.flatten()
            print(f"\n{name}:")
            print(f"  Shape: {act.shape}")
            print(f"  Min: {act_flat.min():.4f}")
            print(f"  Max: {act_flat.max():.4f}")
            print(f"  Mean: {act_flat.mean():.4f}")
            print(f"  Std: {act_flat.std():.4f}")

            # Check for dead neurons (always 0 after activation)
            if 'layer' in name:
                dead = (act_flat == 0).mean() * 100
                if dead > 0:
                    print(f"  Dead neurons (=0): {dead:.1f}%")

            # Check sigmoid saturation for final output
            if 'output' in name or 'final' in name:
                near_0 = (act_flat < 0.1).mean() * 100
                near_1 = (act_flat > 0.9).mean() * 100
                print(f"  Near 0 (<0.1): {near_0:.1f}%")
                print(f"  Near 1 (>0.9): {near_1:.1f}%")

    return activations


def analyze_probability_distribution(
    model,
    scaler: StandardScaler,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_size: int = 1000
) -> Dict:
    """Analyze the distribution of probabilities on training data."""

    # Sample from training data
    if len(X_train) > sample_size:
        idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train.iloc[idx].values
        y_sample = y_train.iloc[idx].values
    else:
        X_sample = X_train.values
        y_sample = y_train.values

    # Scale and predict
    X_scaled = scaler.transform(X_sample)
    X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)

    with torch.no_grad():
        probs = model(X_tensor).cpu().numpy().flatten()

    # Analyze distribution
    results = {
        'overall': {
            'mean': probs.mean(),
            'std': probs.std(),
            'min': probs.min(),
            'max': probs.max(),
            'median': np.median(probs),
            'q25': np.percentile(probs, 25),
            'q75': np.percentile(probs, 75),
        },
        'class_0': {  # F2 wins
            'mean': probs[y_sample == 0].mean(),
            'std': probs[y_sample == 0].std(),
            'count': (y_sample == 0).sum(),
        },
        'class_1': {  # F1 wins
            'mean': probs[y_sample == 1].mean(),
            'std': probs[y_sample == 1].std(),
            'count': (y_sample == 1).sum(),
        },
        'probs': probs,
        'labels': y_sample
    }

    print("\n" + "="*60)
    print("Probability Distribution on Training Data")
    print("="*60)
    print(f"\nOverall Statistics:")
    print(f"  Mean:   {results['overall']['mean']:.4f}")
    print(f"  Std:    {results['overall']['std']:.4f}")
    print(f"  Min:    {results['overall']['min']:.4f}")
    print(f"  Max:    {results['overall']['max']:.4f}")
    print(f"  Median: {results['overall']['median']:.4f}")
    print(f"  Q25:    {results['overall']['q25']:.4f}")
    print(f"  Q75:    {results['overall']['q75']:.4f}")

    print(f"\nBy Class:")
    print(
        f"  F2 Wins (class=0): mean={results['class_0']['mean']:.4f}, std={results['class_0']['std']:.4f}, n={results['class_0']['count']}")
    print(
        f"  F1 Wins (class=1): mean={results['class_1']['mean']:.4f}, std={results['class_1']['std']:.4f}, n={results['class_1']['count']}")

    # Check for problematic distributions
    print(f"\nDistribution Health Check:")
    if results['overall']['std'] < 0.1:
        print(
            f"  ⚠️ LOW VARIANCE: std={results['overall']['std']:.4f} - model outputs are clustered")
    else:
        print(
            f"  ✓ Variance looks reasonable: std={results['overall']['std']:.4f}")

    separation = results['class_1']['mean'] - results['class_0']['mean']
    print(f"  Class separation (mean diff): {separation:.4f}")
    if separation < 0.1:
        print(f"  ⚠️ LOW SEPARATION: Classes not well separated")
    else:
        print(f"  ✓ Classes are reasonably separated")

    return results


def run_full_diagnostic(
    fighter1: Optional[str] = None,
    fighter2: Optional[str] = None,
    with_odds: bool = False,
    models_path: str = "models",
    data_path: str = "data/processed"
) -> Dict:
    """Run complete diagnostic analysis."""

    print("="*70)
    print("FightPredict MLP Diagnostic Tool")
    print("="*70)

    # Load model
    print("\n📦 Loading MLP model...")
    model, scaler, threshold, config = load_mlp_model(
        models_path, with_odds=with_odds)
    print(f"  Threshold: {threshold:.4f}")

    # Load training data
    print("\n📦 Loading training data...")
    if with_odds:
        filepath = Path(data_path) / "features_model_ready_with_odds.csv"
    else:
        filepath = Path(data_path) / "features_model_ready.csv"

    df = pd.read_csv(filepath)
    X_train = df.drop('target', axis=1)
    y_train = df['target']
    feature_names = X_train.columns.tolist()
    print(f"  Loaded {len(X_train)} samples, {len(feature_names)} features")

    # Analyze scaler
    print("\n📊 Scaler Statistics (top 10 by std):")
    scaler_stats = analyze_scaler(scaler, feature_names)
    print(scaler_stats.nlargest(10, 'std')[
          ['feature', 'mean', 'std']].to_string(index=False))

    # Analyze probability distribution on training data
    prob_dist = analyze_probability_distribution(
        model, scaler, X_train, y_train)

    # If fighters specified, analyze their matchup
    if fighter1 and fighter2:
        print("\n" + "="*70)
        print(f"Analyzing Matchup: {fighter1} vs {fighter2}")
        print("="*70)

        from generate_matchup_features import MatchupFeatureGenerator

        generator = MatchupFeatureGenerator()
        generator.load_data()

        features, names = generator.generate_features(
            fighter1=fighter1,
            fighter2=fighter2,
            include_odds=with_odds
        )

        # Feature distribution comparison
        print("\n📊 Feature Distribution Comparison:")
        feat_comparison = analyze_feature_distribution(
            X_train, features, scaler, feature_names)

        # Show outliers
        outliers = feat_comparison[feat_comparison['is_outlier']]
        if len(outliers) > 0:
            print(f"\n⚠️ OUTLIER FEATURES ({len(outliers)} found):")
            for _, row in outliers.iterrows():
                print(f"  {row['feature']}: z-score={row['z_score']:.2f}, "
                      f"value={row['inference_value']:.4f}, "
                      f"percentile={row['percentile']:.1f}%")
        else:
            print("\n✓ No extreme outliers detected")

        # Show most extreme features
        print("\n📊 Top 10 Most Extreme Features (by |z-score|):")
        feat_comparison['abs_z_score'] = feat_comparison['z_score'].abs()
        extreme = feat_comparison.nlargest(10, 'abs_z_score')
        for _, row in extreme.iterrows():
            print(f"  {row['feature']}: z={row['z_score']:+.2f}, "
                  f"raw={row['inference_value']:.4f}, "
                  f"percentile={row['percentile']:.1f}%")

        # Analyze activations
        X_scaled = scaler.transform(features.reshape(1, -1))
        activations = analyze_mlp_activations(model, X_scaled)

        # Final prediction
        X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)
        with torch.no_grad():
            prob = model(X_tensor).cpu().numpy().flatten()[0]

        print("\n" + "="*60)
        print("Final Prediction")
        print("="*60)
        print(f"  P({fighter1} wins) = {prob:.4f} ({prob*100:.1f}%)")
        print(f"  Threshold = {threshold:.4f}")
        print(f"  Margin = {prob - threshold:+.4f}")

        if prob < threshold:
            print(f"  → Prediction: {fighter2} wins")
        else:
            print(f"  → Prediction: {fighter1} wins")

        # Compare to training distribution
        percentile = (prob_dist['probs'] < prob).mean() * 100
        print(
            f"\n  This probability is at the {percentile:.1f}th percentile of training predictions")

        if percentile < 5:
            print(f"  ⚠️ This is an unusually LOW probability")
        elif percentile > 95:
            print(f"  ⚠️ This is an unusually HIGH probability")
        else:
            print(f"  ✓ This probability is within normal range")

        return {
            'features': features,
            'feature_comparison': feat_comparison,
            'activations': activations,
            'probability': prob,
            'prob_distribution': prob_dist,
            'percentile': percentile
        }

    return {
        'scaler_stats': scaler_stats,
        'prob_distribution': prob_dist
    }


def plot_diagnostic_results(results: Dict, output_path: str = "models"):
    """Create diagnostic plots."""
    output_path = Path(output_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Probability distribution
    ax = axes[0, 0]
    probs = results['prob_distribution']['probs']
    labels = results['prob_distribution']['labels']

    ax.hist(probs[labels == 0], bins=30, alpha=0.6,
            label='F2 Wins (class=0)', color='blue', density=True)
    ax.hist(probs[labels == 1], bins=30, alpha=0.6,
            label='F1 Wins (class=1)', color='red', density=True)

    if 'probability' in results:
        ax.axvline(results['probability'], color='green', linestyle='--',
                   linewidth=2, label=f"Inference: {results['probability']:.3f}")

    ax.set_xlabel('Probability (P(F1 wins))')
    ax.set_ylabel('Density')
    ax.set_title('MLP Probability Distribution by Class')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Feature z-scores (if available)
    ax = axes[0, 1]
    if 'feature_comparison' in results:
        feat_comp = results['feature_comparison']
        # Show top 20 most extreme
        extreme = feat_comp.nlargest(20, 'abs_z_score')
        z_scores = extreme['z_score'].values
        feature_labels = extreme['feature'].values

        colors = ['red' if abs(z) > 3 else 'steelblue' for z in z_scores]
        ax.barh(range(len(z_scores)), z_scores, color=colors)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.axvline(-3, color='red', linestyle='--',
                   alpha=0.5, label='Outlier threshold')
        ax.axvline(3, color='red', linestyle='--', alpha=0.5)
        ax.set_yticks(range(len(z_scores)))
        ax.set_yticklabels(feature_labels, fontsize=8)
        ax.set_xlabel('Z-Score')
        ax.set_title('Top 20 Most Extreme Features')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No matchup analyzed', ha='center', va='center')
        ax.set_title('Feature Z-Scores')

    # Plot 3: Cumulative probability distribution
    ax = axes[1, 0]
    sorted_probs = np.sort(probs)
    cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
    ax.plot(sorted_probs, cumulative, linewidth=2)

    if 'probability' in results:
        ax.axvline(results['probability'], color='green',
                   linestyle='--', linewidth=2)
        ax.axhline(results['percentile']/100, color='green',
                   linestyle='--', linewidth=2)
        ax.scatter([results['probability']], [results['percentile']/100],
                   color='green', s=100, zorder=5,
                   label=f"Inference ({results['percentile']:.1f}%ile)")
        ax.legend()

    ax.set_xlabel('Probability')
    ax.set_ylabel('Cumulative Distribution')
    ax.set_title('Probability CDF (where does inference fall?)')
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = "Diagnostic Summary\n" + "="*35 + "\n\n"
    summary_text += f"Training Probability Stats:\n"
    summary_text += f"  Mean:   {results['prob_distribution']['overall']['mean']:.4f}\n"
    summary_text += f"  Std:    {results['prob_distribution']['overall']['std']:.4f}\n"
    summary_text += f"  Min:    {results['prob_distribution']['overall']['min']:.4f}\n"
    summary_text += f"  Max:    {results['prob_distribution']['overall']['max']:.4f}\n\n"

    summary_text += f"Class Separation:\n"
    summary_text += f"  F1 mean: {results['prob_distribution']['class_1']['mean']:.4f}\n"
    summary_text += f"  F2 mean: {results['prob_distribution']['class_0']['mean']:.4f}\n"
    sep = results['prob_distribution']['class_1']['mean'] - \
        results['prob_distribution']['class_0']['mean']
    summary_text += f"  Diff:    {sep:.4f}\n\n"

    # Health assessment
    std = results['prob_distribution']['overall']['std']
    if std < 0.1:
        summary_text += "⚠️ LOW VARIANCE - outputs clustered\n"
    else:
        summary_text += "✓ Variance OK\n"

    if sep < 0.1:
        summary_text += "⚠️ LOW CLASS SEPARATION\n"
    else:
        summary_text += "✓ Class separation OK\n"

    if 'probability' in results:
        summary_text += f"\nInference Result:\n"
        summary_text += f"  Probability: {results['probability']:.4f}\n"
        summary_text += f"  Percentile:  {results['percentile']:.1f}%\n"

        if 'feature_comparison' in results:
            n_outliers = results['feature_comparison']['is_outlier'].sum()
            summary_text += f"  Outliers:    {n_outliers}\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path / "mlp_diagnostic.png", dpi=150)
    plt.show()

    print(f"\n✓ Diagnostic plot saved to {output_path / 'mlp_diagnostic.png'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLP Diagnostic Tool")
    parser.add_argument("--fighter1", "-f1", type=str, help="Fighter 1 name")
    parser.add_argument("--fighter2", "-f2", type=str, help="Fighter 2 name")
    parser.add_argument("--with-odds", action="store_true",
                        help="Use model with odds")
    parser.add_argument("--models-path", type=str, default="models")
    parser.add_argument("--data-path", type=str, default="data/processed")
    parser.add_argument("--plot", action="store_true",
                        help="Generate diagnostic plots")

    args = parser.parse_args()

    results = run_full_diagnostic(
        fighter1=args.fighter1,
        fighter2=args.fighter2,
        with_odds=args.with_odds,
        models_path=args.models_path,
        data_path=args.data_path
    )

    if args.plot:
        plot_diagnostic_results(results, args.models_path)
