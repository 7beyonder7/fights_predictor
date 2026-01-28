"""
FightPredict RAG Analyzer
=========================
Natural language fight analysis using RAG (Retrieval-Augmented Generation).

Combines:
- FAISS similarity search for historical fight context
- XGBoost + MLP model predictions
- LLM-powered natural language analysis

Supports multiple LLM providers:
- Ollama (local) - default for development
- Claude API (Anthropic)
- OpenAI API

Usage:
    from rag_analyzer import FightAnalyzer
    
    # Local model (Ollama)
    analyzer = FightAnalyzer(provider="ollama", model="llama3.1:8b")
    
    # Claude API
    analyzer = FightAnalyzer(provider="claude", model="claude-sonnet-4-20250514")
    
    # OpenAI API
    analyzer = FightAnalyzer(provider="openai", model="gpt-4o")
    
    # Analyze a matchup
    analysis = analyzer.analyze_matchup(fighter1_features, fighter2_features)
    print(analysis)
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

# ML imports
import joblib
import torch

# Optional imports with graceful fallbacks
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AnalyzerConfig:
    """Configuration for the FightAnalyzer."""
    # LLM settings
    provider: Literal["ollama", "claude", "openai"] = "ollama"
    model: str = "llama3.1:8b"
    temperature: float = 0.7
    max_tokens: int = 2000

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"

    # Retrieval settings
    similar_fights_k: int = 10

    # Model paths
    xgboost_path: str = "models/xgboost_model_with_odds.pkl"
    xgboost_no_odds_path: str = "models/xgboost_model.pkl"
    mlp_path: str = "models/mlp_model_with_odds.pt"
    mlp_no_odds_path: str = "models/mlp_model.pt"
    faiss_index_path: str = "faiss_index"

    # Data paths
    fights_path: str = "data/processed/fights_cleaned.csv"
    fighters_path: str = "data/processed/fighters.csv"
    features_path: str = "data/processed/features_model_ready_with_odds.csv"


# =============================================================================
# LLM Provider Abstraction
# =============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, model: str, base_url: str, temperature: float, max_tokens: int):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def is_available(self) -> bool:
        if not HAS_REQUESTS:
            return False
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if not HAS_REQUESTS:
            raise RuntimeError(
                "requests library not installed. Run: pip install requests")

        url = f"{self.base_url}/api/generate"

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: ollama serve"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")


class ClaudeProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, model: str, temperature: float, max_tokens: int):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None

        if HAS_ANTHROPIC:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)

    def is_available(self) -> bool:
        return HAS_ANTHROPIC and self.client is not None

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if not self.is_available():
            raise RuntimeError(
                "Claude API not available. Install anthropic and set ANTHROPIC_API_KEY: "
                "pip install anthropic && export ANTHROPIC_API_KEY=your_key"
            )

        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        # Only add temperature if not using extended thinking
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, model: str, temperature: float, max_tokens: int):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None

        if HAS_OPENAI:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)

    def is_available(self) -> bool:
        return HAS_OPENAI and self.client is not None

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if not self.is_available():
            raise RuntimeError(
                "OpenAI API not available. Install openai and set OPENAI_API_KEY: "
                "pip install openai && export OPENAI_API_KEY=your_key"
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content


def create_provider(config: AnalyzerConfig) -> LLMProvider:
    """Factory function to create the appropriate LLM provider."""
    if config.provider == "ollama":
        return OllamaProvider(
            model=config.model,
            base_url=config.ollama_base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    elif config.provider == "claude":
        return ClaudeProvider(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    elif config.provider == "openai":
        return OpenAIProvider(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


# =============================================================================
# Model Loaders - FIXED VERSION
# =============================================================================

class ModelEnsemble:
    """
    Loads and manages XGBoost and MLP models for predictions.

    FIXES APPLIED:
    1. Proper threshold-based predictions for each model
    2. Meaningful confidence calculation
    3. Correct ensemble logic using voting + calibrated probabilities
    """

    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.xgboost_model = None
        self.xgboost_threshold = 0.5
        self.mlp_model = None
        self.mlp_scaler = None
        self.mlp_threshold = 0.5
        self.mlp_config = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.use_odds = True

    def load_xgboost(self, with_odds: bool = True) -> bool:
        """Load XGBoost model."""
        path = self.config.xgboost_path if with_odds else self.config.xgboost_no_odds_path
        try:
            data = joblib.load(path)
            self.xgboost_model = data['model']
            self.xgboost_threshold = data.get('optimal_threshold', 0.5)
            self.use_odds = with_odds
            print(f"✓ Loaded XGBoost model from {path}")
            print(f"  Threshold: {self.xgboost_threshold:.3f}")
            return True
        except FileNotFoundError:
            print(f"✗ XGBoost model not found at {path}")
            return False
        except Exception as e:
            print(f"✗ Error loading XGBoost: {e}")
            return False

    def load_mlp(self, with_odds: bool = True) -> bool:
        """Load MLP model (supports both v1 and v2)."""
        # Try v2 first, fall back to v1
        base_path = self.config.mlp_no_odds_path if not with_odds else self.config.mlp_path
        v2_path = base_path.replace('.pt', '_v2.pt').replace(
            'mlp_model', 'mlp_model')

        # Check for v2 model first
        paths_to_try = []
        if not with_odds:
            paths_to_try = ['models/mlp_model.pt',
                            self.config.mlp_no_odds_path]
        else:
            paths_to_try = [
                'models/mlp_model_with_odds.pt', self.config.mlp_path]

        for path in paths_to_try:
            try:
                data = torch.load(
                    path, map_location=self.device, weights_only=False)
                self.mlp_config = data['model_config']

                # Check model version
                model_version = data.get('model_version', 'v1')

                if model_version == 'v2_improved' or 'activation' in self.mlp_config:
                    # Load v2 model (now in train_mlp.py)
                    try:
                        from train_mlp import ImprovedMLPClassifier
                    except ImportError:
                        from src.train_mlp import ImprovedMLPClassifier
                    self.mlp_model = ImprovedMLPClassifier(**self.mlp_config)
                    print(f"✓ Loaded MLP model from {path}")
                else:
                    # Load v1 model (backward compatibility alias)
                    try:
                        from train_mlp import UFCMLPClassifier
                    except ImportError:
                        from src.train_mlp import UFCMLPClassifier
                    self.mlp_model = UFCMLPClassifier(**self.mlp_config)
                    print(f"✓ Loaded MLP model from {path}")

                self.mlp_model.load_state_dict(data['model_state_dict'])
                self.mlp_model.to(self.device)
                self.mlp_model.eval()

                self.mlp_scaler = data['scaler']
                self.mlp_threshold = data.get('optimal_threshold', 0.5)
                self.use_odds = with_odds

                print(f"  Threshold: {self.mlp_threshold:.3f}")
                print(f"  Device: {self.device}")

                # Show architecture info
                if 'activation' in self.mlp_config:
                    print(f"  Activation: {self.mlp_config['activation']}")
                print(f"  Architecture: {self.mlp_config['hidden_sizes']}")

                return True

            except FileNotFoundError:
                continue
            except ImportError as e:
                print(f"✗ Import error: {e}")
                continue
            except Exception as e:
                print(f"✗ Error loading from {path}: {e}")
                continue

        print(f"✗ No MLP model found")
        return False

    def predict_xgboost(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Get XGBoost prediction with probability.

        Returns probability that Fighter 1 wins (target=1).
        Uses optimized threshold for binary prediction.
        """
        if self.xgboost_model is None:
            return None

        features = np.atleast_2d(features)
        # predict_proba returns [P(class=0), P(class=1)]
        # P(class=1) = P(Fighter 1 wins)
        proba_f1_wins = self.xgboost_model.predict_proba(features)[:, 1][0]

        # Use optimized threshold for prediction
        prediction = int(proba_f1_wins >= self.xgboost_threshold)

        # Calculate confidence relative to threshold
        # How far is the probability from the decision boundary?
        if prediction == 1:
            # Predicted F1 wins: confidence = how much above threshold
            confidence = (proba_f1_wins - self.xgboost_threshold) / \
                (1.0 - self.xgboost_threshold)
        else:
            # Predicted F2 wins: confidence = how much below threshold
            confidence = (self.xgboost_threshold -
                          proba_f1_wins) / self.xgboost_threshold

        confidence = float(np.clip(confidence, 0, 1))

        return {
            "model": "XGBoost",
            "prediction": prediction,  # 1 = F1 wins, 0 = F2 wins
            "probability_f1": float(proba_f1_wins),  # Raw P(F1 wins)
            "threshold": self.xgboost_threshold,
            "confidence": confidence,  # 0-1 scale, how confident in the prediction
            # Positive = F1, Negative = F2
            "margin": float(proba_f1_wins - self.xgboost_threshold)
        }

    def predict_mlp(self, features: np.ndarray, debug: bool = False) -> Dict[str, Any]:
        """
        Get MLP prediction with probability.

        Returns probability that Fighter 1 wins (target=1).
        Uses optimized threshold for binary prediction.
        """
        if self.mlp_model is None:
            return None

        features = np.atleast_2d(features)
        X_scaled = self.mlp_scaler.transform(features)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            proba_f1_wins = self.mlp_model(X_tensor).cpu().numpy().flatten()[0]

        # Clamp extreme values to avoid 0% / 100%
        proba_f1_wins = float(np.clip(proba_f1_wins, 0.001, 0.999))

        if debug:
            print(f"  [MLP Debug] Raw proba: {proba_f1_wins:.6f}")
            print(f"  [MLP Debug] Threshold: {self.mlp_threshold:.3f}")
            print(
                f"  [MLP Debug] Scaled features min/max: {X_scaled.min():.2f} / {X_scaled.max():.2f}")

        # Use optimized threshold for prediction
        prediction = int(proba_f1_wins >= self.mlp_threshold)

        # Calculate confidence relative to threshold
        if prediction == 1:
            confidence = (proba_f1_wins - self.mlp_threshold) / \
                (1.0 - self.mlp_threshold)
        else:
            confidence = (self.mlp_threshold - proba_f1_wins) / \
                self.mlp_threshold

        confidence = float(np.clip(confidence, 0, 1))

        return {
            "model": "MLP",
            "prediction": prediction,
            "probability_f1": float(proba_f1_wins),
            "threshold": self.mlp_threshold,
            "confidence": confidence,
            "margin": float(proba_f1_wins - self.mlp_threshold)
        }

    def predict_ensemble(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Get ensemble prediction combining both models.

        LOGIC:
        1. Confidence-weighted average of probabilities from both models
           (more confident model has more influence)
        2. Use weighted probability for prediction (threshold 0.5)
        3. Confidence based on how far from 0.5 the ensemble probability is
        """
        xgb_pred = self.predict_xgboost(features)
        mlp_pred = self.predict_mlp(features)

        predictions = []
        if xgb_pred:
            predictions.append(xgb_pred)
        if mlp_pred:
            predictions.append(mlp_pred)

        if not predictions:
            return None

        # Single model case
        if len(predictions) == 1:
            pred = predictions[0]
            return {
                "ensemble_prediction": pred["prediction"],
                "ensemble_probability_f1": pred["probability_f1"],
                "ensemble_confidence": pred["confidence"],
                "models_agree": True,
                "individual_predictions": predictions,
                "method": "single_model"
            }

        # Two model case - confidence-weighted probability averaging
        xgb_prob = xgb_pred["probability_f1"]
        mlp_prob = mlp_pred["probability_f1"]
        xgb_conf = xgb_pred["confidence"]
        mlp_conf = mlp_pred["confidence"]

        # Confidence-weighted ensemble
        # When one model is much more confident, weight it higher
        total_conf = xgb_conf + mlp_conf
        if total_conf > 0:
            xgb_weight = xgb_conf / total_conf
            mlp_weight = mlp_conf / total_conf
            ensemble_prob_f1 = xgb_prob * xgb_weight + mlp_prob * mlp_weight
        else:
            # Fallback to simple average if both have zero confidence
            ensemble_prob_f1 = (xgb_prob + mlp_prob) / 2
            xgb_weight = 0.5
            mlp_weight = 0.5

        # Prediction based on ensemble probability
        ensemble_pred = 1 if ensemble_prob_f1 >= 0.5 else 0

        # Check if individual models agree
        xgb_vote = xgb_pred["prediction"]
        mlp_vote = mlp_pred["prediction"]
        models_agree = (xgb_vote == mlp_vote)

        # Confidence: how far from 50-50 is the ensemble probability
        # Scale to 0-1 where 0 = 50% prob, 1 = 100% or 0% prob
        ensemble_confidence = abs(ensemble_prob_f1 - 0.5) * 2

        # Reduce confidence if models disagree
        if not models_agree:
            ensemble_confidence *= 0.7
            method = "probability_average_disagree"
        else:
            method = "probability_average"

        return {
            "ensemble_prediction": ensemble_pred,
            "ensemble_probability_f1": float(np.clip(ensemble_prob_f1, 0.01, 0.99)),
            "ensemble_confidence": float(np.clip(ensemble_confidence, 0, 1)),
            "models_agree": models_agree,
            "individual_predictions": predictions,
            "method": method,
            "xgb_weight": float(xgb_weight),
            "mlp_weight": float(mlp_weight)
        }


# =============================================================================
# FAISS Integration (Stub - integrates with your existing faiss_search.py)
# =============================================================================

class SimilaritySearchWrapper:
    """Wrapper for FAISS similarity search integration."""

    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.searcher = None
        self._fighter_db = None  # Unified fighter database

    def load(self) -> bool:
        """Load FAISS index and fighter data."""
        try:
            # Import FAISS search module
            try:
                from faiss_search import FightSimilaritySearch
            except ImportError:
                from src.faiss_search import FightSimilaritySearch

            self.searcher = FightSimilaritySearch()
            self.searcher.load_index(self.config.faiss_index_path)

            # Load unified fighter database
            try:
                from fighters import FighterDatabase
            except ImportError:
                from src.fighters import FighterDatabase
            self._fighter_db = FighterDatabase(
                fights_path=self.config.fights_path,
                fighters_path=self.config.fighters_path
            )
            self._fighter_db.load()

            print(
                f"✓ Loaded FAISS index with {len(self.searcher.fight_metadata)} fights")
            return True
        except ImportError as e:
            print(f"✗ Import error: {e}")
            return False
        except Exception as e:
            print(f"✗ Error loading FAISS index: {e}")
            return False

    def find_similar_fights(
        self,
        features: np.ndarray,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """Find similar historical fights."""
        if self.searcher is None:
            return []

        # FAISS index uses 101 features (no odds)
        # Trim to 101 if we have 110
        features = np.atleast_2d(features)
        if features.shape[1] > 101:
            features = features[:, :101]

        results = self.searcher.find_similar_fights(features, k=k)
        return results

    def analyze_similar_fights(
        self,
        features: np.ndarray,
        k: int = 10
    ) -> Dict[str, Any]:
        """Analyze similar fights and get prediction."""
        if self.searcher is None:
            return {}

        if not hasattr(self.searcher, 'index') or self.searcher.index is None:
            return {}

        # FAISS index uses 101 features (no odds)
        features = np.atleast_2d(features)
        if features.shape[1] > 101:
            features = features[:, :101]

        return self.searcher.analyze_similar_fights(features, k=k)

    def get_fighter_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get fighter stats by name using unified database."""
        if self._fighter_db is None:
            return None

        snapshot = self._fighter_db.get_snapshot(name)
        return snapshot.to_dict() if snapshot else None

    def search_fighter(self, query: str) -> List[str]:
        """Search for fighters by partial name using unified database."""
        if self._fighter_db is not None:
            return self._fighter_db.search(query)

        # Fallback to FAISS searcher if available
        if self.searcher is not None and hasattr(self.searcher, 'search_fighter'):
            results = self.searcher.search_fighter(query)
            # Return just names if results are dicts
            if results and isinstance(results[0], dict):
                return [r.get('name', str(r)) for r in results]
            return results

        return []


# =============================================================================
# Prompt Templates
# =============================================================================

SYSTEM_PROMPT = """You are an expert UFC analyst with deep knowledge of MMA techniques, fighter styles, and fight prediction. You provide insightful, data-driven analysis while being engaging and accessible.

Your analysis should:
1. Be grounded in the data provided (model predictions, similar fights, fighter stats)
2. Explain the key factors driving the prediction
3. Discuss stylistic matchup considerations
4. Acknowledge uncertainty where appropriate
5. Be engaging and readable, not just a data dump

Format your response with clear sections but keep it conversational."""

ANALYSIS_PROMPT_TEMPLATE = """Analyze this UFC matchup and provide your expert prediction:

**FIGHTERS:**
- Fighter 1 (Red Corner): {fighter1_name}
- Fighter 2 (Blue Corner): {fighter2_name}

**MODEL PREDICTIONS:**
{model_predictions}

**SIMILAR HISTORICAL FIGHTS:**
{similar_fights}

**KEY STATS COMPARISON:**
{feature_summary}

Based on this data, provide a comprehensive analysis including:
1. **PREDICTION**: Who wins and how (method, round if possible)
2. **CONFIDENCE**: Your confidence level and why
3. **KEY FACTORS**: The 3-4 most important factors driving this prediction
4. **STYLE BREAKDOWN**: How the styles match up
5. **PATHS TO VICTORY**: How each fighter could win
6. **UPSET POTENTIAL**: What could cause an upset

Be specific and reference the data provided."""


# =============================================================================
# Formatting Helpers - UPDATED
# =============================================================================

def format_model_predictions(ensemble_result: Dict, f1_name: str, f2_name: str) -> str:
    """Format model predictions for prompt - FIXED VERSION."""
    if not ensemble_result:
        return "Model predictions unavailable."

    lines = []

    # Overall prediction
    winner = f1_name if ensemble_result["ensemble_prediction"] == 1 else f2_name
    loser = f2_name if ensemble_result["ensemble_prediction"] == 1 else f1_name
    confidence = ensemble_result["ensemble_confidence"]
    prob_f1 = ensemble_result["ensemble_probability_f1"]

    lines.append(f"**Ensemble Prediction:** {winner} defeats {loser}")
    lines.append(f"**Confidence:** {confidence:.1%}")
    lines.append(
        f"**Win Probability:** {f1_name}: {prob_f1:.1%}, {f2_name}: {1-prob_f1:.1%}")

    # Model agreement
    if ensemble_result["models_agree"]:
        lines.append("**Model Agreement:** ✓ Both models agree")
    else:
        lines.append(
            f"**Model Agreement:** ⚠ Models disagree (used {ensemble_result['method']})")

    lines.append("")

    # Individual model details
    lines.append("**Individual Model Predictions:**")
    for pred in ensemble_result.get("individual_predictions", []):
        model_name = pred["model"]
        model_winner = f1_name if pred["prediction"] == 1 else f2_name
        prob = pred["probability_f1"]
        threshold = pred["threshold"]
        conf = pred["confidence"]
        margin = pred["margin"]

        direction = "above" if margin > 0 else "below"
        lines.append(
            f"- {model_name}: {model_winner} "
            f"(P({f1_name})={prob:.1%}, threshold={threshold:.3f}, "
            f"{abs(margin):.1%} {direction} threshold, "
            f"confidence={conf:.1%})"
        )

    return "\n".join(lines)


def format_similar_fights(similar_fights: List[Dict]) -> str:
    """Format similar fights for prompt."""
    if not similar_fights:
        return "No similar historical fights found."

    lines = ["Found similar historical matchups:"]
    for i, fight in enumerate(similar_fights[:5], 1):
        f1 = fight.get('f_1_name', 'Unknown')
        f2 = fight.get('f_2_name', 'Unknown')
        winner = fight.get('winner', 'Unknown')
        result = fight.get('result', 'Unknown')
        similarity = fight.get('similarity', 0)
        lines.append(
            f"{i}. {f1} vs {f2} - Winner: {winner} ({result}) [Similarity: {similarity:.2%}]")

    return "\n".join(lines)


def format_feature_summary(features: np.ndarray, feature_names: List[str]) -> str:
    """Format key features for prompt."""
    if feature_names is None or len(feature_names) == 0:
        return "Feature details unavailable."

    features = np.atleast_2d(features).flatten()

    # Key features to highlight (indices may vary based on your feature order)
    key_features = [
        'height_diff', 'reach_diff', 'age_diff',
        'win_rate_diff', 'exp_diff',
        'slpm_diff', 'str_acc_diff', 'str_def_diff',
        'td_avg_diff', 'td_def_diff', 'sub_avg_diff',
        'streak_diff', 'ko_power_diff', 'effectiveness_diff'
    ]

    lines = ["Key stat differentials (positive = Fighter 1 advantage):"]
    for fname in key_features:
        if fname in feature_names:
            idx = feature_names.index(fname)
            if idx < len(features):
                val = features[idx]
                lines.append(f"- {fname}: {val:+.2f}")

    return "\n".join(lines)


# =============================================================================
# Main FightAnalyzer Class
# =============================================================================

class FightAnalyzer:
    """
    Main class for UFC fight analysis combining ML models and LLM.

    Usage:
        analyzer = FightAnalyzer(provider="ollama")
        analyzer.load_models(with_odds=False)

        # Quick prediction (no LLM)
        result = analyzer.quick_predict(features, "Fighter A", "Fighter B")

        # Full analysis with LLM
        analysis = analyzer.analyze_from_features(features, "Fighter A", "Fighter B")
    """

    def __init__(
        self,
        provider: Literal["ollama", "claude", "openai"] = "ollama",
        model: Optional[str] = None,
        config: Optional[AnalyzerConfig] = None
    ):
        # Set up config
        if config:
            self.config = config
        else:
            self.config = AnalyzerConfig()

        self.config.provider = provider

        # Set default model based on provider
        if model:
            self.config.model = model
        elif provider == "ollama":
            self.config.model = "llama3.1:8b"
        elif provider == "claude":
            self.config.model = "claude-sonnet-4-20250514"
        elif provider == "openai":
            self.config.model = "gpt-4o"

        # Initialize components
        self.llm = create_provider(self.config)
        self.models = ModelEnsemble(self.config)
        self.similarity = SimilaritySearchWrapper(self.config)
        self.feature_names = None

        # Print config
        print(f"\n{'='*60}")
        print("FightPredict RAG Analyzer")
        print(f"{'='*60}")
        print(f"Provider: {self.config.provider}")
        print(f"Model: {self.config.model}")
        print(f"{'='*60}\n")

    def load_models(self, with_odds: bool = True) -> bool:
        """Load all models and data."""
        print("Loading models...")

        success = True

        # Load ML models
        if not self.models.load_xgboost(with_odds):
            success = False
        if not self.models.load_mlp(with_odds):
            warnings.warn("MLP model not loaded, will use XGBoost only")

        # Load FAISS
        if not self.similarity.load():
            warnings.warn(
                "FAISS index not loaded, similar fights will not be available")

        # Load feature names
        try:
            features_path = (
                self.config.features_path if with_odds
                else self.config.features_path.replace("_with_odds", "")
            )
            df = pd.read_csv(features_path)
            self.feature_names = [c for c in df.columns if c != 'target']
            print(f"✓ Loaded {len(self.feature_names)} feature names")
        except Exception as e:
            print(f"✗ Could not load feature names: {e}")

        # Check LLM availability
        if self.llm.is_available():
            print(f"✓ LLM provider ({self.config.provider}) is available")
        else:
            print(f"✗ LLM provider ({self.config.provider}) is NOT available")
            if self.config.provider == "ollama":
                print("  Make sure Ollama is running: ollama serve")
            else:
                print(
                    f"  Make sure {self.config.provider.upper()}_API_KEY is set")

        print()
        return success

    def analyze_from_features(
        self,
        features: np.ndarray,
        fighter1_name: str = "Fighter 1",
        fighter2_name: str = "Fighter 2",
        verbose: bool = True
    ) -> str:
        """
        Analyze a matchup given pre-computed features.

        Args:
            features: Feature vector (101 or 110 dimensions)
            fighter1_name: Name of fighter 1 (red corner)
            fighter2_name: Name of fighter 2 (blue corner)
            verbose: Print progress

        Returns:
            Natural language analysis string
        """
        features = np.atleast_2d(features).astype(np.float32)

        if verbose:
            print(f"Analyzing: {fighter1_name} vs {fighter2_name}")
            print("-" * 40)

        # Get model predictions
        if verbose:
            print("Getting model predictions...")
        ensemble_result = self.models.predict_ensemble(features)

        # Get similar fights
        if verbose:
            print("Finding similar historical fights...")
        similar_fights = self.similarity.find_similar_fights(
            features,
            k=self.config.similar_fights_k
        )

        # Format prompt components
        model_pred_text = format_model_predictions(
            ensemble_result, fighter1_name, fighter2_name
        )
        similar_fights_text = format_similar_fights(similar_fights)
        feature_summary_text = format_feature_summary(
            features, self.feature_names)

        # Build the prompt
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            fighter1_name=fighter1_name,
            fighter2_name=fighter2_name,
            model_predictions=model_pred_text,
            similar_fights=similar_fights_text,
            feature_summary=feature_summary_text
        )

        if verbose:
            print("Generating analysis with LLM...")

        # Generate analysis
        try:
            analysis = self.llm.generate(prompt, SYSTEM_PROMPT)
        except Exception as e:
            analysis = f"Error generating analysis: {e}\n\n**Raw Data:**\n\n{model_pred_text}\n\n{similar_fights_text}"

        if verbose:
            print("Done!\n")

        return analysis

    def quick_predict(
        self,
        features: np.ndarray,
        fighter1_name: str = "Fighter 1",
        fighter2_name: str = "Fighter 2"
    ) -> Dict[str, Any]:
        """
        Get a quick prediction without LLM analysis.

        Returns dict with predictions from all available sources.
        """
        features = np.atleast_2d(features).astype(np.float32)

        result = {
            "fighter1": fighter1_name,
            "fighter2": fighter2_name,
        }

        # Model predictions
        ensemble = self.models.predict_ensemble(features)
        if ensemble:
            result["ensemble"] = ensemble
            winner = fighter1_name if ensemble["ensemble_prediction"] == 1 else fighter2_name
            result["predicted_winner"] = winner
            result["confidence"] = ensemble["ensemble_confidence"]
            result["win_probability"] = {
                fighter1_name: ensemble["ensemble_probability_f1"],
                fighter2_name: 1 - ensemble["ensemble_probability_f1"]
            }

        # Similar fights analysis
        similar_analysis = self.similarity.analyze_similar_fights(
            features,
            k=self.config.similar_fights_k
        )
        if similar_analysis:
            result["similar_fights_analysis"] = similar_analysis

        return result

    def search_fighters(self, query: str) -> List[str]:
        """Search for fighters by name."""
        return self.similarity.search_fighter(query)

    def get_fighter_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """Get fighter statistics."""
        return self.similarity.get_fighter_info(name)

    def switch_provider(
        self,
        provider: Literal["ollama", "claude", "openai"],
        model: Optional[str] = None
    ):
        """Switch LLM provider on the fly."""
        self.config.provider = provider
        if model:
            self.config.model = model
        elif provider == "ollama":
            self.config.model = "llama3.1:8b"
        elif provider == "claude":
            self.config.model = "claude-sonnet-4-20250514"
        elif provider == "openai":
            self.config.model = "gpt-4o"

        self.llm = create_provider(self.config)
        print(f"Switched to {provider} ({self.config.model})")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="FightPredict RAG Analyzer")
    parser.add_argument(
        "--provider",
        choices=["ollama", "claude", "openai"],
        default="ollama",
        help="LLM provider to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (defaults based on provider)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a test analysis with sample data"
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Search for a fighter by name"
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = FightAnalyzer(provider=args.provider, model=args.model)
    analyzer.load_models()

    if args.search:
        print(f"\nSearching for '{args.search}'...")
        results = analyzer.search_fighters(args.search)
        if results:
            print("Found fighters:")
            for name in results:
                print(f"  - {name}")
        else:
            print("No fighters found.")
        return

    if args.test:
        print("\nRunning test analysis with random features...")
        # Generate random test features
        np.random.seed(42)
        test_features = np.random.randn(1, 101).astype(np.float32)

        # Quick predict
        result = analyzer.quick_predict(
            test_features,
            fighter1_name="Test Fighter A",
            fighter2_name="Test Fighter B"
        )
        print("\nQuick Prediction Result:")
        print(json.dumps(result, indent=2, default=str))

        # Full analysis
        print("\n" + "="*60)
        print("FULL LLM ANALYSIS")
        print("="*60 + "\n")

        analysis = analyzer.analyze_from_features(
            test_features,
            fighter1_name="Test Fighter A",
            fighter2_name="Test Fighter B"
        )
        print(analysis)


if __name__ == "__main__":
    main()
