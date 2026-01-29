# 🥊 FightPredict

**UFC Fight Prediction System powered by Machine Learning + RAG-Enhanced AI Analysis**

---

## 📊 Project Stats

| Metric                 | Value                   |
| ---------------------- | ----------------------- |
| Historical Fights      | 8,230                   |
| Unique Fighters        | 2,615                   |
| Engineered Features    | 101 (+ 9 odds features) |
| XGBoost Accuracy       | 72.2% (73.2% with odds) |
| MLP Accuracy           | 71.3% (71.3% with odds) |
| Baseline (F1 win rate) | 64.6%                   |
| **Improvement**        | **+8.6% over baseline** |

---

## 🎯 Features

- **🔍 Fighter Search** - Autocomplete across 2,615 UFC fighters
- **📊 Side-by-Side Stats** - Striking, grappling, record comparison
- **🤖 ML Predictions** - XGBoost + MLP ensemble with threshold-optimized confidence
- **📜 Similar Fights** - FAISS-powered historical matchup retrieval (RAG)
- **🔬 SHAP Explainability** - Understand why the model predicts what it does
- **💬 AI Analysis** - LLM-powered natural language breakdowns (4 styles)
- **💰 Betting Support** - Enhanced predictions with odds input
- **🔧 MLP Diagnostics** - Debug probability distributions and feature outliers

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/7beyonder7/fights_predictor.git
cd FightPredict

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run src/app.py
```

Opens at `http://localhost:8501`

### 3. CLI Predictions

```bash
# Quick prediction (no LLM)
python src/predict_fight.py -f1 "Islam Makhachev" -f2 "Charles Oliveira" --quick
python src/predict_fight.py -f1 "Brandon Royval" -f2 "Manel Kape" --quick
python src/predict_fight.py -f1 "Alexander Volkanovski" -f2 "Benoit Saint Denis" --quick

# Full analysis with LLM
python src/predict_fight.py -f1 "Jon Jones" -f2 "Stipe Miocic" --title-fight

# With betting odds
python src/predict_fight.py -f1 "Alex Pereira" -f2 "Jamahal Hill" --f1-odds -180 --f2-odds 150

# Interactive mode
python src/predict_fight.py --interactive

# SHAP Explainer Directly
python src/shap_explainer.py # This generates SHAP analysis plots in the models/ folder.
```

---

## 📁 Project Structure

```
FightPredict/
├── data/
│   ├── raw/
│   │   └── UFC_full_data_silver.csv      # Raw UFC data
│   └── processed/
│       ├── fights_cleaned.csv            # 8,230 fights with metadata
│       ├── fighters.csv                  # 2,615 fighter snapshots
│       ├── features_model_ready.csv      # 101 features + target
│       └── features_model_ready_with_odds.csv  # 110 features
│
├── models/
│   ├── xgboost_model.pkl                 # XGBoost (72.2%)
│   ├── xgboost_model_with_odds.pkl       # XGBoost with odds (73.2%) ⭐
│   ├── mlp_model.pt                      # MLP (71.3%)
│   ├── mlp_model_with_odds.pt            # MLP with odds (71.3%)
│   ├── faiss_index.bin                   # FAISS vector index
│   ├── faiss_index_metadata.pkl          # FAISS metadata + scaler
│   ├── shap_analysis.json                # SHAP feature importance
│   └── training_results.json             # Training metrics
│
├── src/
│   ├── app.py                            # 🌟 Streamlit UI (main)
│   ├── data_cleaning.py                  # Data preprocessing
│   ├── feature_engineering.py            # 101 feature pipeline
│   ├── fighters.py                       # Unified fighter database
│   ├── train_xgboost.py                  # XGBoost training (Optuna)
│   ├── train_mlp.py                      # MLP training (ImprovedMLPClassifier)
│   ├── test_cuda_available.py            # Test cuda availability
│   ├── generate_matchup_features.py      # Feature generation for inference
│   ├── rag_analyzer.py                   # Ensemble + LLM analysis
│   ├── faiss_search.py                   # Similarity search (RAG retrieval)
│   ├── shap_explainer.py                 # SHAP explainability
│   ├── mlp_diagnostic.py                 # MLP debugging tool
│   ├── predict_fight.py                  # CLI interface
│   └── prompts.py                        # LLM prompt templates
│
├── requirements.txt
└── README.md
```

---

## 🎨 Streamlit UI

### 5-Tab Interface

| Tab                   | Description                                          |
| --------------------- | ---------------------------------------------------- |
| 🏆 **Prediction**     | Winner, confidence, probability bar, model breakdown |
| 📊 **Stats**          | Side-by-side fighter comparison                      |
| 🔍 **Why?**           | SHAP explainability - factors driving prediction     |
| 📜 **Similar Fights** | Historical matchups via FAISS (RAG retrieval)        |
| 🤖 **AI Analysis**    | LLM breakdown with 4 styles                          |

### AI Analysis Styles

| Style         | Use Case       | Output                            |
| ------------- | -------------- | --------------------------------- |
| **Quick**     | Fast answer    | ~100 words, winner + 3 reasons    |
| **Detailed**  | Full breakdown | Paths to victory, upset potential |
| **Betting**   | Finding value  | Model vs odds, unit sizing        |
| **Technical** | MMA deep dive  | Striking/grappling analysis       |

---

## 🧠 Model Details

### XGBoost (Primary Model)

- **Accuracy**: 72.2% (73.2% with odds)
- **AUC-ROC**: 0.8135 (0.8247 with odds)
- Optuna Bayesian optimization (100 trials)
- Threshold stability verified across 5-fold CV
- SHAP explainability integrated

**Optimized Thresholds:**
| Model | Threshold | Stability |
|-------|-----------|-----------|
| No Odds | 0.582 | ✅ std=0.045 |
| With Odds | 0.466 | ✅ std=0.019 |

### MLP (ImprovedMLPClassifier)

- **Accuracy**: 71.3% (71.3% with odds)
- **AUC-ROC**: 0.7885 (0.7943 with odds)
- Architecture: `[256, 128, 64]` (monotonically decreasing)
- **LeakyReLU activation** (prevents dead neurons)
- Dropout: 0.25, Weight decay: 0.01

**Key Fix**: Capped `days_since_fight` features at 1000 days to prevent extreme values from causing probability clustering around 0.5.

### Ensemble Logic

```python
# Average raw probabilities from both models
# confidence-weighted:
xgb_weight = xgb_confidence / (xgb_confidence + mlp_confidence)
mlp_weight = mlp_confidence / (xgb_confidence + mlp_confidence)
ensemble_prob = xgb_prob * xgb_weight + mlp_prob * mlp_weight

# Apply ensemble threshold
prediction = "F1" if ensemble_prob > threshold else "F2"

# Confidence = distance from decision boundary
confidence = abs(ensemble_prob - 0.5) * 2  # Scaled to [0, 1]

# Reduce confidence if models disagree
if xgb_pred != mlp_pred:
    confidence *= 0.8
```

---

## 📈 Feature Categories (101 Features)

| Category      | Count | Examples                                          |
| ------------- | ----- | ------------------------------------------------- |
| Physical      | 6     | height_diff, reach_diff, age_diff, ape_index_diff |
| Career/Record | 8     | win_rate_diff, exp_diff, record_score_diff        |
| Striking      | 12    | slpm_diff, str_acc_diff, str_def_diff             |
| Grappling     | 10    | td_avg_diff, td_def_diff, sub_avg_diff            |
| Momentum      | 8     | streak_diff, days_since_fight\*, activity_diff    |
| Style         | 9     | striker_score, grappler_score, style_ratio        |
| Finish Rates  | 12    | ko_power_diff, sub_threat_diff, durability_diff   |
| Experience    | 9     | ufc_exp_diff, is_debut, is_veteran                |
| Matchup       | 7     | stance_clash, td_battle, striking_battle          |
| Categorical   | 4     | stance_encoded, weight_class_encoded, title_fight |
| **Odds**      | 9     | f_1_odds, f_2_odds, odds_diff, implied_prob_norm  |

\*`days_since_fight` capped at 1000 days (~2.7 years) to prevent extreme values.

---

## 🔍 RAG System (Retrieval-Augmented Generation)

FightPredict uses a **RAG architecture** for enhanced predictions. RAG combines:

1. **Retrieval** — Finding relevant information from a database
2. **Augmented** — Adding that information to a prompt
3. **Generation** — LLM generates a response using that context

### Complete RAG Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INPUT                                   │
│              "Makhachev vs Tsarukyan"                           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. MODEL PREDICTIONS (Ensemble)                                │
│     ├── XGBoost → 72% Makhachev wins                            │
│     └── MLP → 68% Makhachev wins                                │
│     = Ensemble: 70% Makhachev                                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. RETRIEVAL (FAISS) ← This is the "R" in RAG                  │
│     Finds 10 similar historical fights:                         │
│     - Khabib vs Gaethje (89% similar)                           │
│     - Makhachev vs Green (85% similar)                          │
│     - etc.                                                      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. AUGMENTED PROMPT (combines everything)                      │
│     "Here are model predictions: ...                            │
│      Here are similar fights: ...                               │
│      Here are stat differentials: ...                           │
│      Analyze this matchup."                                     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. GENERATION (LLM)                                            │
│     Ollama / Claude / OpenAI generates natural language         │
│     analysis based on all the context                           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: "Makhachev is predicted to win due to his              │
│           superior grappling, evidenced by similar              │
│           fights against Gaethje-style opponents..."            │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components in `rag_analyzer.py`

| Component                 | What it does                                    |
| ------------------------- | ----------------------------------------------- |
| `ModelEnsemble`           | Loads XGBoost + MLP, combines their predictions |
| `SimilaritySearchWrapper` | FAISS search for similar historical fights      |
| `LLMProvider` (abstract)  | Interface for Ollama/Claude/OpenAI              |
| `FightAnalyzer`           | Main class that orchestrates everything         |

### The Retrieval Step (FAISS)

```python
# This is the core RETRIEVAL step
similar_fights = self.similarity.find_similar_fights(features, k=10)
```

FAISS (Facebook AI Similarity Search) finds the 10 most similar historical fights based on 101-dimensional feature vectors. This gives the LLM **real historical context** to reason about, rather than hallucinating.

### Two Prediction Modes

| Mode      | Command   | What happens                            |
| --------- | --------- | --------------------------------------- |
| **Quick** | `--quick` | Models only, no LLM, instant results    |
| **Full**  | (default) | Models + FAISS retrieval + LLM analysis |

### Why RAG Matters

**Without RAG**: LLM makes up analysis based on general knowledge  
**With RAG**: LLM grounds analysis in actual historical fight data

**Example**: For Volkanovski vs Saint Denis, FAISS might find that in 10 similar historical matchups (experienced striker vs younger grappler), the experienced fighter won 8/10 times — providing valuable context that the LLM incorporates into its analysis.

---

## 🤖 LLM Setup

### Environment Variables (.env file)

Create a `.env` file in the project root to store your API keys:

```bash
# Create .env file
touch .env  # or create manually on Windows
```

Add your API keys to `.env`:

```env
# .env file
ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENAI_API_KEY=sk-your-key-here
```

### Ollama (Local - Free) - Recommended

```bash
# Install from https://ollama.ai
ollama pull llama3.1:8b
ollama serve
```

No API key needed - runs locally.

### Claude API

Add to `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Or set directly in terminal:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."  # Linux/Mac
$env:ANTHROPIC_API_KEY="sk-ant-..."    # PowerShell
```

### OpenAI API

Add to `.env`:

```env
OPENAI_API_KEY=sk-your-key-here
```

Or set directly in terminal:

```bash
export OPENAI_API_KEY="sk-..."  # Linux/Mac
$env:OPENAI_API_KEY="sk-..."    # PowerShell
```

---

## 🔧 Training Pipeline

### Full Retrain (after data/feature changes)

```bash
# 1. Clean raw data
python src/data_cleaning.py

# 2. Generate features (101 base + 9 odds)
python src/feature_engineering.py

# 3. Train XGBoost (Optuna + threshold tuning + SHAP)
python src/train_xgboost.py --trials 100

# 4. Train MLP
python src/train_mlp.py --quick              # Default architecture (recommended)
python src/train_mlp.py --with-odds --quick  # With odds features
python src/train_mlp.py --trials 50          # Optuna tuning

# 5. Build FAISS index
python src/faiss_search.py
```

### MLP Diagnostic Tool

Debug MLP probability distributions:

```bash
# Basic diagnostic (analyze training data)
python src/mlp_diagnostic.py --plot

# Analyze specific matchup
python src/mlp_diagnostic.py -f1 "Islam Makhachev" -f2 "Charles Oliveira" --plot

# Test model with odds
python src/mlp_diagnostic.py --with-odds --plot
```

**What it checks:**

- Probability distribution spread (should be > 0.1 std)
- Class separation (F1 vs F2 mean difference)
- Feature outliers (z-score > 3)
- Dead neuron rates
- Activation saturation

---

## 💻 Python API

```python
from generate_matchup_features import MatchupFeatureGenerator
from rag_analyzer import FightAnalyzer

# Initialize
gen = MatchupFeatureGenerator()
gen.load_data()

analyzer = FightAnalyzer(provider="ollama")
analyzer.load_models(with_odds=False)

# Generate features
features, names = gen.generate_features("Islam Makhachev", "Charles Oliveira")

# Get quick prediction (no LLM)
result = analyzer.quick_predict(features, "Islam Makhachev", "Charles Oliveira")
print(f"Winner: {result['predicted_winner']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"P(Fighter 1): {result['ensemble_prob']:.1%}")

# Get full analysis with LLM
analysis = analyzer.analyze_from_features(
    features=features,
    fighter1="Islam Makhachev",
    fighter2="Charles Oliveira",
    style="detailed"
)
print(analysis['llm_analysis'])

# Get SHAP explanation
from shap_explainer import SHAPExplainer
explainer = SHAPExplainer()
explanation = explainer.explain_single_prediction(features, names)
for factor in explanation['top_factors']:
    print(f"{factor['feature']}: {factor['impact']}")
```

---

## 📊 Example Predictions

```
$ python src/predict_fight.py -f1 "Brandon Royval" -f2 "Manel Kape" --quick

🏆 PREDICTED WINNER: Manel Kape
   Confidence: 41.4%
   Win Probabilities:
   • Brandon Royval: 29.3%
   • Manel Kape: 70.7%

   Model      Prediction    P(F1)    Threshold  Margin
   --------------------------------------------------------
   XGBoost    Manel Kape    32.8%    0.582      -25.4%
   MLP        Manel Kape    25.8%    0.440      -18.2%
   ✓ Models AGREE on prediction
```

```
$ python src/predict_fight.py -f1 "Alexander Volkanovski" -f2 "Benoit Saint Denis" --quick

🏆 PREDICTED WINNER: Alexander Volkanovski
   Confidence: 34.2%
   Win Probabilities:
   • Alexander Volkanovski: 67.1%
   • Benoit Saint Denis: 32.9%

   Model      Prediction              P(F1)    Threshold  Margin
   ----------------------------------------------------------------
   XGBoost    Alexander Volkanovski   73.0%    0.582      +14.8%
   MLP        Alexander Volkanovski   61.2%    0.440      +17.2%
   ✓ Models AGREE on prediction
```

---

## ⚠️ Known Limitations

1. **Historical data only** - Can't account for injuries, weight cuts, training camps, or fight-week factors
2. **Class imbalance** - Dataset is 64.6% Fighter 1 wins (handled via threshold tuning)
3. **Feature lag** - Stats reflect most recent fight, not current form
4. **Weight class moves** - Model may not fully capture division changes
5. **LLM dependency** - Full analysis requires Ollama/Claude/OpenAI running

---

## 🐛 Troubleshooting

### MLP outputs ~50% for everything (coin flip)

Run the diagnostic to check probability distribution:

```bash
python src/mlp_diagnostic.py --plot
```

If std < 0.1, check for extreme feature values. The `days_since_fight` capping fix should prevent this.

### FAISS index not found

Rebuild the index:

```bash
python src/faiss_search.py
```

### LLM not responding

Check Ollama is running:

```bash
ollama serve
curl http://localhost:11434/api/tags
```

---

## 📦 Requirements

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
torch>=2.0.0
faiss-cpu>=1.7.4
shap>=0.42.0
optuna>=3.0.0
streamlit>=1.28.0
plotly>=5.18.0
requests>=2.28.0
matplotlib>=3.7.0
anthropic>=0.18.0
openai>=1.0.0
```
