"""
FightPredict - Streamlit UI (Enhanced Version)
===============================================
A polished web interface for UFC fight predictions with demo mode support.

Usage:
    streamlit run src/app.py

Features:
    - Fighter search with autocomplete
    - Side-by-side stat comparison  
    - ML model predictions (XGBoost + MLP ensemble)
    - Similar historical fights via FAISS
    - SHAP explainability (why the model predicts what it does)
    - Refined LLM-powered analysis with multiple styles
    - Demo mode when data files are unavailable
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Optional imports for SHAP visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import unified fighter database (handle both direct and package imports)
try:
    from fighters import FighterDatabase, FighterSnapshot
except ImportError:
    from src.fighters import FighterDatabase, FighterSnapshot

# =============================================================================
# Page Configuration (MUST be first)
# =============================================================================

st.set_page_config(
    page_title="FightPredict | UFC Prediction System",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/FightPredict',
        'Report a bug': 'https://github.com/yourusername/FightPredict/issues',
        'About': "# FightPredict\nUFC Fight Prediction System using ML & AI"
    }
)

# =============================================================================
# Custom Styling
# =============================================================================


def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap');
    
    :root {
        --red-corner: #DC2626;
        --blue-corner: #2563EB;
        --gold-accent: #F59E0B;
        --dark-bg: #0A0A0A;
        --card-bg: #141414;
        --border: #262626;
        --text-primary: #FAFAFA;
        --text-muted: #A1A1AA;
        --success: #22C55E;
        --warning: #EAB308;
    }
    
    /* Global */
    .stApp { background-color: var(--dark-bg); color: var(--text-primary) }
    
    /* Hide defaults */
    #MainMenu, footer { visibility: hidden; }
    
    /* Header */
    .hero-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: clamp(2.5rem, 8vw, 4.5rem);
        letter-spacing: 6px;
        text-align: center;
        background: linear-gradient(135deg, var(--red-corner) 0%, var(--gold-accent) 50%, var(--blue-corner) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
        line-height: 1.1;
    }
    
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: var(--text-muted);
        text-align: center;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }
    
    /* Cards */
    .stat-card {
        background: linear-gradient(145deg, var(--card-bg), #0D0D0D);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .red-accent { border-left: 4px solid var(--red-corner); }
    .blue-accent { border-left: 4px solid var(--blue-corner); }
    .gold-accent { border: 2px solid var(--gold-accent); box-shadow: 0 0 20px rgba(245, 158, 11, 0.15); }
    
    /* Fighter names */
    .fighter-name {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.8rem;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    .red-text { color: var(--red-corner); }
    .blue-text { color: var(--blue-corner); }
    .gold-text { color: var(--gold-accent); }
    
    /* Winner banner */
    .winner-banner {
        text-align: center;
        padding: 2rem;
    }
    .winner-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 0.5rem;
    }
    .winner-name {
        font-family: 'Bebas Neue', sans-serif;
        font-size: clamp(2rem, 6vw, 3.5rem);
        letter-spacing: 3px;
        color: var(--gold-accent);
    }
    .confidence-badge {
        display: inline-block;
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid var(--gold-accent);
        border-radius: 999px;
        padding: 0.5rem 1.5rem;
        margin-top: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        color: var(--gold-accent);
    }
    
    /* Probability bar */
    .prob-container {
        display: flex;
        height: 48px;
        border-radius: 24px;
        overflow: hidden;
        margin: 1.5rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }
    .prob-red {
        background: linear-gradient(90deg, var(--red-corner), #EF4444);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .prob-blue {
        background: linear-gradient(90deg, #3B82F6, var(--blue-corner));
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Stats table */
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid var(--border);
    }
    .stat-row:last-child { border-bottom: none; }
    .stat-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
    }
    .stat-better { color: var(--success); font-weight: 600; }
    .stat-neutral { color: var(--text-primary); }
    
    /* Model breakdown */
    .model-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.25rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
    }
    
    /* Similar fights */
    .fight-card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        border-left: 3px solid var(--gold-accent);
    }
    .fight-matchup {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--text-primary);
    }
    .fight-meta {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }
    .fight-result {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
    }
    .result-f1 { background: rgba(34, 197, 94, 0.15); color: var(--success); }
    .result-f2 { background: rgba(239, 68, 68, 0.15); color: #EF4444; }
    
    /* Button override */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, var(--red-corner) 0%, #B91C1C 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.4rem;
        letter-spacing: 3px;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(220, 38, 38, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--card-bg);
        padding: 4px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: var(--red-corner) !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(37, 99, 235, 0.1);
        border-left: 4px solid var(--blue-corner);
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .warning-box {
        background: rgba(245, 158, 11, 0.1);
        border-left: 4px solid var(--gold-accent);
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* VS badge */
    .vs-badge {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2rem;
        color: var(--text-muted);
        text-align: center;
        line-height: 1;
    }
    </style>
    """, unsafe_allow_html=True)


load_custom_css()

# =============================================================================
# Data Loading with Caching
# =============================================================================


@st.cache_resource(show_spinner=False)
def load_shared_components():
    """
    Load shared components that don't depend on odds selection.
    These are loaded ONCE at app start.
    """
    components = {
        'generator': None,
        'fighter_db': None,
        'fighter_list': [],
        'status': 'unknown'
    }

    # Load unified fighter database first
    try:
        fighter_db = FighterDatabase()
        if fighter_db.load():
            components['fighter_db'] = fighter_db
            components['fighter_list'] = fighter_db.get_all_names()
            components['status'] = 'loaded'
    except Exception as e:
        components['status'] = f'error loading fighters: {e}'

    # Load feature generator (uses same fighter database internally)
    try:
        from generate_matchup_features import MatchupFeatureGenerator
        gen = MatchupFeatureGenerator()
        if gen.load_data():
            components['generator'] = gen
    except Exception as e:
        if components['status'] == 'unknown':
            components['status'] = f'error: {e}'

    return components


@st.cache_resource(show_spinner=False)
def load_analyzer(with_odds: bool):
    """
    Load FightAnalyzer with appropriate model (with/without odds).
    Cached per with_odds value, so switching is fast after first load.
    """
    try:
        from rag_analyzer import FightAnalyzer
        analyzer = FightAnalyzer(provider="ollama")
        analyzer.load_models(with_odds=with_odds)
        return analyzer
    except Exception as e:
        st.warning(f"Could not load analyzer: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_shap_explainer(with_odds: bool):
    """
    Load SHAP explainer for the appropriate model.
    Cached per with_odds value.
    """
    try:
        import joblib
        from shap_explainer import SHAPExplainer

        model_filename = "xgboost_model_with_odds.pkl" if with_odds else "xgboost_model.pkl"
        model_path = Path(f"models/{model_filename}")

        if model_path.exists():
            model_data = joblib.load(model_path)
            xgb_model = model_data['model']
            return {
                'xgb_model': xgb_model,
                'shap_explainer': SHAPExplainer(xgb_model, output_path="models")
            }
    except Exception as e:
        pass  # SHAP is optional

    return {'xgb_model': None, 'shap_explainer': None}


@st.cache_data(show_spinner=False)
def get_shap_explanation(_explainer, features: np.ndarray, feature_names: List[str],
                         f1_name: str, f2_name: str) -> Optional[Dict]:
    """Get SHAP explanation for a prediction (cached)."""
    if _explainer is None:
        return None

    try:
        _explainer.feature_names = feature_names
        explanation = _explainer.explain_single_prediction(
            features, fighter_1_name=f1_name, fighter_2_name=f2_name
        )
        return explanation
    except Exception as e:
        return None

# =============================================================================
# Helper Functions
# =============================================================================


def format_record(w: int, l: int, d: int) -> str:
    return f"{w}-{l}-{d}"


def render_stat_comparison(label: str, f1_val, f2_val, higher_better: bool = True,
                           is_pct: bool = False, reverse_display: bool = False):
    """Render a stat comparison row."""
    # Format values
    if is_pct:
        f1_str = f"{f1_val:.1%}"
        f2_str = f"{f2_val:.1%}"
    elif isinstance(f1_val, float):
        f1_str = f"{f1_val:.2f}"
        f2_str = f"{f2_val:.2f}"
    else:
        f1_str = str(f1_val)
        f2_str = str(f2_val)

    # Determine who's better
    if higher_better:
        f1_better = f1_val > f2_val
    else:
        f1_better = f1_val < f2_val

    f1_class = "stat-better" if f1_better else "stat-neutral"
    f2_class = "stat-better" if not f1_better else "stat-neutral"

    if reverse_display:
        st.markdown(f"""
        <div class="stat-row">
            <span class="{f2_class} stat-value">{f2_str}</span>
            <span class="stat-label">{label}</span>
            <span class="{f1_class} stat-value">{f1_str}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="stat-row">
            <span class="{f1_class} stat-value">{f1_str}</span>
            <span class="stat-label">{label}</span>
            <span class="{f2_class} stat-value">{f2_str}</span>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# Main Application
# =============================================================================


def main():
    # Header
    st.markdown('<h1 class="hero-title">🥊 FIGHTPREDICT</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Machine Learning Powered UFC Predictions</p>',
                unsafe_allow_html=True)
    st.markdown("")

    # Load shared components (fighter database, feature generator)
    with st.spinner("🔄 Loading fight database..."):
        shared = load_shared_components()

    generator = shared['generator']
    fighter_db = shared['fighter_db']
    fighter_list = shared['fighter_list']

    # Check if data is available
    if fighter_db is None and generator is None:
        st.error(
            "⚠️ **Data files not found.** Please ensure the following files exist:")
        st.code("""
data/processed/fights_cleaned.csv
data/processed/features_model_ready.csv
        """)
        st.info(
            "Run the data pipeline first: `python src/data_cleaning.py && python src/feature_engineering.py`")
        return

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        llm_provider = st.selectbox(
            "LLM Provider",
            ["ollama", "claude", "openai"],
            help="Select provider for AI analysis"
        )

        quick_mode = st.checkbox("⚡ Quick Mode", value=True,
                                 help="Skip LLM for faster predictions")

        st.markdown("---")
        st.markdown("### 📊 System Stats")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fighters", f"{len(fighter_list):,}")
        with col2:
            fights_count = len(
                fighter_db.fights_df) if fighter_db and fighter_db.fights_df is not None else 0
            st.metric("Fights", f"{fights_count:,}")

        st.markdown("---")
        st.markdown("### 🎯 Model Accuracy")
        st.progress(0.734, text="XGBoost: 73.4%")
        st.progress(0.722, text="MLP v2: 72.2%")

        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.75rem; color: #71717A;">
        Built with XGBoost, PyTorch, FAISS, and Streamlit.<br>
        Data from UFCStats.com
        </div>
        """, unsafe_allow_html=True)

    # Fighter Selection
    st.markdown("## 👊 Select Fighters")

    col1, col_vs, col2 = st.columns([5, 1, 5])

    with col1:
        st.markdown("#### 🔴 Red Corner")
        fighter1 = st.selectbox(
            "Fighter 1",
            options=[""] + fighter_list,
            key="f1_select",
            label_visibility="collapsed",
            placeholder="Select or search..."
        )

    with col_vs:
        st.markdown(
            '<div class="vs-badge" style="margin-top: 2rem;">VS</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### 🔵 Blue Corner")
        fighter2 = st.selectbox(
            "Fighter 2",
            options=[""] + fighter_list,
            key="f2_select",
            label_visibility="collapsed",
            placeholder="Select or search..."
        )

    # Fight options
    with st.expander("🏆 Fight Details (Optional)", expanded=False):
        opt_col1, opt_col2, opt_col3 = st.columns(3)

        with opt_col1:
            title_fight = st.checkbox("Championship Bout", value=False)

        with opt_col2:
            weight_class = st.selectbox(
                "Weight Class",
                ["Auto-detect", "Flyweight", "Bantamweight", "Featherweight",
                 "Lightweight", "Welterweight", "Middleweight",
                 "Light Heavyweight", "Heavyweight"]
            )

        with opt_col3:
            use_odds = st.checkbox("Add Betting Odds", value=False)

        if use_odds:
            odds_col1, odds_col2 = st.columns(2)
            with odds_col1:
                f1_odds = st.number_input(
                    "F1 Odds (American)", value=-150, step=10)
            with odds_col2:
                f2_odds = st.number_input(
                    "F2 Odds (American)", value=130, step=10)
        else:
            f1_odds = f2_odds = None

    st.markdown("")

    # Predict and Clear buttons
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1, 2, 1, 1])
    with col_btn2:
        predict_btn = st.button("🎯 PREDICT FIGHT", use_container_width=True)
    with col_btn4:
        if 'prediction_data' in st.session_state:
            if st.button("🔄 New", use_container_width=True, help="Clear and start new prediction"):
                del st.session_state['prediction_data']
                st.rerun()

    st.markdown("---")

    # Run prediction
    if predict_btn:
        if not fighter1 or not fighter2:
            st.warning("⚠️ Please select both fighters.")
            return

        if fighter1 == fighter2:
            st.warning("⚠️ Please select two different fighters.")
            return

        # Get fighter stats using unified database
        f1_stats = fighter_db.get_snapshot(
            fighter1) if fighter_db else generator.get_fighter_snapshot(fighter1)
        f2_stats = fighter_db.get_snapshot(
            fighter2) if fighter_db else generator.get_fighter_snapshot(fighter2)

        if f1_stats is None:
            st.error(f"❌ Fighter not found: {fighter1}")
            return
        if f2_stats is None:
            st.error(f"❌ Fighter not found: {fighter2}")
            return

        # Generate prediction
        with st.spinner("🔮 Calculating prediction..."):
            try:
                # Determine if using odds based on user input
                include_odds = (f1_odds is not None and f2_odds is not None)

                wc = weight_class if weight_class != "Auto-detect" else "Unknown"
                features, feature_names = generator.generate_features(
                    fighter1=f1_stats.name,
                    fighter2=f2_stats.name,
                    weight_class=wc,
                    title_fight=title_fight,
                    f1_odds=f1_odds,
                    f2_odds=f2_odds,
                    include_odds=include_odds
                )

                # Load analyzer with matching odds setting
                analyzer = load_analyzer(with_odds=include_odds)

                result = None
                if analyzer:
                    result = analyzer.quick_predict(
                        features, f1_stats.name, f2_stats.name)

                # Load SHAP explainer with matching odds setting
                shap_components = load_shap_explainer(with_odds=include_odds)

                # Generate SHAP explanation if available
                shap_explanation = None
                if shap_components.get('shap_explainer'):
                    # Feature count should match: 101 for no odds, 110 for with odds
                    expected_features = 110 if include_odds else 101
                    if len(features) == expected_features:
                        shap_explanation = get_shap_explanation(
                            shap_components['shap_explainer'],
                            features,
                            feature_names,
                            f1_stats.name,
                            f2_stats.name
                        )

                # Store in session state to persist across reruns
                st.session_state['prediction_data'] = {
                    'features': features,
                    'feature_names': feature_names,
                    'result': result,
                    'f1_stats': f1_stats,
                    'f2_stats': f2_stats,
                    'shap_explanation': shap_explanation,
                    'include_odds': include_odds  # Store for later use
                }

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                return

    # Display results if we have prediction data (either fresh or from session state)
    if 'prediction_data' in st.session_state:
        pred_data = st.session_state['prediction_data']
        features = pred_data['features']
        feature_names = pred_data['feature_names']
        result = pred_data['result']
        f1_stats = pred_data['f1_stats']
        f2_stats = pred_data['f2_stats']
        shap_explanation = pred_data['shap_explanation']
        include_odds = pred_data.get('include_odds', False)

        # Load analyzer with correct odds setting for tabs that need it
        analyzer = load_analyzer(with_odds=include_odds)

        # Display Results in Tabs
        tab_pred, tab_stats, tab_explain, tab_similar, tab_ai = st.tabs([
            "🏆 Prediction", "📊 Stats", "🔍 Why?", "📜 Similar Fights", "🤖 AI Analysis"
        ])

        # TAB 1: Prediction
        with tab_pred:
            if result and "predicted_winner" in result:
                winner = result["predicted_winner"]
                confidence = result["confidence"]

                win_prob = result.get("win_probability", {})
                f1_prob = win_prob.get(f1_stats.name, 0.5)
                f2_prob = win_prob.get(f2_stats.name, 0.5)

                ensemble = result.get("ensemble", {})
                models_agree = ensemble.get("models_agree", True)

                # Check if this is essentially a toss-up
                is_tossup = confidence < 0.15 or (abs(f1_prob - 0.5) < 0.05)

                if is_tossup:
                    # Toss-up display
                    st.markdown(f"""
                    <div class="stat-card" style="border: 2px solid #F59E0B; text-align: center; padding: 2rem;">
                        <div style="font-size: 0.9rem; color: #A1A1AA; text-transform: uppercase; letter-spacing: 2px;">Prediction</div>
                        <div style="font-family: 'Bebas Neue', sans-serif; font-size: 2.5rem; color: #F59E0B; margin: 0.5rem 0;">
                            ⚖️ TOO CLOSE TO CALL
                        </div>
                        <div style="color: #A1A1AA;">
                            Models show no clear favorite ({confidence:.0%} confidence)
                        </div>
                        <div style="margin-top: 1rem; padding: 0.75rem; background: rgba(245, 158, 11, 0.1); border-radius: 8px;">
                            <strong>Slight lean:</strong> {winner} — but this is essentially a coin flip
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Normal winner display
                    # Color confidence badge based on level
                    if confidence >= 0.6:
                        conf_color = "#22C55E"  # Green - high
                        conf_label = "HIGH"
                    elif confidence >= 0.35:
                        conf_color = "#F59E0B"  # Amber - medium
                        conf_label = "MEDIUM"
                    else:
                        conf_color = "#EF4444"  # Red - low
                        conf_label = "LOW"

                    st.markdown(f"""
                    <div class="stat-card gold-accent">
                        <div class="winner-banner">
                            <div class="winner-label">Predicted Winner</div>
                            <div class="winner-name">{winner}</div>
                            <div style="display: inline-block; background: rgba(245, 158, 11, 0.15); border: 1px solid {conf_color}; border-radius: 999px; padding: 0.5rem 1.5rem; margin-top: 1rem; font-family: 'JetBrains Mono', monospace; color: {conf_color};">
                                {conf_label} CONFIDENCE ({confidence:.0%})
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Probability bar
                st.markdown("#### Win Probability")
                st.markdown(f"""
                <div class="prob-container">
                    <div class="prob-red" style="width: {f1_prob*100}%">
                        {f1_stats.name.split()[-1]} {f1_prob:.0%}
                    </div>
                    <div class="prob-blue" style="width: {f2_prob*100}%">
                        {f2_stats.name.split()[-1]} {f2_prob:.0%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Model breakdown
                st.markdown("#### Model Breakdown")

                preds = ensemble.get("individual_predictions", [])

                if preds:
                    model_cols = st.columns(len(preds))
                    for i, pred in enumerate(preds):
                        with model_cols[i]:
                            model_winner = f1_stats.name if pred["prediction"] == 1 else f2_stats.name
                            agrees = "✅" if model_winner == winner else "⚠️"
                            st.markdown(f"""
                            <div class="stat-card">
                                <strong>{pred['model']}</strong><br>
                                <span style="color: var(--gold-accent);">{agrees} {model_winner.split()[-1]}</span><br>
                                <small style="color: var(--text-muted);">
                                    P(F1): {pred['probability_f1']:.1%}<br>
                                    Confidence: {pred['confidence']:.1%}
                                </small>
                            </div>
                            """, unsafe_allow_html=True)

                    if models_agree:
                        st.success("✅ Both models agree on the prediction")
                    else:
                        st.warning(
                            f"⚠️ Models disagree - using {ensemble.get('method', 'higher confidence')}")
            else:
                st.warning("Could not generate prediction. Check model files.")

        # TAB 2: Stats Comparison
        with tab_stats:
            st.markdown("### Fighter Comparison")

            # Fighter cards side by side
            card_col1, card_col2 = st.columns(2)

            with card_col1:
                st.markdown(f"""
                <div class="stat-card red-accent">
                    <div class="fighter-name red-text">{f1_stats.name}</div>
                    <div style="color: var(--text-muted); margin-bottom: 1rem;">
                        {format_record(f1_stats.wins, f1_stats.losses, f1_stats.draws)} • 
                        {f1_stats.stance} • Age {f1_stats.age:.0f}
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Height</span>
                        <span class="stat-value">{f1_stats.height_cm:.0f} cm</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Reach</span>
                        <span class="stat-value">{f1_stats.reach_cm:.0f} cm</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">UFC Fights</span>
                        <span class="stat-value">{f1_stats.ufc_fights}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Streak</span>
                        <span class="stat-value" style="color: {'var(--success)' if f1_stats.current_streak > 0 else 'var(--text-muted)'};">
                            {f1_stats.current_streak:+d}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with card_col2:
                st.markdown(f"""
                <div class="stat-card blue-accent">
                    <div class="fighter-name blue-text">{f2_stats.name}</div>
                    <div style="color: var(--text-muted); margin-bottom: 1rem;">
                        {format_record(f2_stats.wins, f2_stats.losses, f2_stats.draws)} • 
                        {f2_stats.stance} • Age {f2_stats.age:.0f}
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Height</span>
                        <span class="stat-value">{f2_stats.height_cm:.0f} cm</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Reach</span>
                        <span class="stat-value">{f2_stats.reach_cm:.0f} cm</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">UFC Fights</span>
                        <span class="stat-value">{f2_stats.ufc_fights}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Streak</span>
                        <span class="stat-value" style="color: {'var(--success)' if f2_stats.current_streak > 0 else 'var(--text-muted)'};">
                            {f2_stats.current_streak:+d}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Detailed comparison
            st.markdown("### 🥊 Striking")
            render_stat_comparison("SLpM", f1_stats.slpm,
                                   f2_stats.slpm, True, False)
            render_stat_comparison(
                "Strike Accuracy", f1_stats.str_acc, f2_stats.str_acc, True, True)
            render_stat_comparison(
                "Strikes Absorbed", f1_stats.sapm, f2_stats.sapm, False, False)
            render_stat_comparison(
                "Strike Defense", f1_stats.str_def, f2_stats.str_def, True, True)

            st.markdown("### 🤼 Grappling")
            render_stat_comparison(
                "TD Average", f1_stats.td_avg, f2_stats.td_avg, True, False)
            render_stat_comparison(
                "TD Accuracy", f1_stats.td_acc, f2_stats.td_acc, True, True)
            render_stat_comparison(
                "TD Defense", f1_stats.td_def, f2_stats.td_def, True, True)
            render_stat_comparison(
                "Sub Average", f1_stats.sub_avg, f2_stats.sub_avg, True, False)

        # TAB 3: SHAP Explainability (NEW)
        with tab_explain:
            st.markdown("### 🔍 Why This Prediction?")
            st.markdown(
                "SHAP (SHapley Additive exPlanations) shows which factors most influenced the prediction.")

            if shap_explanation:
                # Build custom summary using ensemble result (not raw SHAP summary)
                ensemble = result.get("ensemble", {})
                confidence = result.get("confidence", 0)
                winner = result.get("predicted_winner", "Unknown")

                # Determine if toss-up
                is_tossup = confidence < 0.15

                top_factors = shap_explanation.get('top_factors', [])[:7]

                # Build summary text
                if is_tossup:
                    summary_lines = [
                        f"Prediction: Too close to call (slight lean: {winner})", ""]
                else:
                    summary_lines = [
                        f"Prediction: {winner} ({confidence:.0%} confidence)", ""]

                summary_lines.append("Key factors driving prediction:")
                for i, factor in enumerate(top_factors[:5], 1):
                    # Fix feature name formatting
                    feat_name = factor['feature']
                    feat_name = feat_name.replace('_', ' ')
                    feat_name = feat_name.replace(
                        'str differenceerential', 'str differential')  # Fix typo
                    feat_name = feat_name.replace('diff', 'difference')

                    shap_val = factor['shap_value']
                    direction = "favoring F1" if shap_val > 0 else "favoring F2"
                    sign = "+" if shap_val > 0 else ""
                    summary_lines.append(
                        f"  {i}. {feat_name}: {sign}{shap_val:.3f} ({direction})")

                summary_text = "\n".join(summary_lines)
                st.code(summary_text, language=None)

                st.markdown("### Top Factors Driving Prediction")

                if top_factors and PLOTLY_AVAILABLE:
                    # Prepare data for chart - fix feature names
                    features_list = []
                    for f in top_factors:
                        name = f['feature'].replace('_', ' ').title()
                        name = name.replace(
                            'Str Differenceerential', 'Str Differential')  # Fix typo
                        features_list.append(name)

                    values = [f['shap_value'] for f in top_factors]
                    colors = ['#22C55E' if v >
                              0 else '#EF4444' for v in values]

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=values,
                        y=features_list,
                        orientation='h',
                        marker_color=colors,
                        text=[f"{v:+.3f}" for v in values],
                        textposition='outside'
                    ))

                    fig.update_layout(
                        title="Feature Contributions to Prediction",
                        xaxis_title="SHAP Value (→ favors F1, ← favors F2)",
                        yaxis_title="",
                        height=400,
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter", size=12),
                        yaxis=dict(autorange="reversed")
                    )

                    fig.add_vline(x=0, line_dash="dash", line_color="gray")

                    st.plotly_chart(fig, use_container_width=True)

                elif top_factors:
                    # Fallback: show as styled table
                    st.markdown("#### Key Factors")

                    for factor in top_factors[:7]:
                        feat_name = factor['feature'].replace('_', ' ').title()
                        shap_val = factor['shap_value']

                        if shap_val > 0:
                            icon = "🟢"
                            direction = "Favors F1"
                            color = "#22C55E"
                        else:
                            icon = "🔴"
                            direction = "Favors F2"
                            color = "#EF4444"

                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #333;">
                            <span>{icon} <strong>{feat_name}</strong></span>
                            <span style="color: {color}; font-family: monospace;">{shap_val:+.3f} ({direction})</span>
                        </div>
                        """, unsafe_allow_html=True)

                # Interpretation guide
                with st.expander("📖 How to Read SHAP Values"):
                    st.markdown("""
                    **SHAP Values Explained:**
                    
                    - **Positive values (green)**: Push prediction toward Fighter 1 winning
                    - **Negative values (red)**: Push prediction toward Fighter 2 winning
                    - **Magnitude**: Larger absolute values = stronger influence
                    
                    **Example interpretations:**
                    - `exp_diff: +0.15` → Fighter 1's experience advantage strongly favors them
                    - `f_2_win_rate: -0.08` → Fighter 2's win rate slightly favors them
                    
                    The model combines all these factors to produce the final prediction.
                    """)
            else:
                st.markdown("""
                <div class="warning-box">
                    <strong>SHAP Analysis Not Available</strong><br><br>
                    To enable SHAP explainability:
                    <ol>
                        <li>Ensure XGBoost model exists at <code>models/xgboost_model.pkl</code></li>
                        <li>Install SHAP: <code>pip install shap</code></li>
                        <li>Restart the Streamlit app</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)

        # TAB 4: Similar Fights
        with tab_similar:
            st.markdown("### 📜 Similar Historical Matchups")

            if analyzer and hasattr(analyzer, 'similarity') and analyzer.similarity:
                try:
                    similar = analyzer.similarity.find_similar_fights(
                        features, k=10)

                    if similar:
                        f1_wins = sum(
                            1 for f in similar if f.get('f1_won', False))

                        st.markdown(f"""
                        <div class="info-box">
                            In <strong>{len(similar)}</strong> similar historical matchups, the Fighter 1 profile 
                            won <strong>{f1_wins}</strong> times (<strong>{f1_wins/len(similar):.0%}</strong>).
                        </div>
                        """, unsafe_allow_html=True)

                        for fight in similar:
                            f1_won = fight.get('f1_won', False)
                            result_class = "result-f1" if f1_won else "result-f2"
                            result_text = "F1 Won" if f1_won else "F2 Won"

                            st.markdown(f"""
                            <div class="fight-card">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <div class="fight-matchup">
                                            {fight.get('f1_name', 'Fighter 1')} vs {fight.get('f2_name', 'Fighter 2')}
                                        </div>
                                        <div class="fight-meta">
                                            {fight.get('event_name', '')} • {fight.get('result', '')}
                                        </div>
                                    </div>
                                    <div class="fight-result {result_class}">{result_text}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No similar fights found.")

                except Exception as e:
                    st.warning(f"Could not load similar fights: {e}")
            else:
                st.info(
                    "FAISS index not available. Build it with: `python src/faiss_search.py --build`")

        # TAB 5: AI Analysis
        with tab_ai:
            st.markdown("### 🤖 AI-Powered Analysis")

            # Analysis style selector
            analysis_style = st.radio(
                "Analysis Style",
                ["Quick", "Detailed", "Betting", "Technical"],
                horizontal=True,
                help="Choose the type of analysis you want"
            )

            if quick_mode and analysis_style != "Quick":
                st.markdown("""
                <div class="warning-box">
                    <strong>Quick Mode Enabled</strong><br>
                    Disable Quick Mode in the sidebar for detailed LLM analysis.
                </div>
                """, unsafe_allow_html=True)

            elif analyzer and analyzer.llm.is_available():
                with st.spinner("🔮 Generating AI analysis..."):
                    try:
                        # Try to use refined prompts
                        try:
                            from prompts import PromptEngine, AnalysisStyle, format_fighter_context, format_shap_explanation

                            style_map = {
                                "Quick": AnalysisStyle.QUICK,
                                "Detailed": AnalysisStyle.DETAILED,
                                "Betting": AnalysisStyle.BETTING,
                                "Technical": AnalysisStyle.TECHNICAL
                            }

                            engine = PromptEngine(
                                style=style_map[analysis_style])

                            # Build feature summary
                            key_features = ['exp_diff', 'win_rate_diff',
                                            'slpm_diff', 'td_avg_diff', 'streak_diff']
                            feature_summary = ""
                            for fname in key_features:
                                if fname in feature_names:
                                    idx = feature_names.index(fname)
                                    feature_summary += f"{fname}: {features[idx]:+.2f}\n"

                            # Format SHAP if available
                            shap_text = ""
                            if shap_explanation:
                                top_factors = shap_explanation.get(
                                    'top_factors', [])[:5]
                                for f in top_factors:
                                    direction = "→F1" if f['shap_value'] > 0 else "→F2"
                                    shap_text += f"- {f['feature']}: {f['shap_value']:+.3f} {direction}\n"

                            # Build prompt
                            system_prompt = engine.get_system_prompt()
                            user_prompt = engine.build_analysis_prompt(
                                fighter1_name=f1_stats.name,
                                fighter2_name=f2_stats.name,
                                model_predictions=result.get('ensemble', {}),
                                similar_fights=similar if 'similar' in dir() else [],
                                feature_summary=feature_summary,
                                shap_explanation=shap_text,
                                fighter1_context=format_fighter_context(
                                    f1_stats),
                                fighter2_context=format_fighter_context(
                                    f2_stats)
                            )

                            # Generate with refined prompt
                            analysis = analyzer.llm.generate(
                                user_prompt, system_prompt)

                        except ImportError:
                            # Fallback to default analyzer
                            analysis = analyzer.analyze_from_features(
                                features, f1_stats.name, f2_stats.name, verbose=False
                            )

                        st.markdown(analysis)

                    except Exception as e:
                        st.error(f"AI analysis failed: {e}")
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <strong>LLM Provider Not Available</strong><br><br>
                    To enable AI analysis:<br>
                    • <strong>Ollama</strong>: Run <code>ollama serve</code> and install a model<br>
                    • <strong>Claude</strong>: Set <code>ANTHROPIC_API_KEY</code><br>
                    • <strong>OpenAI</strong>: Set <code>OPENAI_API_KEY</code>
                </div>
                """, unsafe_allow_html=True)

    # Show instructions when no prediction yet
    if 'prediction_data' not in st.session_state:
        st.markdown("""
        <div class="info-box">
            <strong>How to use FightPredict:</strong>
            <ol style="margin-top: 0.5rem; margin-bottom: 0;">
                <li>Select a fighter from each corner using the dropdowns above</li>
                <li>Optionally add fight details (title fight, odds)</li>
                <li>Click <strong>PREDICT FIGHT</strong> to see the analysis</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        # Quick stats showcase
        st.markdown("### 🏆 Recent Top Predictions")

        # Sample showcase (static for demo)
        showcase_fights = [
            {"f1": "Islam Makhachev", "f2": "Charles Oliveira",
                "winner": "Makhachev", "conf": "74%"},
            {"f1": "Jon Jones", "f2": "Stipe Miocic",
                "winner": "Jones", "conf": "68%"},
            {"f1": "Alex Pereira", "f2": "Jamahal Hill",
                "winner": "Pereira", "conf": "61%"},
        ]

        cols = st.columns(3)
        for i, fight in enumerate(showcase_fights):
            with cols[i]:
                st.markdown(f"""
                <div class="stat-card">
                    <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem;">
                        {fight['f1']} vs {fight['f2']}
                    </div>
                    <div style="font-family: 'Bebas Neue'; font-size: 1.3rem; color: var(--gold-accent);">
                        {fight['winner']}
                    </div>
                    <div style="font-size: 0.8rem; color: var(--text-muted);">
                        {fight['conf']} confidence
                    </div>
                </div>
                """, unsafe_allow_html=True)

# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    main()
