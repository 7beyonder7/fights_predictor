"""
FightPredict - Prompt Engineering Module
=========================================
Refined prompts for LLM-powered fight analysis.

Features:
- Structured system prompts with MMA expertise
- Chain-of-thought reasoning
- Few-shot examples for consistency
- Betting-focused analysis option
- Confidence calibration guidelines
- Edge case handling (debuts, comebacks, etc.)

Usage:
    from prompts import PromptEngine, AnalysisStyle
    
    engine = PromptEngine(style=AnalysisStyle.DETAILED)
    system_prompt = engine.get_system_prompt()
    user_prompt = engine.build_analysis_prompt(
        fighter1="Islam Makhachev",
        fighter2="Charles Oliveira",
        model_predictions=predictions,
        similar_fights=similar,
        feature_summary=features,
        shap_explanation=shap_data
    )
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np


class AnalysisStyle(Enum):
    """Analysis output styles"""
    QUICK = "quick"           # Brief, key points only
    DETAILED = "detailed"     # Full breakdown
    BETTING = "betting"       # Betting-focused with value analysis
    TECHNICAL = "technical"   # Deep technical MMA analysis


@dataclass
class FighterContext:
    """Fighter context for prompts"""
    name: str
    record: str
    age: int
    stance: str
    height_cm: float
    reach_cm: float
    streak: int
    ufc_fights: int
    slpm: float
    str_acc: float
    td_avg: float
    sub_avg: float


# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPTS = {
    AnalysisStyle.QUICK: """You are a UFC analyst providing quick fight predictions STRICTLY BASED ON THE MODEL DATA PROVIDED.

CRITICAL RULES:
- Your confidence level MUST match the model's confidence (if model says 7%, you say LOW)
- If model confidence is <15%, say "TOO CLOSE TO CALL" - do NOT predict a confident winner
- Do NOT make up statistics or fight history
- Do NOT contradict the model predictions
- Be concise (max 150 words)

CONFIDENCE MAPPING:
- Model <15% = "TOO CLOSE TO CALL" (genuine toss-up)
- Model 15-40% = LOW confidence
- Model 40-60% = MEDIUM confidence  
- Model >60% = HIGH confidence

FORMAT:
🏆 PREDICTION: [Winner] by [Method] OR "TOO CLOSE TO CALL"
📊 CONFIDENCE: [Must match model - if 7% say "TOSS-UP (7%)"]
🔑 KEY FACTORS:
1. [Based on stats provided]
2. [Based on stats provided]
3. [Based on stats provided]""",

    AnalysisStyle.DETAILED: """You are an elite UFC analyst providing analysis GROUNDED IN THE MODEL DATA PROVIDED.

CRITICAL RULES:
- Your confidence MUST align with the model's confidence level
- If model confidence is <15%, acknowledge this is a TOSS-UP - do not claim high confidence
- If models disagree, acknowledge the uncertainty
- Reference the actual statistics provided - do NOT invent numbers
- Do NOT hallucinate fight history or make up facts

YOUR ANALYSIS APPROACH:
1. EXAMINE the model predictions and stats objectively
2. Your confidence level should MATCH the model's confidence
3. IDENTIFY the 3-4 factors from the provided data that matter most
4. ACKNOWLEDGE genuine uncertainty in close fights
5. EXPLAIN your reasoning based on the data, don't contradict it

CONFIDENCE CALIBRATION (must match model):
- Model <15% = TOSS-UP - acknowledge no clear favorite
- Model 15-40% = LOW confidence - slight edge only
- Model 40-60% = MEDIUM confidence - competitive but one fighter has an edge
- Model >60% = HIGH confidence - clear favorite
- Understanding of fighter psychology and momentum

YOUR ANALYSIS APPROACH:
1. EXAMINE the data objectively before forming conclusions
2. IDENTIFY the 3-4 factors that will most likely determine the outcome
3. CONSIDER how styles interact (striker vs grappler, pressure vs counter, etc.)
4. ACKNOWLEDGE genuine uncertainty in close fights
5. EXPLAIN your reasoning, don't just state conclusions

CONFIDENCE CALIBRATION:
- HIGH (>70%): Clear skill gap, dominant stylistic advantage, or significant momentum difference
- MEDIUM (50-70%): Slight edge but competitive matchup, one clear path to victory
- LOW (<50%): True toss-up, both fighters have clear paths to victory

OUTPUT GUIDELINES:
- Be specific: "Makhachev's 5.2 takedowns per fight against Oliveira's 62% TD defense" not "good wrestling"
- Reference the data provided
- Avoid generic MMA clichés ("anything can happen", "on any given night")
- If models disagree, explain what might cause the discrepancy""",

    AnalysisStyle.BETTING: """You are a professional MMA betting analyst focused on finding VALUE, not just picking winners.

YOUR EXPERTISE:
- 10+ years of profitable MMA betting
- Deep understanding of line movement and market inefficiencies
- Statistical modeling background
- Bankroll management principles

ANALYSIS FRAMEWORK:
1. MODEL PROBABILITY: What do the ML models say?
2. MARKET PROBABILITY: What do the odds imply?
3. VALUE IDENTIFICATION: Is there a discrepancy worth betting?
4. RISK ASSESSMENT: What could go wrong with this bet?

CONFIDENCE FOR BETTING:
- STRONG PLAY (3+ units): >10% edge over market, high model confidence, styles favor prediction
- STANDARD PLAY (1-2 units): 5-10% edge, solid model confidence
- LEAN/SMALL PLAY (<1 unit): Slight edge but higher variance
- NO BET: Edge too small or variance too high

IMPORTANT:
- Always compare model probability to implied odds
- Identify specific scenarios that would lose the bet
- Consider prop bets and alternative lines if main line has no value
- Never recommend chasing or emotional betting""",

    AnalysisStyle.TECHNICAL: """You are a technical MMA analyst breaking down the martial arts aspects of this matchup.

YOUR BACKGROUND:
- BJJ black belt with competition experience
- Trained striking (Muay Thai, Boxing)
- Studied film on thousands of UFC fights
- Understanding of modern MMA meta-game

ANALYSIS FOCUS:
1. STRIKING DYNAMICS: Range management, power vs volume, defensive responsibility
2. GRAPPLING EXCHANGES: Takedown entries, scrambles, top control, submission threats
3. CAGE CRAFT: Footwork, clinch work, cage positioning
4. CARDIO & PACING: How will the fight unfold over rounds?
5. INTANGIBLES: Fight IQ, adjustments, championship experience

TECHNICAL VOCABULARY:
- Use proper MMA terminology
- Reference specific techniques when relevant
- Discuss positional hierarchies in grappling
- Analyze distance management in striking

OUTPUT:
Provide analysis that a serious MMA practitioner would appreciate - go beyond surface-level observations."""
}


# =============================================================================
# Few-Shot Examples
# =============================================================================

FEW_SHOT_EXAMPLE = """
**EXAMPLE ANALYSIS:**

Matchup: Islam Makhachev vs Charles Oliveira (Lightweight Title)

**MODEL DATA:**
- Ensemble Prediction: Makhachev (74% confidence)
- XGBoost: Makhachev (71% probability)
- MLP: Makhachev (69% probability)
- Models AGREE

**ANALYSIS:**

🏆 **PREDICTION:** Islam Makhachev via Decision or Late Submission

📊 **CONFIDENCE:** HIGH (74%)

The models strongly favor Makhachev, and the technical breakdown supports this.

**KEY FACTORS:**

1. **Grappling Control** - Makhachev's chain wrestling (3.4 TD/fight, 61% accuracy) will neutralize Oliveira's dangerous guard. Unlike most opponents, Islam doesn't panic in bad positions - he methodically advances.

2. **Takedown Defense Discrepancy** - Oliveira's 62% TD defense vs Makhachev's relentless pressure creates problems. Charles has been taken down 2.7 times per fight in his last 5.

3. **Cardio Edge** - Makhachev maintains pace through 5 rounds. Oliveira has faded in championship rounds (Chandler R2, Poirier R3 rallies came after early trouble).

**STYLE BREAKDOWN:**
Oliveira is most dangerous in scrambles and during striking exchanges where he can land the overhand or drag opponents into his guard. Makhachev will likely fight conservatively on the feet, using feints and level changes to set up takedowns against the cage.

**PATHS TO VICTORY:**
- *Makhachev:* Control rounds with wrestling, ground-and-pound to decision or late submission when Oliveira fatigues
- *Oliveira:* Catch Makhachev in a scramble, guillotine on a takedown attempt, or land the kill shot on the feet

**UPSET POTENTIAL:** 26%
Oliveira's submission game is legitimately elite - he's finished 10 straight opponents. If Islam gets overconfident or sloppy with a takedown attempt, the guillotine is always there.

---
"""


# =============================================================================
# Prompt Templates
# =============================================================================

ANALYSIS_TEMPLATE_DETAILED = """
Analyze this UFC matchup using the data provided:

## FIGHTERS

**🔴 RED CORNER:** {fighter1_name}
{fighter1_context}

**🔵 BLUE CORNER:** {fighter2_name}
{fighter2_context}

## MODEL PREDICTIONS

{model_predictions}

## SIMILAR HISTORICAL FIGHTS

{similar_fights}

## KEY STAT DIFFERENTIALS
(Positive = Fighter 1 advantage)

{feature_summary}

## SHAP FEATURE IMPORTANCE
(What's driving the prediction)

{shap_explanation}

---

Now provide your analysis following this structure:

1. **🏆 PREDICTION** - Winner, method, and round (if applicable)
2. **📊 CONFIDENCE** - High/Medium/Low with percentage and reasoning
3. **🔑 KEY FACTORS** - The 3-4 most important factors (be specific, reference data)
4. **⚔️ STYLE BREAKDOWN** - How the styles interact
5. **🛤️ PATHS TO VICTORY** - How each fighter wins
6. **⚠️ UPSET POTENTIAL** - What percentage chance for upset and why

Be specific and analytical. Reference the numbers provided.
"""

ANALYSIS_TEMPLATE_QUICK = """
Quick analysis needed for: {fighter1_name} vs {fighter2_name}

⚠️ IMPORTANT - YOU MUST FOLLOW THIS MODEL DATA:
{prediction_summary}

{model_disagreement_note}

KEY STATS (positive = F1 advantage):
{feature_summary}

Based ONLY on the model data above:
1. Winner prediction (or "TOO CLOSE TO CALL" if confidence <15%)
2. Your confidence MUST match the model's confidence level
3. Top 3 reasons based on the stats provided

DO NOT invent statistics. DO NOT contradict the model.
"""

ANALYSIS_TEMPLATE_BETTING = """
## BETTING ANALYSIS: {fighter1_name} vs {fighter2_name}

### MODEL PREDICTIONS
{model_predictions}

### MARKET ODDS (if provided)
{odds_info}

### HISTORICAL SIMILAR FIGHTS
{similar_fights}

### KEY DIFFERENTIALS
{feature_summary}

---

Provide betting-focused analysis:

1. **📈 MODEL EDGE ANALYSIS**
   - Model probability vs implied probability
   - Is there value on either side?

2. **🎯 RECOMMENDED PLAYS**
   - Main line recommendation (with unit sizing)
   - Any prop bet value?
   - Round/method props worth considering?

3. **⚠️ RISK FACTORS**
   - What scenarios would lose this bet?
   - Variance assessment

4. **💰 BOTTOM LINE**
   - Clear recommendation with reasoning
"""

ANALYSIS_TEMPLATE_TECHNICAL = """
## TECHNICAL BREAKDOWN: {fighter1_name} vs {fighter2_name}

### FIGHTER PROFILES
{fighter1_context}
---
{fighter2_context}

### STATISTICAL COMPARISON
{feature_summary}

### MODEL ANALYSIS
{model_predictions}

---

Provide deep technical analysis:

1. **🥊 STRIKING ANALYSIS**
   - Range dynamics and distance management
   - Power vs volume approach
   - Defensive tendencies and vulnerabilities

2. **🤼 GRAPPLING ANALYSIS**
   - Takedown game (entries, timing, setups)
   - Ground control and advancement
   - Submission threats and defense

3. **🏟️ CAGE CRAFT & STRATEGY**
   - Likely game plans for each fighter
   - Pacing and cardio considerations
   - Adjustments to expect between rounds

4. **🎯 TECHNICAL PREDICTION**
   - Most likely sequence of events
   - Key moments to watch for
   - Technical skills that will determine outcome
"""


# =============================================================================
# Edge Case Prompts
# =============================================================================

EDGE_CASE_ADDITIONS = {
    "debut": """
⚠️ **DEBUT FIGHTER ALERT**
{fighter_name} is making their UFC debut. Consider:
- Octagon jitters and adjustment period
- Unknown variables not captured in regional stats
- Increase uncertainty in prediction
""",

    "comeback": """
⚠️ **COMEBACK FIGHT**
{fighter_name} is returning after {days} days (>{months} months) away. Consider:
- Ring rust factor
- Age-related decline if applicable
- Motivation and preparation unknowns
""",

    "short_notice": """
⚠️ **SHORT NOTICE REPLACEMENT**
This appears to be a short-notice matchup. Consider:
- Reduced preparation time
- Weight cut concerns
- Game plan limitations
""",

    "moving_weight": """
⚠️ **WEIGHT CLASS CHANGE**
{fighter_name} is fighting at a new weight class. Consider:
- Size differential adjustments
- Potential strength/cardio changes
- Limited data at new weight
""",

    "close_fight": """
⚠️ **EXTREMELY CLOSE MATCHUP**
Models show low confidence ({confidence}%). This is a genuine toss-up.
- Both fighters have clear paths to victory
- Small factors could swing this either way
- Consider: who has more ways to win?
""",

    "model_disagreement": """
⚠️ **MODEL DISAGREEMENT**
XGBoost predicts {xgb_winner} but MLP predicts {mlp_winner}.
- This often indicates a close fight
- Check which model's strengths apply here
- XGBoost better with tree-structured patterns
- MLP may capture different nonlinear relationships
"""
}


# =============================================================================
# Prompt Engine Class
# =============================================================================

class PromptEngine:
    """
    Main class for generating refined prompts.

    Usage:
        engine = PromptEngine(style=AnalysisStyle.DETAILED)
        system = engine.get_system_prompt()
        prompt = engine.build_analysis_prompt(...)
    """

    def __init__(self, style: AnalysisStyle = AnalysisStyle.DETAILED):
        self.style = style
        self.include_few_shot = style == AnalysisStyle.DETAILED

    def get_system_prompt(self) -> str:
        """Get the system prompt for current style."""
        return SYSTEM_PROMPTS[self.style]

    def build_analysis_prompt(
        self,
        fighter1_name: str,
        fighter2_name: str,
        model_predictions: Dict[str, Any],
        similar_fights: List[Dict] = None,
        feature_summary: str = "",
        shap_explanation: str = "",
        fighter1_context: str = "",
        fighter2_context: str = "",
        odds_info: str = "",
        edge_cases: List[str] = None
    ) -> str:
        """
        Build the full analysis prompt.

        Args:
            fighter1_name: Red corner fighter
            fighter2_name: Blue corner fighter
            model_predictions: Dict with ensemble predictions
            similar_fights: List of similar historical fights
            feature_summary: Formatted feature differentials
            shap_explanation: SHAP-based explanation of prediction drivers
            fighter1_context: Fighter 1 stats summary
            fighter2_context: Fighter 2 stats summary
            odds_info: Betting odds if available
            edge_cases: List of edge case keys to include

        Returns:
            Complete prompt string
        """
        # Format model predictions
        pred_text = self._format_predictions(
            model_predictions, fighter1_name, fighter2_name)

        # Format similar fights
        similar_text = self._format_similar_fights(
            similar_fights) if similar_fights else "No similar fights data available."

        # Select template based on style
        if self.style == AnalysisStyle.QUICK:
            template = ANALYSIS_TEMPLATE_QUICK

            # Build detailed prediction summary for Quick mode
            conf = model_predictions.get("ensemble_confidence", 0)
            prob_f1 = model_predictions.get("ensemble_probability_f1", 0.5)
            models_agree = model_predictions.get("models_agree", True)

            if conf < 0.15:
                pred_summary = f"⚠️ TOO CLOSE TO CALL - Model confidence is only {conf:.0%}\n"
                pred_summary += f"Probabilities: {fighter1_name} {prob_f1:.0%} vs {fighter2_name} {1-prob_f1:.0%}"
            else:
                winner = fighter1_name if model_predictions.get(
                    "ensemble_prediction", 0) == 1 else fighter2_name
                pred_summary = f"{winner} predicted to win\n"
                pred_summary += f"Confidence: {conf:.0%}\n"
                pred_summary += f"Probabilities: {fighter1_name} {prob_f1:.0%} vs {fighter2_name} {1-prob_f1:.0%}"

            # Model disagreement note
            if not models_agree:
                disagree_note = "⚠️ MODELS DISAGREE - XGBoost and MLP predict different winners. This is a close fight."
            else:
                disagree_note = "✅ Both models agree on the prediction."

            prompt = template.format(
                fighter1_name=fighter1_name,
                fighter2_name=fighter2_name,
                prediction_summary=pred_summary,
                model_disagreement_note=disagree_note,
                feature_summary=feature_summary
            )

        elif self.style == AnalysisStyle.BETTING:
            template = ANALYSIS_TEMPLATE_BETTING
            prompt = template.format(
                fighter1_name=fighter1_name,
                fighter2_name=fighter2_name,
                model_predictions=pred_text,
                similar_fights=similar_text,
                feature_summary=feature_summary,
                odds_info=odds_info or "Market odds not provided."
            )

        elif self.style == AnalysisStyle.TECHNICAL:
            template = ANALYSIS_TEMPLATE_TECHNICAL
            prompt = template.format(
                fighter1_name=fighter1_name,
                fighter2_name=fighter2_name,
                model_predictions=pred_text,
                feature_summary=feature_summary,
                fighter1_context=fighter1_context or "Stats not available",
                fighter2_context=fighter2_context or "Stats not available"
            )

        else:  # DETAILED (default)
            template = ANALYSIS_TEMPLATE_DETAILED
            prompt = template.format(
                fighter1_name=fighter1_name,
                fighter2_name=fighter2_name,
                model_predictions=pred_text,
                similar_fights=similar_text,
                feature_summary=feature_summary,
                shap_explanation=shap_explanation or "SHAP analysis not available.",
                fighter1_context=fighter1_context or "Stats not available",
                fighter2_context=fighter2_context or "Stats not available"
            )

            # Add few-shot example for detailed analysis
            if self.include_few_shot:
                prompt = FEW_SHOT_EXAMPLE + "\n\n---\n\nNOW ANALYZE THIS FIGHT:\n\n" + prompt

        # Add edge case warnings
        if edge_cases:
            edge_text = self._format_edge_cases(
                edge_cases, model_predictions, fighter1_name, fighter2_name)
            prompt = edge_text + "\n\n" + prompt

        return prompt

    def _format_predictions(self, pred: Dict, f1: str, f2: str) -> str:
        """Format model predictions section."""
        if not pred:
            return "Model predictions unavailable."

        lines = []

        winner = f1 if pred.get("ensemble_prediction", 0) == 1 else f2
        confidence = pred.get("ensemble_confidence", 0)
        prob_f1 = pred.get("ensemble_probability_f1", 0.5)

        lines.append(f"**Ensemble Prediction:** {winner}")
        lines.append(f"**Confidence:** {confidence:.1%}")
        lines.append(
            f"**Probability:** {f1}: {prob_f1:.1%} | {f2}: {1-prob_f1:.1%}")

        if pred.get("models_agree"):
            lines.append("**Agreement:** ✅ Both models agree")
        else:
            lines.append(f"**Agreement:** ⚠️ Models disagree")

        lines.append("")
        lines.append("**Individual Models:**")

        for p in pred.get("individual_predictions", []):
            model_winner = f1 if p["prediction"] == 1 else f2
            lines.append(
                f"- {p['model']}: {model_winner} "
                f"(P({f1})={p['probability_f1']:.1%}, conf={p['confidence']:.1%})"
            )

        return "\n".join(lines)

    def _format_similar_fights(self, fights: List[Dict]) -> str:
        """Format similar fights section."""
        if not fights:
            return "No similar historical fights found."

        lines = [f"Top {len(fights[:5])} similar historical matchups:"]

        f1_wins = 0
        for i, fight in enumerate(fights[:5], 1):
            f1 = fight.get('f_1_name', 'Fighter 1')
            f2 = fight.get('f_2_name', 'Fighter 2')
            winner = fight.get('winner', 'Unknown')
            result = fight.get('result', '')
            similarity = fight.get('similarity', 0)

            if fight.get('target', 0) == 1 or winner == f1:
                f1_wins += 1
                outcome = "F1 Won"
            else:
                outcome = "F2 Won"

            lines.append(
                f"{i}. {f1} vs {f2} → {winner} ({result}) [{similarity:.0%} similar] - {outcome}")

        lines.append(
            f"\n**Pattern:** Fighter 1 profile won {f1_wins}/{len(fights[:5])} similar matchups ({f1_wins/len(fights[:5])*100:.0f}%)")

        return "\n".join(lines)

    def _get_prediction_summary(self, pred: Dict, f1: str, f2: str) -> str:
        """Get brief prediction summary for quick template."""
        if not pred:
            return "No prediction available"

        winner = f1 if pred.get("ensemble_prediction", 0) == 1 else f2
        conf = pred.get("ensemble_confidence", 0)
        return f"{winner} ({conf:.0%} confidence)"

    def _format_edge_cases(self, cases: List[str], pred: Dict, f1: str, f2: str) -> str:
        """Format edge case warnings."""
        warnings = []

        for case in cases:
            if case in EDGE_CASE_ADDITIONS:
                template = EDGE_CASE_ADDITIONS[case]

                # Fill in template variables based on case
                if case == "close_fight":
                    text = template.format(
                        confidence=f"{pred.get('ensemble_confidence', 0):.0%}")
                elif case == "model_disagreement":
                    preds = pred.get("individual_predictions", [])
                    if len(preds) >= 2:
                        xgb = next(
                            (p for p in preds if p["model"] == "XGBoost"), None)
                        mlp = next(
                            (p for p in preds if p["model"] == "MLP"), None)
                        if xgb and mlp:
                            text = template.format(
                                xgb_winner=f1 if xgb["prediction"] == 1 else f2,
                                mlp_winner=f1 if mlp["prediction"] == 1 else f2
                            )
                        else:
                            continue
                    else:
                        continue
                else:
                    text = template

                warnings.append(text)

        return "\n".join(warnings) if warnings else ""

    def detect_edge_cases(
        self,
        f1_stats: Any,
        f2_stats: Any,
        model_predictions: Dict
    ) -> List[str]:
        """
        Automatically detect edge cases from fighter stats.

        Returns list of edge case keys to include.
        """
        cases = []

        # Check for debuts
        if hasattr(f1_stats, 'ufc_fights') and f1_stats.ufc_fights == 0:
            cases.append("debut")
        if hasattr(f2_stats, 'ufc_fights') and f2_stats.ufc_fights == 0:
            cases.append("debut")

        # Check for ring rust (>365 days)
        if hasattr(f1_stats, 'days_since_last_fight') and f1_stats.days_since_last_fight > 365:
            cases.append("comeback")
        if hasattr(f2_stats, 'days_since_last_fight') and f2_stats.days_since_last_fight > 365:
            cases.append("comeback")

        # Check for close fight
        confidence = model_predictions.get("ensemble_confidence", 1.0)
        if confidence < 0.4:
            cases.append("close_fight")

        # Check for model disagreement
        if not model_predictions.get("models_agree", True):
            cases.append("model_disagreement")

        return cases


# =============================================================================
# SHAP Explanation Formatter
# =============================================================================

def format_shap_explanation(
    shap_data: Dict,
    top_n: int = 5,
    fighter1_name: str = "Fighter 1",
    fighter2_name: str = "Fighter 2"
) -> str:
    """
    Format SHAP explanation for prompt inclusion.

    Args:
        shap_data: Output from SHAPExplainer.explain_single_prediction()
        top_n: Number of top features to include
        fighter1_name: Name of fighter 1
        fighter2_name: Name of fighter 2

    Returns:
        Formatted string for prompt
    """
    if not shap_data:
        return "SHAP analysis not available."

    lines = []

    # Top features pushing toward prediction
    top_for = shap_data.get("top_positive", [])[:top_n]
    top_against = shap_data.get("top_negative", [])[:top_n]

    predicted_winner = shap_data.get(
        "predicted_winner", "the predicted winner")

    lines.append(f"**Factors favoring {predicted_winner}:**")
    for feat in top_for:
        name = feat.get("feature", "unknown")
        impact = feat.get("impact", 0)
        value = feat.get("value", 0)
        lines.append(f"- {name}: {value:+.2f} (impact: {impact:+.4f})")

    lines.append("")
    lines.append(f"**Factors against prediction:**")
    for feat in top_against:
        name = feat.get("feature", "unknown")
        impact = feat.get("impact", 0)
        value = feat.get("value", 0)
        lines.append(f"- {name}: {value:+.2f} (impact: {impact:+.4f})")

    return "\n".join(lines)


def format_fighter_context(stats: Any) -> str:
    """Format fighter stats for prompt context."""
    if stats is None:
        return "Stats not available"

    lines = [
        f"Record: {stats.wins}-{stats.losses}-{stats.draws}",
        f"Age: {stats.age:.0f} | Stance: {stats.stance}",
        f"Height: {stats.height_cm:.0f}cm | Reach: {stats.reach_cm:.0f}cm",
        f"UFC Fights: {stats.ufc_fights} | Streak: {stats.current_streak:+d}",
        f"Striking: {stats.slpm:.2f} SLpM, {stats.str_acc:.0%} acc, {stats.str_def:.0%} def",
        f"Grappling: {stats.td_avg:.2f} TD/15min, {stats.td_acc:.0%} acc, {stats.td_def:.0%} def",
        f"Submissions: {stats.sub_avg:.2f} per 15min"
    ]

    return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_quick_prompt(f1: str, f2: str, prediction: Dict, features: str) -> tuple:
    """Get quick analysis prompt."""
    engine = PromptEngine(style=AnalysisStyle.QUICK)
    return engine.get_system_prompt(), engine.build_analysis_prompt(
        fighter1_name=f1,
        fighter2_name=f2,
        model_predictions=prediction,
        feature_summary=features
    )


def get_detailed_prompt(
    f1: str, f2: str,
    prediction: Dict,
    similar: List[Dict],
    features: str,
    shap: str,
    f1_context: str,
    f2_context: str
) -> tuple:
    """Get detailed analysis prompt."""
    engine = PromptEngine(style=AnalysisStyle.DETAILED)
    return engine.get_system_prompt(), engine.build_analysis_prompt(
        fighter1_name=f1,
        fighter2_name=f2,
        model_predictions=prediction,
        similar_fights=similar,
        feature_summary=features,
        shap_explanation=shap,
        fighter1_context=f1_context,
        fighter2_context=f2_context
    )


def get_betting_prompt(
    f1: str, f2: str,
    prediction: Dict,
    similar: List[Dict],
    features: str,
    odds: str = ""
) -> tuple:
    """Get betting analysis prompt."""
    engine = PromptEngine(style=AnalysisStyle.BETTING)
    return engine.get_system_prompt(), engine.build_analysis_prompt(
        fighter1_name=f1,
        fighter2_name=f2,
        model_predictions=prediction,
        similar_fights=similar,
        feature_summary=features,
        odds_info=odds
    )


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Demo the prompt engine
    engine = PromptEngine(style=AnalysisStyle.DETAILED)

    print("=" * 60)
    print("SYSTEM PROMPT:")
    print("=" * 60)
    print(engine.get_system_prompt())

    print("\n" + "=" * 60)
    print("SAMPLE USER PROMPT:")
    print("=" * 60)

    # Mock data
    mock_predictions = {
        "ensemble_prediction": 1,
        "ensemble_confidence": 0.74,
        "ensemble_probability_f1": 0.82,
        "models_agree": True,
        "individual_predictions": [
            {"model": "XGBoost", "prediction": 1,
                "probability_f1": 0.79, "confidence": 0.71},
            {"model": "MLP", "prediction": 1,
                "probability_f1": 0.76, "confidence": 0.68}
        ]
    }

    mock_similar = [
        {"f_1_name": "Khabib", "f_2_name": "McGregor", "winner": "Khabib",
            "result": "Submission", "similarity": 0.89, "target": 1},
        {"f_1_name": "Khabib", "f_2_name": "Poirier", "winner": "Khabib",
            "result": "Submission", "similarity": 0.85, "target": 1},
    ]

    prompt = engine.build_analysis_prompt(
        fighter1_name="Islam Makhachev",
        fighter2_name="Charles Oliveira",
        model_predictions=mock_predictions,
        similar_fights=mock_similar,
        feature_summary="exp_diff: +5.0\nwin_rate_diff: +0.12\ntd_avg_diff: +2.1",
        shap_explanation="Top factors: exp_diff (+0.15), td_avg_diff (+0.12)",
        fighter1_context="Record: 26-1\nStreak: +15",
        fighter2_context="Record: 34-9\nStreak: +3"
    )

    print(prompt[:2000] + "...\n[truncated]")
