"""
FightPredict - Matchup Feature Generator
==============================================

Generates feature vectors for hypothetical matchups using the EXACT same
approach as feature_engineering.py:

1. Pull fighter stats from their MOST RECENT FIGHT in fights_cleaned.csv
2. Compute temporal features (UFC fights, streak, days since fight) from fight history
3. Use identical formulas to feature_engineering.py

This ensures compatibility with trained models.

Usage:
    from generate_matchup_features import MatchupFeatureGenerator
    
    gen = MatchupFeatureGenerator()
    gen.load_data()
    
    features, names = gen.generate_features("Islam Makhachev", "Charles Oliveira")
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import warnings

# Import unified fighter database (handle both direct and package imports)
try:
    from fighters import FighterDatabase, FighterSnapshot
except ImportError:
    from src.fighters import FighterDatabase, FighterSnapshot


class MatchupFeatureGenerator:
    """
    Generates feature vectors for hypothetical matchups.

    KEY: Uses fighter stats from their most recent fight in fights_cleaned.csv,
    matching how feature_engineering.py works.

    Uses unified FighterDatabase for fighter lookups.
    """

    STANCE_ENCODING = {
        'Orthodox': 0,
        'Southpaw': 1,
        'Switch': 2,
        'Open Stance': 3,
        'Sideways': 4,
    }

    WEIGHT_CLASS_ENCODING = {
        'Strawweight': 0, "Women's Strawweight": 0,
        'Flyweight': 1, "Women's Flyweight": 1,
        'Bantamweight': 2, "Women's Bantamweight": 2,
        'Featherweight': 3, "Women's Featherweight": 3,
        'Lightweight': 4,
        'Welterweight': 5,
        'Middleweight': 6,
        'Light Heavyweight': 7,
        'Heavyweight': 8,
        'Super Heavyweight': 9,
        'Catch Weight': 5, 'Open Weight': 5, 'Unknown': 5
    }

    def __init__(
        self,
        fights_path: str = "data/processed/fights_cleaned.csv",
        features_path: str = "data/processed/features_model_ready.csv"
    ):
        self.fights_path = fights_path
        self.features_path = features_path

        self.fights_df = None
        self.feature_names = []
        self.feature_names_with_odds = []

        # Use unified fighter database
        self._fighter_db = FighterDatabase(fights_path)

    def load_data(self) -> bool:
        """Load fight data."""
        try:
            # Load fighter database
            if not self._fighter_db.load():
                return False

            # Keep reference to fights_df for backward compatibility
            self.fights_df = self._fighter_db.fights_df

            # Load feature names
            features_df = pd.read_csv(self.features_path, nrows=1)
            self.feature_names = [
                c for c in features_df.columns if c != 'target']

            try:
                odds_path = self.features_path.replace(
                    '.csv', '_with_odds.csv')
                odds_df = pd.read_csv(odds_path, nrows=1)
                self.feature_names_with_odds = [
                    c for c in odds_df.columns if c != 'target']
            except:
                self.feature_names_with_odds = self.feature_names

            print(
                f"✓ Loaded {len(self.feature_names)} feature names ({len(self.feature_names_with_odds)} with odds)")
            return True

        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False

    def search_fighter(self, query: str, limit: int = 10) -> List[str]:
        """Search for fighters by partial name match."""
        return self._fighter_db.search(query, limit)

    def get_fighter_snapshot(self, name: str) -> Optional[FighterSnapshot]:
        """
        Get fighter's stats from their most recent fight.
        Delegates to unified FighterDatabase.
        """
        return self._fighter_db.get_snapshot(name)

    def generate_features(
        self,
        fighter1: str,
        fighter2: str,
        weight_class: str = "Unknown",
        title_fight: bool = False,
        f1_odds: Optional[float] = None,
        f2_odds: Optional[float] = None,
        include_odds: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate feature vector for a hypothetical matchup.

        Uses stats from each fighter's most recent fight, matching
        how feature_engineering.py processes data.
        """
        # Get fighter snapshots
        f1 = self.get_fighter_snapshot(fighter1)
        f2 = self.get_fighter_snapshot(fighter2)

        if f1 is None:
            raise ValueError(f"Fighter not found: {fighter1}")
        if f2 is None:
            raise ValueError(f"Fighter not found: {fighter2}")

        features = {}

        # =====================================================================
        # PHYSICAL FEATURES
        # =====================================================================
        features['height_diff'] = f1.height_cm - f2.height_cm
        features['reach_diff'] = f1.reach_cm - f2.reach_cm
        features['age_diff'] = f2.age - f1.age  # Younger is advantage for f1

        # Ape index = reach / height (RATIO)
        f1_ape = f1.reach_cm / (f1.height_cm + 1e-6)
        f2_ape = f2.reach_cm / (f2.height_cm + 1e-6)
        features['ape_index_diff'] = f1_ape - f2_ape

        # Prime age (27-32)
        f1_in_prime = 1 if 27 <= f1.age <= 32 else 0
        f2_in_prime = 1 if 27 <= f2.age <= 32 else 0
        features['f_1_in_prime'] = f1_in_prime
        features['f_2_in_prime'] = f2_in_prime
        features['prime_advantage'] = f1_in_prime - f2_in_prime

        # Age risk
        features['f_1_age_risk'] = 1 if (f1.age < 23 or f1.age > 37) else 0
        features['f_2_age_risk'] = 1 if (f2.age < 23 or f2.age > 37) else 0

        # =====================================================================
        # CAREER FEATURES
        # =====================================================================
        f1_total = f1.wins + f1.losses + f1.draws
        f2_total = f2.wins + f2.losses + f2.draws

        features['f_1_win_rate'] = f1.wins / (f1_total + 1e-6)
        features['f_2_win_rate'] = f2.wins / (f2_total + 1e-6)
        features['win_rate_diff'] = features['f_1_win_rate'] - \
            features['f_2_win_rate']

        features['f_1_loss_rate'] = f1.losses / (f1_total + 1e-6)
        features['f_2_loss_rate'] = f2.losses / (f2_total + 1e-6)
        features['loss_rate_diff'] = features['f_2_loss_rate'] - \
            features['f_1_loss_rate']

        features['exp_diff'] = f1_total - f2_total

        features['f_1_exp_level'] = self._get_exp_level(f1_total)
        features['f_2_exp_level'] = self._get_exp_level(f2_total)

        features['f_1_record_score'] = (
            f1.wins - f1.losses) / (f1_total + 1e-6)
        features['f_2_record_score'] = (
            f2.wins - f2.losses) / (f2_total + 1e-6)
        features['record_score_diff'] = features['f_1_record_score'] - \
            features['f_2_record_score']

        # =====================================================================
        # STRIKING FEATURES
        # =====================================================================
        features['f_1_slpm'] = f1.slpm
        features['f_2_slpm'] = f2.slpm
        features['slpm_diff'] = f1.slpm - f2.slpm

        features['f_1_str_acc'] = f1.str_acc
        features['f_2_str_acc'] = f2.str_acc
        features['str_acc_diff'] = f1.str_acc - f2.str_acc

        features['f_1_sapm'] = f1.sapm
        features['f_2_sapm'] = f2.sapm
        features['sapm_diff'] = f2.sapm - f1.sapm  # Less absorbed is better

        features['f_1_str_def'] = f1.str_def
        features['f_2_str_def'] = f2.str_def
        features['str_def_diff'] = f1.str_def - f2.str_def

        # Striking differential
        f1_str_diff = f1.slpm - f1.sapm
        f2_str_diff = f2.slpm - f2.sapm
        features['f_1_str_differential'] = f1_str_diff
        features['f_2_str_differential'] = f2_str_diff
        features['str_differential_diff'] = f1_str_diff - f2_str_diff

        # =====================================================================
        # GRAPPLING FEATURES
        # =====================================================================
        features['f_1_td_avg'] = f1.td_avg
        features['f_2_td_avg'] = f2.td_avg
        features['td_avg_diff'] = f1.td_avg - f2.td_avg

        features['f_1_td_acc'] = f1.td_acc
        features['f_2_td_acc'] = f2.td_acc
        features['td_acc_diff'] = f1.td_acc - f2.td_acc

        features['f_1_td_def'] = f1.td_def
        features['f_2_td_def'] = f2.td_def
        features['td_def_diff'] = f1.td_def - f2.td_def

        features['f_1_sub_avg'] = f1.sub_avg
        features['f_2_sub_avg'] = f2.sub_avg
        features['sub_avg_diff'] = f1.sub_avg - f2.sub_avg

        # =====================================================================
        # STYLE FEATURES
        # =====================================================================
        f1_striker = f1.slpm * f1.str_acc
        f2_striker = f2.slpm * f2.str_acc
        features['f_1_striker_score'] = f1_striker
        features['f_2_striker_score'] = f2_striker

        f1_grappler = f1.td_avg + f1.sub_avg
        f2_grappler = f2.td_avg + f2.sub_avg
        features['f_1_grappler_score'] = f1_grappler
        features['f_2_grappler_score'] = f2_grappler

        features['f_1_style_ratio'] = f1_striker / (f1_grappler + 1e-6)
        features['f_2_style_ratio'] = f2_striker / (f2_grappler + 1e-6)

        # Effectiveness = str_differential + td_avg*2 + sub_avg*3
        f1_eff = f1_str_diff + f1.td_avg * 2 + f1.sub_avg * 3
        f2_eff = f2_str_diff + f2.td_avg * 2 + f2.sub_avg * 3
        features['f_1_effectiveness'] = f1_eff
        features['f_2_effectiveness'] = f2_eff
        features['effectiveness_diff'] = f1_eff - f2_eff

        # =====================================================================
        # FINISH FEATURES
        # =====================================================================
        f1_ko = f1.slpm * (1 - f1.str_def + 0.5)
        f2_ko = f2.slpm * (1 - f2.str_def + 0.5)
        features['f_1_ko_power'] = f1_ko
        features['f_2_ko_power'] = f2_ko
        features['ko_power_diff'] = f1_ko - f2_ko

        f1_sub_threat = f1.sub_avg * (f1.td_avg + 1)
        f2_sub_threat = f2.sub_avg * (f2.td_avg + 1)
        features['f_1_sub_threat'] = f1_sub_threat
        features['f_2_sub_threat'] = f2_sub_threat
        features['sub_threat_diff'] = f1_sub_threat - f2_sub_threat

        f1_durability = f1.str_def / (f1.sapm + 1e-6)
        f2_durability = f2.str_def / (f2.sapm + 1e-6)
        features['f_1_durability'] = f1_durability
        features['f_2_durability'] = f2_durability
        features['durability_diff'] = f1_durability - f2_durability

        features['f_1_finish_potential'] = f1_ko + f1_sub_threat
        features['f_2_finish_potential'] = f2_ko + f2_sub_threat
        features['finish_potential_diff'] = features['f_1_finish_potential'] - \
            features['f_2_finish_potential']

        # =====================================================================
        # EXPERIENCE FEATURES
        # =====================================================================
        features['f_1_ufc_fights'] = f1.ufc_fights
        features['f_2_ufc_fights'] = f2.ufc_fights
        features['ufc_exp_diff'] = f1.ufc_fights - f2.ufc_fights

        features['f_1_is_debut'] = 1 if f1.ufc_fights == 0 else 0
        features['f_2_is_debut'] = 1 if f2.ufc_fights == 0 else 0
        features['debut_matchup'] = features['f_1_is_debut'] + \
            features['f_2_is_debut']

        features['f_1_is_veteran'] = 1 if f1.ufc_fights >= 10 else 0
        features['f_2_is_veteran'] = 1 if f2.ufc_fights >= 10 else 0
        features['veteran_vs_rookie'] = features['f_1_is_veteran'] - \
            features['f_2_is_veteran']

        # =====================================================================
        # MOMENTUM FEATURES
        # =====================================================================
        features['f_1_streak'] = f1.current_streak
        features['f_2_streak'] = f2.current_streak
        features['streak_diff'] = f1.current_streak - f2.current_streak

        features['f_1_on_win_streak'] = 1 if f1.current_streak >= 2 else 0
        features['f_2_on_win_streak'] = 1 if f2.current_streak >= 2 else 0

        features['f_1_on_lose_streak'] = 1 if f1.current_streak <= -2 else 0
        features['f_2_on_lose_streak'] = 1 if f2.current_streak <= -2 else 0

        features['f_1_days_since_fight'] = f1.days_since_last_fight
        features['f_2_days_since_fight'] = f2.days_since_last_fight

        features['f_1_ring_rust'] = 1 if f1.days_since_last_fight > 365 else 0
        features['f_2_ring_rust'] = 1 if f2.days_since_last_fight > 365 else 0

        # Activity diff = f2_days - f1_days (positive = f1 more active)
        features['activity_diff'] = f2.days_since_last_fight - \
            f1.days_since_last_fight

        # =====================================================================
        # MATCHUP FEATURES
        # =====================================================================
        f1_stance_enc = self.STANCE_ENCODING.get(f1.stance, 0)
        f2_stance_enc = self.STANCE_ENCODING.get(f2.stance, 0)

        features['stance_clash'] = 1 if f1_stance_enc != f2_stance_enc else 0
        features['style_clash'] = abs(
            features['f_1_style_ratio'] - features['f_2_style_ratio'])

        features['f_1_grappler_vs_f_2_striker'] = 1 if (
            f1_grappler > f1_striker and f2_striker > f2_grappler) else 0
        features['f_1_striker_vs_f_2_grappler'] = 1 if (
            f1_striker > f1_grappler and f2_grappler > f2_striker) else 0

        features['td_battle'] = f1.td_avg * \
            (1 - f2.td_def) - f2.td_avg * (1 - f1.td_def)
        features['striking_battle'] = (f1.slpm * f1.str_acc * (1 - f2.str_def) -
                                       f2.slpm * f2.str_acc * (1 - f1.str_def))
        features['matchup_score'] = features['striking_battle'] + \
            features['td_battle'] * 2 + features['sub_avg_diff'] * 3

        # =====================================================================
        # CATEGORICAL FEATURES
        # =====================================================================
        features['f_1_stance_encoded'] = f1_stance_enc
        features['f_2_stance_encoded'] = f2_stance_enc
        features['weight_class_encoded'] = self.WEIGHT_CLASS_ENCODING.get(
            weight_class, 5)
        features['title_fight'] = 1 if title_fight else 0

        # =====================================================================
        # BETTING FEATURES (optional)
        # =====================================================================
        if include_odds and f1_odds is not None and f2_odds is not None:
            features['f_1_odds'] = f1_odds
            features['f_2_odds'] = f2_odds
            features['odds_diff'] = f1_odds - f2_odds
            features['f_1_is_favorite'] = 1.0 if f1_odds < f2_odds else 0.0

            features['f_1_implied_prob'] = 1 / (f1_odds + 1e-6)
            features['f_2_implied_prob'] = 1 / (f2_odds + 1e-6)
            total_prob = features['f_1_implied_prob'] + \
                features['f_2_implied_prob']
            features['f_1_implied_prob_norm'] = features['f_1_implied_prob'] / \
                (total_prob + 1e-6)
            features['f_2_implied_prob_norm'] = features['f_2_implied_prob'] / \
                (total_prob + 1e-6)
            features['odds_value_diff'] = features['f_1_implied_prob_norm'] - \
                features['f_2_implied_prob_norm']
            features['f_1_heavy_favorite'] = 1.0 if f1_odds < 1.4 else 0.0
            features['f_1_heavy_underdog'] = 1.0 if f1_odds > 3.0 else 0.0

        # =====================================================================
        # BUILD FEATURE VECTOR
        # =====================================================================
        target_features = self.feature_names_with_odds if include_odds else self.feature_names

        feature_vector = []
        missing_features = []

        for feat_name in target_features:
            if feat_name in features:
                feature_vector.append(features[feat_name])
            else:
                missing_features.append(feat_name)
                feature_vector.append(0.0)

        if missing_features:
            warnings.warn(
                f"Missing {len(missing_features)} features: {missing_features[:5]}...")

        return np.array(feature_vector, dtype=np.float32), target_features

    def _get_exp_level(self, total_fights: int) -> float:
        """Experience level category (0-3)."""
        if total_fights <= 5:
            return 0.0
        elif total_fights <= 15:
            return 1.0
        elif total_fights <= 30:
            return 2.0
        return 3.0

    # Aliases for compatibility
    def get_fighter_stats(self, name: str) -> Optional[FighterSnapshot]:
        """Alias for get_fighter_snapshot for compatibility."""
        return self.get_fighter_snapshot(name)

    def get_all_fighter_names(self) -> List[str]:
        """Get all fighter names for autocomplete."""
        return self._fighter_db.get_all_names()

    def get_matchup_summary(self, fighter1: str, fighter2: str) -> Dict[str, Any]:
        """Get human-readable matchup summary."""
        f1 = self.get_fighter_snapshot(fighter1)
        f2 = self.get_fighter_snapshot(fighter2)

        if f1 is None or f2 is None:
            return {"error": "Fighter(s) not found"}

        return {
            "fighter1": {
                "name": f1.name, "record": f"{f1.wins}-{f1.losses}-{f1.draws}",
                "age": round(f1.age, 1), "height_cm": f1.height_cm, "reach_cm": f1.reach_cm,
                "stance": f1.stance, "ufc_fights": f1.ufc_fights, "streak": f1.current_streak,
                "slpm": f1.slpm, "str_acc": f1.str_acc, "td_avg": f1.td_avg, "sub_avg": f1.sub_avg
            },
            "fighter2": {
                "name": f2.name, "record": f"{f2.wins}-{f2.losses}-{f2.draws}",
                "age": round(f2.age, 1), "height_cm": f2.height_cm, "reach_cm": f2.reach_cm,
                "stance": f2.stance, "ufc_fights": f2.ufc_fights, "streak": f2.current_streak,
                "slpm": f2.slpm, "str_acc": f2.str_acc, "td_avg": f2.td_avg, "sub_avg": f2.sub_avg
            }
        }


def main():
    """CLI for testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate matchup features (v3)")
    parser.add_argument("--fighter1", "-f1", type=str)
    parser.add_argument("--fighter2", "-f2", type=str)
    parser.add_argument("--search", "-s", type=str)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--index", type=int, default=100)

    args = parser.parse_args()

    gen = MatchupFeatureGenerator()
    if not gen.load_data():
        return

    if args.search:
        results = gen.search_fighter(args.search)
        print(f"\nFound {len(results)} fighters:")
        for name in results:
            print(f"  • {name}")

    elif args.compare:
        import pandas as pd
        features_df = pd.read_csv("data/processed/features_model_ready.csv")
        fights_df = pd.read_csv("data/processed/fights_cleaned.csv")

        fight = fights_df.iloc[args.index]
        print(f"\nComparing: {fight['f_1_name']} vs {fight['f_2_name']}")
        print(f"Event: {fight['event_name']}")

        original = features_df.iloc[args.index].drop('target').values
        new_features, names = gen.generate_features(
            fight['f_1_name'], fight['f_2_name'])

        print(f"\n{'Feature':<30} {'Original':>12} {'New':>12} {'Diff':>12}")
        print("-" * 66)

        diffs = []
        for name, orig, new in zip(names, original, new_features):
            diff = abs(orig - new)
            if diff > 0.01:
                diffs.append((name, orig, new, diff))

        for name, orig, new, diff in sorted(diffs, key=lambda x: -x[3])[:20]:
            print(f"{name:<30} {orig:>12.4f} {new:>12.4f} {diff:>12.4f}")

        print(f"\nFeatures with diff > 0.01: {len(diffs)} / {len(names)}")

        if len(diffs) < 10:
            print("\n✓ Feature generation is closely matching!")

    elif args.fighter1 and args.fighter2:
        features, names = gen.generate_features(args.fighter1, args.fighter2)
        print(
            f"\nGenerated {len(features)} features for {args.fighter1} vs {args.fighter2}")


if __name__ == "__main__":
    main()
