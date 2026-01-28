"""
FightPredict - Feature Engineering Pipeline
============================================
Creates advanced features for UFC fight prediction.

Features include:
- Physical advantages (height, reach, age)
- Career statistics (win rate, finish rate, streak)
- Fighting style profiles (striker vs grappler)
- Historical performance metrics
- Momentum and form indicators
- Matchup-specific features

Usage:
    python src/feature_engineering.py

Output:
    - data/processed/features_engineered.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from datetime import datetime


class FeatureEngineer:
    """Creates advanced features for UFC fight prediction"""

    def __init__(self, processed_data_path: str = "data/processed"):
        self.processed_path = Path(processed_data_path)
        self.df = None
        self.feature_names = []

    def load_data(self) -> 'FeatureEngineer':
        """Load cleaned fight data"""
        filepath = self.processed_path / "fights_cleaned.csv"
        self.df = pd.read_csv(filepath, parse_dates=['event_date'])
        print(f"Loaded {len(self.df):,} fights")
        return self

    def engineer_all_features(self) -> pd.DataFrame:
        """Run complete feature engineering pipeline"""
        print("\n" + "=" * 60)
        print("Feature Engineering Pipeline")
        print("=" * 60)

        df = self.df.copy()

        # Sort by date for time-based features
        df = df.sort_values('event_date').reset_index(drop=True)

        # 1. Physical Features
        print("\n[1/8] Physical features...")
        df = self._create_physical_features(df)

        # 2. Career Record Features
        print("[2/8] Career record features...")
        df = self._create_career_features(df)

        # 3. Fighting Style Features
        print("[3/8] Fighting style features...")
        df = self._create_style_features(df)

        # 4. Finish Rate Features
        print("[4/8] Finish rate features...")
        df = self._create_finish_features(df)

        # Defragment DataFrame to avoid performance warnings
        df = df.copy()

        # 5. Experience Features
        print("[5/8] Experience features...")
        df = self._create_experience_features(df)

        # 6. Momentum Features (streak, layoff)
        print("[6/8] Momentum features...")
        df = self._create_momentum_features(df)

        # Defragment again
        df = df.copy()

        # 7. Matchup Features
        print("[7/8] Matchup features...")
        df = self._create_matchup_features(df)

        # 8. Betting Features
        print("[8/8] Betting features...")
        df = self._create_betting_features(df)

        # Final defragmentation
        df = df.copy()

        self.df = df
        return df

    def _create_physical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Physical attribute features"""

        # Raw physical stats
        df['f_1_height'] = df['f_1_fighter_height_cm']
        df['f_2_height'] = df['f_2_fighter_height_cm']
        df['f_1_reach'] = df['f_1_fighter_reach_cm']
        df['f_2_reach'] = df['f_2_fighter_reach_cm']

        # Physical advantages (positive = fighter 1 advantage)
        df['height_diff'] = df['f_1_height'] - df['f_2_height']
        df['reach_diff'] = df['f_1_reach'] - df['f_2_reach']
        df['age_diff'] = df['f_2_age'] - df['f_1_age']  # Younger is advantage

        # Reach-to-height ratio (ape index) - higher is better
        df['f_1_ape_index'] = df['f_1_reach'] / (df['f_1_height'] + 1e-6)
        df['f_2_ape_index'] = df['f_2_reach'] / (df['f_2_height'] + 1e-6)
        df['ape_index_diff'] = df['f_1_ape_index'] - df['f_2_ape_index']

        # Age categories (prime = 27-32)
        df['f_1_in_prime'] = ((df['f_1_age'] >= 27) & (
            df['f_1_age'] <= 32)).astype(int)
        df['f_2_in_prime'] = ((df['f_2_age'] >= 27) & (
            df['f_2_age'] <= 32)).astype(int)
        df['prime_advantage'] = df['f_1_in_prime'] - df['f_2_in_prime']

        # Age risk (too young or too old)
        df['f_1_age_risk'] = ((df['f_1_age'] < 23) | (
            df['f_1_age'] > 37)).astype(int)
        df['f_2_age_risk'] = ((df['f_2_age'] < 23) | (
            df['f_2_age'] > 37)).astype(int)

        return df

    def _create_career_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Career record features"""

        # Total fights
        df['f_1_total_fights'] = df['f_1_fighter_w'] + \
            df['f_1_fighter_l'] + df['f_1_fighter_d']
        df['f_2_total_fights'] = df['f_2_fighter_w'] + \
            df['f_2_fighter_l'] + df['f_2_fighter_d']

        # Win rate
        df['f_1_win_rate'] = df['f_1_fighter_w'] / \
            (df['f_1_total_fights'] + 1e-6)
        df['f_2_win_rate'] = df['f_2_fighter_w'] / \
            (df['f_2_total_fights'] + 1e-6)
        df['win_rate_diff'] = df['f_1_win_rate'] - df['f_2_win_rate']

        # Loss rate
        df['f_1_loss_rate'] = df['f_1_fighter_l'] / \
            (df['f_1_total_fights'] + 1e-6)
        df['f_2_loss_rate'] = df['f_2_fighter_l'] / \
            (df['f_2_total_fights'] + 1e-6)
        df['loss_rate_diff'] = df['f_2_loss_rate'] - \
            df['f_1_loss_rate']  # Less losses is better

        # Experience difference
        df['exp_diff'] = df['f_1_total_fights'] - df['f_2_total_fights']

        # Experience level categories
        df['f_1_exp_level'] = pd.cut(df['f_1_total_fights'],
                                     bins=[0, 5, 15, 30, 100],
                                     labels=[0, 1, 2, 3]).astype(float)
        df['f_2_exp_level'] = pd.cut(df['f_2_total_fights'],
                                     bins=[0, 5, 15, 30, 100],
                                     labels=[0, 1, 2, 3]).astype(float)

        # Record quality score (wins - losses normalized)
        df['f_1_record_score'] = (
            df['f_1_fighter_w'] - df['f_1_fighter_l']) / (df['f_1_total_fights'] + 1e-6)
        df['f_2_record_score'] = (
            df['f_2_fighter_w'] - df['f_2_fighter_l']) / (df['f_2_total_fights'] + 1e-6)
        df['record_score_diff'] = df['f_1_record_score'] - df['f_2_record_score']

        return df

    def _create_style_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fighting style and technique features"""

        # === Striking Metrics ===
        # Strikes landed per minute
        df['f_1_slpm'] = df['f_1_fighter_SlpM']
        df['f_2_slpm'] = df['f_2_fighter_SlpM']
        df['slpm_diff'] = df['f_1_slpm'] - df['f_2_slpm']

        # Strike accuracy
        df['f_1_str_acc'] = df['f_1_fighter_Str_Acc']
        df['f_2_str_acc'] = df['f_2_fighter_Str_Acc']
        df['str_acc_diff'] = df['f_1_str_acc'] - df['f_2_str_acc']

        # Strikes absorbed per minute (less is better)
        df['f_1_sapm'] = df['f_1_fighter_SApM']
        df['f_2_sapm'] = df['f_2_fighter_SApM']
        df['sapm_diff'] = df['f_2_sapm'] - \
            df['f_1_sapm']  # Absorbing less is better

        # Strike defense
        df['f_1_str_def'] = df['f_1_fighter_Str_Def']
        df['f_2_str_def'] = df['f_2_fighter_Str_Def']
        df['str_def_diff'] = df['f_1_str_def'] - df['f_2_str_def']

        # Striking differential (landed - absorbed)
        df['f_1_str_differential'] = df['f_1_slpm'] - df['f_1_sapm']
        df['f_2_str_differential'] = df['f_2_slpm'] - df['f_2_sapm']
        df['str_differential_diff'] = df['f_1_str_differential'] - \
            df['f_2_str_differential']

        # === Grappling Metrics ===
        # Takedown average
        df['f_1_td_avg'] = df['f_1_fighter_TD_Avg']
        df['f_2_td_avg'] = df['f_2_fighter_TD_Avg']
        df['td_avg_diff'] = df['f_1_td_avg'] - df['f_2_td_avg']

        # Takedown accuracy
        df['f_1_td_acc'] = df['f_1_fighter_TD_Acc']
        df['f_2_td_acc'] = df['f_2_fighter_TD_Acc']
        df['td_acc_diff'] = df['f_1_td_acc'] - df['f_2_td_acc']

        # Takedown defense
        df['f_1_td_def'] = df['f_1_fighter_TD_Def']
        df['f_2_td_def'] = df['f_2_fighter_TD_Def']
        df['td_def_diff'] = df['f_1_td_def'] - df['f_2_td_def']

        # Submission average
        df['f_1_sub_avg'] = df['f_1_fighter_Sub_Avg']
        df['f_2_sub_avg'] = df['f_2_fighter_Sub_Avg']
        df['sub_avg_diff'] = df['f_1_sub_avg'] - df['f_2_sub_avg']

        # === Style Classification ===
        # Striker score (high volume + accuracy)
        df['f_1_striker_score'] = df['f_1_slpm'] * df['f_1_str_acc']
        df['f_2_striker_score'] = df['f_2_slpm'] * df['f_2_str_acc']

        # Grappler score (takedowns + submissions)
        df['f_1_grappler_score'] = df['f_1_td_avg'] + df['f_1_sub_avg']
        df['f_2_grappler_score'] = df['f_2_td_avg'] + df['f_2_sub_avg']

        # Style ratio (striker vs grappler, >1 = more striker)
        # Use minimum denominator of 0.1 to prevent explosion, then cap to reasonable range
        df['f_1_style_ratio'] = df['f_1_striker_score'] / \
            np.maximum(df['f_1_grappler_score'], 0.1)
        df['f_2_style_ratio'] = df['f_2_striker_score'] / \
            np.maximum(df['f_2_grappler_score'], 0.1)
        # Cap style ratio to [-20, 20] range (very extreme striker/grappler)
        df['f_1_style_ratio'] = df['f_1_style_ratio'].clip(-20, 20)
        df['f_2_style_ratio'] = df['f_2_style_ratio'].clip(-20, 20)

        # Overall effectiveness score
        df['f_1_effectiveness'] = (df['f_1_str_differential'] +
                                   df['f_1_td_avg'] * 2 +
                                   df['f_1_sub_avg'] * 3)
        df['f_2_effectiveness'] = (df['f_2_str_differential'] +
                                   df['f_2_td_avg'] * 2 +
                                   df['f_2_sub_avg'] * 3)
        df['effectiveness_diff'] = df['f_1_effectiveness'] - \
            df['f_2_effectiveness']

        return df

    def _create_finish_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Finish rate and durability features"""

        # KO power (approximate from striking stats)
        # Higher SlpM with lower fight count suggests finisher
        df['f_1_ko_power'] = df['f_1_slpm'] * (1 - df['f_1_str_def'] + 0.5)
        df['f_2_ko_power'] = df['f_2_slpm'] * (1 - df['f_2_str_def'] + 0.5)
        df['ko_power_diff'] = df['f_1_ko_power'] - df['f_2_ko_power']

        # Submission threat
        df['f_1_sub_threat'] = df['f_1_sub_avg'] * (df['f_1_td_avg'] + 1)
        df['f_2_sub_threat'] = df['f_2_sub_avg'] * (df['f_2_td_avg'] + 1)
        df['sub_threat_diff'] = df['f_1_sub_threat'] - df['f_2_sub_threat']

        # Durability proxy (inverse of strikes absorbed, adjusted for defense)
        # Use minimum denominator of 0.5 to prevent explosion, then cap
        df['f_1_durability'] = df['f_1_str_def'] / \
            np.maximum(df['f_1_sapm'], 0.5)
        df['f_2_durability'] = df['f_2_str_def'] / \
            np.maximum(df['f_2_sapm'], 0.5)
        # Cap durability to reasonable range [0, 5]
        df['f_1_durability'] = df['f_1_durability'].clip(0, 5)
        df['f_2_durability'] = df['f_2_durability'].clip(0, 5)
        df['durability_diff'] = df['f_1_durability'] - df['f_2_durability']

        # Finish potential (ability to end fights)
        df['f_1_finish_potential'] = df['f_1_ko_power'] + df['f_1_sub_threat']
        df['f_2_finish_potential'] = df['f_2_ko_power'] + df['f_2_sub_threat']
        df['finish_potential_diff'] = df['f_1_finish_potential'] - \
            df['f_2_finish_potential']

        return df

    def _create_experience_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Experience and octagon time features"""

        # UFC experience (fights in dataset before this one)
        # This requires iterating through fighters' histories
        df['f_1_ufc_fights'] = 0
        df['f_2_ufc_fights'] = 0

        fighter_counts = {}
        for idx, row in df.iterrows():
            f1, f2 = row['f_1_name'], row['f_2_name']

            # Get current counts
            df.at[idx, 'f_1_ufc_fights'] = fighter_counts.get(f1, 0)
            df.at[idx, 'f_2_ufc_fights'] = fighter_counts.get(f2, 0)

            # Update counts
            fighter_counts[f1] = fighter_counts.get(f1, 0) + 1
            fighter_counts[f2] = fighter_counts.get(f2, 0) + 1

        df['ufc_exp_diff'] = df['f_1_ufc_fights'] - df['f_2_ufc_fights']

        # Debut fight indicator
        df['f_1_is_debut'] = (df['f_1_ufc_fights'] == 0).astype(int)
        df['f_2_is_debut'] = (df['f_2_ufc_fights'] == 0).astype(int)
        df['debut_matchup'] = df['f_1_is_debut'] + \
            df['f_2_is_debut']  # 0, 1, or 2

        # Veteran indicator (10+ UFC fights)
        df['f_1_is_veteran'] = (df['f_1_ufc_fights'] >= 10).astype(int)
        df['f_2_is_veteran'] = (df['f_2_ufc_fights'] >= 10).astype(int)
        df['veteran_vs_rookie'] = df['f_1_is_veteran'] - df['f_2_is_veteran']

        return df

    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum, streak, and recent form features"""

        # Track win/loss streaks
        fighter_streaks = {}
        fighter_last_fight = {}

        df['f_1_streak'] = 0
        df['f_2_streak'] = 0
        df['f_1_days_since_fight'] = np.nan
        df['f_2_days_since_fight'] = np.nan

        for idx, row in df.iterrows():
            f1, f2 = row['f_1_name'], row['f_2_name']
            fight_date = row['event_date']
            winner = row['winner']

            # Get current streaks
            df.at[idx, 'f_1_streak'] = fighter_streaks.get(f1, 0)
            df.at[idx, 'f_2_streak'] = fighter_streaks.get(f2, 0)

            # Days since last fight
            if f1 in fighter_last_fight:
                df.at[idx, 'f_1_days_since_fight'] = (
                    fight_date - fighter_last_fight[f1]).days
            if f2 in fighter_last_fight:
                df.at[idx, 'f_2_days_since_fight'] = (
                    fight_date - fighter_last_fight[f2]).days

            # Update streaks based on result
            if winner == f1:
                fighter_streaks[f1] = max(0, fighter_streaks.get(f1, 0)) + 1
                fighter_streaks[f2] = min(0, fighter_streaks.get(f2, 0)) - 1
            elif winner == f2:
                fighter_streaks[f1] = min(0, fighter_streaks.get(f1, 0)) - 1
                fighter_streaks[f2] = max(0, fighter_streaks.get(f2, 0)) + 1

            # Update last fight dates
            fighter_last_fight[f1] = fight_date
            fighter_last_fight[f2] = fight_date

        # Streak difference
        df['streak_diff'] = df['f_1_streak'] - df['f_2_streak']

        # Win streak indicator
        df['f_1_on_win_streak'] = (df['f_1_streak'] >= 2).astype(int)
        df['f_2_on_win_streak'] = (df['f_2_streak'] >= 2).astype(int)

        # Losing streak indicator
        df['f_1_on_lose_streak'] = (df['f_1_streak'] <= -2).astype(int)
        df['f_2_on_lose_streak'] = (df['f_2_streak'] <= -2).astype(int)

        # Activity (days since fight, fill missing with median)
        median_layoff = 180  # ~6 months default
        df['f_1_days_since_fight'] = df['f_1_days_since_fight'].fillna(
            median_layoff)
        df['f_2_days_since_fight'] = df['f_2_days_since_fight'].fillna(
            median_layoff)

        # Cap days_since_fight at ~2.7 years (beyond this, exact days don't matter)
        CAP_DAYS = 1000
        df['f_1_days_since_fight'] = df['f_1_days_since_fight'].clip(
            upper=CAP_DAYS)
        df['f_2_days_since_fight'] = df['f_2_days_since_fight'].clip(
            upper=CAP_DAYS)

        # Ring rust indicator (>365 days)
        df['f_1_ring_rust'] = (df['f_1_days_since_fight'] > 365).astype(int)
        df['f_2_ring_rust'] = (df['f_2_days_since_fight'] > 365).astype(int)

        # Activity advantage (more recent = better)
        df['activity_diff'] = df['f_2_days_since_fight'] - \
            df['f_1_days_since_fight']

        return df

    def _create_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Matchup-specific features"""

        # Stance matchup
        # Orthodox vs Southpaw historically interesting
        df['stance_clash'] = (df['f_1_stance_encoded'] !=
                              df['f_2_stance_encoded']).astype(int)

        # Style clash (striker vs grappler)
        # Both style_ratios are already capped, so difference is bounded
        df['style_clash'] = np.abs(
            df['f_1_style_ratio'] - df['f_2_style_ratio'])
        # Cap style_clash to reasonable range (max difference = 40 given each ratio capped at [-20, 20])
        df['style_clash'] = df['style_clash'].clip(0, 40)

        # Grappler vs striker matchup
        df['f_1_grappler_vs_f_2_striker'] = ((df['f_1_grappler_score'] > df['f_1_striker_score']) &
                                             (df['f_2_striker_score'] > df['f_2_grappler_score'])).astype(int)
        df['f_1_striker_vs_f_2_grappler'] = ((df['f_1_striker_score'] > df['f_1_grappler_score']) &
                                             (df['f_2_grappler_score'] > df['f_2_striker_score'])).astype(int)

        # Takedown battle prediction
        # High TD offense vs high TD defense
        df['td_battle'] = df['f_1_td_avg'] * \
            (1 - df['f_2_td_def']) - df['f_2_td_avg'] * (1 - df['f_1_td_def'])

        # Striking battle prediction
        df['striking_battle'] = (df['f_1_slpm'] * df['f_1_str_acc'] * (1 - df['f_2_str_def']) -
                                 df['f_2_slpm'] * df['f_2_str_acc'] * (1 - df['f_1_str_def']))

        # Overall matchup score
        df['matchup_score'] = df['striking_battle'] + \
            df['td_battle'] * 2 + df['sub_avg_diff'] * 3

        return df

    def _create_betting_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Betting odds features"""

        # Raw odds
        df['f_1_odds'] = df['f_1_odds']
        df['f_2_odds'] = df['f_2_odds']

        # Odds difference
        df['odds_diff'] = df['f_1_odds'] - df['f_2_odds']

        # Favorite indicator
        df['f_1_is_favorite'] = (df['f_1_odds'] < df['f_2_odds']).astype(float)
        df['f_1_is_favorite'] = df['f_1_is_favorite'].where(
            df['f_1_odds'].notna(), np.nan)

        # Implied probability from odds
        df['f_1_implied_prob'] = 1 / (df['f_1_odds'] + 1e-6)
        df['f_2_implied_prob'] = 1 / (df['f_2_odds'] + 1e-6)

        # Normalize implied probabilities
        total_prob = df['f_1_implied_prob'] + df['f_2_implied_prob']
        df['f_1_implied_prob_norm'] = df['f_1_implied_prob'] / \
            (total_prob + 1e-6)
        df['f_2_implied_prob_norm'] = df['f_2_implied_prob'] / \
            (total_prob + 1e-6)

        # Odds-based value indicators
        df['odds_value_diff'] = df['f_1_implied_prob_norm'] - \
            df['f_2_implied_prob_norm']

        # Heavy favorite/underdog flags
        df['f_1_heavy_favorite'] = (df['f_1_odds'] < 1.4).astype(float)
        df['f_1_heavy_favorite'] = df['f_1_heavy_favorite'].where(
            df['f_1_odds'].notna(), np.nan)
        df['f_1_heavy_underdog'] = (df['f_1_odds'] > 3.0).astype(float)
        df['f_1_heavy_underdog'] = df['f_1_heavy_underdog'].where(
            df['f_1_odds'].notna(), np.nan)

        return df

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Return organized feature groups"""
        return {
            'physical': [
                'height_diff', 'reach_diff', 'age_diff', 'ape_index_diff',
                'f_1_in_prime', 'f_2_in_prime', 'prime_advantage',
                'f_1_age_risk', 'f_2_age_risk',
            ],
            'career': [
                'f_1_win_rate', 'f_2_win_rate', 'win_rate_diff',
                'f_1_loss_rate', 'f_2_loss_rate', 'loss_rate_diff',
                'exp_diff', 'f_1_exp_level', 'f_2_exp_level',
                'f_1_record_score', 'f_2_record_score', 'record_score_diff',
            ],
            'striking': [
                'f_1_slpm', 'f_2_slpm', 'slpm_diff',
                'f_1_str_acc', 'f_2_str_acc', 'str_acc_diff',
                'f_1_sapm', 'f_2_sapm', 'sapm_diff',
                'f_1_str_def', 'f_2_str_def', 'str_def_diff',
                'f_1_str_differential', 'f_2_str_differential', 'str_differential_diff',
            ],
            'grappling': [
                'f_1_td_avg', 'f_2_td_avg', 'td_avg_diff',
                'f_1_td_acc', 'f_2_td_acc', 'td_acc_diff',
                'f_1_td_def', 'f_2_td_def', 'td_def_diff',
                'f_1_sub_avg', 'f_2_sub_avg', 'sub_avg_diff',
            ],
            'style': [
                'f_1_striker_score', 'f_2_striker_score',
                'f_1_grappler_score', 'f_2_grappler_score',
                'f_1_style_ratio', 'f_2_style_ratio',
                'f_1_effectiveness', 'f_2_effectiveness', 'effectiveness_diff',
            ],
            'finish': [
                'f_1_ko_power', 'f_2_ko_power', 'ko_power_diff',
                'f_1_sub_threat', 'f_2_sub_threat', 'sub_threat_diff',
                'f_1_durability', 'f_2_durability', 'durability_diff',
                'f_1_finish_potential', 'f_2_finish_potential', 'finish_potential_diff',
            ],
            'experience': [
                'f_1_ufc_fights', 'f_2_ufc_fights', 'ufc_exp_diff',
                'f_1_is_debut', 'f_2_is_debut', 'debut_matchup',
                'f_1_is_veteran', 'f_2_is_veteran', 'veteran_vs_rookie',
            ],
            'momentum': [
                'f_1_streak', 'f_2_streak', 'streak_diff',
                'f_1_on_win_streak', 'f_2_on_win_streak',
                'f_1_on_lose_streak', 'f_2_on_lose_streak',
                'f_1_days_since_fight', 'f_2_days_since_fight', 'activity_diff',
                'f_1_ring_rust', 'f_2_ring_rust',
            ],
            'matchup': [
                'stance_clash', 'style_clash',
                'f_1_grappler_vs_f_2_striker', 'f_1_striker_vs_f_2_grappler',
                'td_battle', 'striking_battle', 'matchup_score',
            ],
            'betting': [
                'f_1_odds', 'f_2_odds', 'odds_diff', 'f_1_is_favorite',
                'f_1_implied_prob_norm', 'f_2_implied_prob_norm', 'odds_value_diff',
                'f_1_heavy_favorite', 'f_1_heavy_underdog',
            ],
            'categorical': [
                'f_1_stance_encoded', 'f_2_stance_encoded',
                'weight_class_encoded', 'title_fight',
            ],
        }

    def prepare_model_data(self, include_odds: bool = False,
                           feature_groups: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix for model training.

        Args:
            include_odds: Include betting odds features
            feature_groups: List of feature groups to include. 
                          If None, includes all except 'betting' (unless include_odds=True)
        """
        df = self.df.copy()
        all_groups = self.get_feature_groups()

        # Select feature groups
        if feature_groups is None:
            feature_groups = ['physical', 'career', 'striking', 'grappling',
                              'style', 'finish', 'experience', 'momentum',
                              'matchup', 'categorical']
            if include_odds:
                feature_groups.append('betting')

        # Collect features
        features = []
        for group in feature_groups:
            if group in all_groups:
                features.extend(all_groups[group])

        # Remove duplicates while preserving order
        features = list(dict.fromkeys(features))

        # Prepare data
        df_model = df[features + ['target']].copy()

        # Handle missing values
        df_model = df_model.dropna(subset=['target'])

        # Fill remaining NaNs with median for each column
        for col in features:
            if df_model[col].isna().any():
                df_model[col] = df_model[col].fillna(df_model[col].median())

        X = df_model[features]
        y = df_model['target']

        print(
            f"\nFeature matrix: {len(X):,} samples, {len(features)} features")
        print(f"Feature groups: {feature_groups}")
        print(
            f"Class distribution: F1={y.sum():,} ({y.mean()*100:.1f}%), F2={(~y.astype(bool)).sum():,} ({(1-y.mean())*100:.1f}%)")

        self.feature_names = features
        return X, y

    def save_engineered_data(self):
        """Save engineered features"""
        # Save full engineered dataset
        self.df.to_csv(self.processed_path /
                       "features_engineered.csv", index=False)

        # Save model-ready versions
        X, y = self.prepare_model_data(include_odds=False)
        model_df = X.copy()
        model_df['target'] = y.values
        model_df.to_csv(self.processed_path /
                        "features_model_ready.csv", index=False)

        X_odds, y_odds = self.prepare_model_data(include_odds=True)
        model_odds_df = X_odds.copy()
        model_odds_df['target'] = y_odds.values
        model_odds_df.to_csv(self.processed_path /
                             "features_model_ready_with_odds.csv", index=False)

        # Save feature names
        feature_groups = self.get_feature_groups()
        with open(self.processed_path / "feature_groups.txt", 'w') as f:
            for group, features in feature_groups.items():
                f.write(f"\n=== {group.upper()} ===\n")
                for feat in features:
                    f.write(f"  {feat}\n")

        print(f"\nSaved engineered features to {self.processed_path}/")
        print(f"  - features_engineered.csv (all features)")
        print(f"  - features_model_ready.csv ({len(X)} samples)")
        print(
            f"  - features_model_ready_with_odds.csv ({len(X_odds)} samples)")
        print(f"  - feature_groups.txt (feature documentation)")


def main():
    """Run feature engineering pipeline"""
    print("=" * 60)
    print("FightPredict - Feature Engineering")
    print("=" * 60)

    engineer = FeatureEngineer(processed_data_path="data/processed")
    engineer.load_data()
    engineer.engineer_all_features()
    engineer.save_engineered_data()

    # Summary
    groups = engineer.get_feature_groups()
    total_features = sum(len(v) for v in groups.values())
    print(f"\n✓ Created {total_features} features across {len(groups)} groups")

    return engineer


if __name__ == "__main__":
    main()
