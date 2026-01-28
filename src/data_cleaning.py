"""
FightPredict - UFC Data Cleaning Pipeline
==========================================
Cleans and preprocesses UFC fight data for model training.

Usage:
    python src/data_cleaning.py

Output:
    - data/processed/fights_cleaned.csv (full cleaned dataset)
    - data/processed/model_ready.csv (features + target for training)
    - data/processed/fighters.csv (unique fighter profiles)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List


class UFCDataCleaner:
    """Handles all data cleaning and preprocessing for UFC dataset"""

    def __init__(self, raw_data_path: str = "data/raw", processed_data_path: str = "data/processed"):
        self.raw_path = Path(raw_data_path)
        self.processed_path = Path(processed_data_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)

        self.df = None
        self.df_cleaned = None
        self.fighters_df = None

    def load_data(self) -> 'UFCDataCleaner':
        """Load raw UFC data"""
        filepath = self.raw_path / "UFC_full_data_silver.csv"
        self.df = pd.read_csv(filepath)
        print(f"Loaded {len(self.df):,} fights from {filepath}")
        return self

    def clean_fights(self) -> pd.DataFrame:
        """Main cleaning pipeline for fight data"""
        df = self.df.copy()

        # 1. Parse dates
        df['event_date'] = pd.to_datetime(df['event_date'])
        df['f_1_fighter_dob'] = pd.to_datetime(
            df['f_1_fighter_dob'], errors='coerce')
        df['f_2_fighter_dob'] = pd.to_datetime(
            df['f_2_fighter_dob'], errors='coerce')

        # 2. Calculate fighter ages at fight time
        df['f_1_age'] = (df['event_date'] -
                         df['f_1_fighter_dob']).dt.days / 365.25
        df['f_2_age'] = (df['event_date'] -
                         df['f_2_fighter_dob']).dt.days / 365.25

        # 3. Create target variable (1 = fighter_1 wins, 0 = fighter_2 wins)
        df['target'] = (df['winner'] == df['f_1_name']).astype(int)

        # 4. Handle draws and no contests (remove for binary classification)
        # Check if winner matches either fighter
        valid_outcomes = (df['winner'] == df['f_1_name']) | (
            df['winner'] == df['f_2_name'])
        df = df[valid_outcomes].copy()
        print(
            f"Removed {(~valid_outcomes).sum()} draws/NC, {len(df):,} fights remaining")

        # 5. Encode categorical variables
        df['title_fight'] = df['title_fight'].astype(int)

        # Encode stance (Orthodox=0, Southpaw=1, Switch=2)
        stance_map = {'Orthodox': 0, 'Southpaw': 1,
                      'Switch': 2, 'Open Stance': 2}
        df['f_1_stance_encoded'] = df['f_1_fighter_stance'].map(
            stance_map).fillna(0)
        df['f_2_stance_encoded'] = df['f_2_fighter_stance'].map(
            stance_map).fillna(0)

        # Encode weight class
        weight_classes = df['weight_class'].unique()
        weight_map = {wc: i for i, wc in enumerate(
            sorted(weight_classes, key=lambda x: str(x)))}
        df['weight_class_encoded'] = df['weight_class'].map(
            weight_map).fillna(0)

        # Encode result type (for multi-class prediction later)
        df['result_type'] = df['result'].apply(self._categorize_result)
        result_map = {'KO/TKO': 0, 'Submission': 1, 'Decision': 2, 'Other': 3}
        df['result_type_encoded'] = df['result_type'].map(result_map)

        # 6. Compute derived features
        df = self._compute_derived_features(df)

        # 7. Clean numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        print(f"Cleaned dataset: {len(df):,} fights")
        self.df_cleaned = df
        return df

    def _categorize_result(self, result: str) -> str:
        """Categorize fight result into main types"""
        if pd.isna(result):
            return 'Other'
        result = str(result).upper()
        if 'KO' in result or 'TKO' in result:
            return 'KO/TKO'
        elif 'SUB' in result:
            return 'Submission'
        elif 'DEC' in result:
            return 'Decision'
        else:
            return 'Other'

    def _compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute advantage and derived features"""

        # Physical advantages (fighter_1 - fighter_2)
        df['height_adv'] = df['f_1_fighter_height_cm'] - \
            df['f_2_fighter_height_cm']
        df['reach_adv'] = df['f_1_fighter_reach_cm'] - \
            df['f_2_fighter_reach_cm']
        df['age_adv'] = df['f_2_age'] - df['f_1_age']  # Younger is advantage

        # Record advantages
        df['wins_adv'] = df['f_1_fighter_w'] - df['f_2_fighter_w']
        df['losses_adv'] = df['f_2_fighter_l'] - \
            df['f_1_fighter_l']  # Fewer losses is better

        # Win rates
        df['f_1_total_fights'] = df['f_1_fighter_w'] + \
            df['f_1_fighter_l'] + df['f_1_fighter_d']
        df['f_2_total_fights'] = df['f_2_fighter_w'] + \
            df['f_2_fighter_l'] + df['f_2_fighter_d']
        df['f_1_win_rate'] = df['f_1_fighter_w'] / \
            (df['f_1_total_fights'] + 1e-6)
        df['f_2_win_rate'] = df['f_2_fighter_w'] / \
            (df['f_2_total_fights'] + 1e-6)
        df['win_rate_adv'] = df['f_1_win_rate'] - df['f_2_win_rate']

        # Experience advantage
        df['exp_adv'] = df['f_1_total_fights'] - df['f_2_total_fights']

        # Striking advantages
        df['slpm_adv'] = df['f_1_fighter_SlpM'] - \
            df['f_2_fighter_SlpM']  # Strikes landed per minute
        df['str_acc_adv'] = df['f_1_fighter_Str_Acc'] - \
            df['f_2_fighter_Str_Acc']  # Strike accuracy
        df['str_def_adv'] = df['f_1_fighter_Str_Def'] - \
            df['f_2_fighter_Str_Def']  # Strike defense
        df['sapm_adv'] = df['f_2_fighter_SApM'] - \
            df['f_1_fighter_SApM']  # Strikes absorbed (less is better)

        # Grappling advantages
        df['td_avg_adv'] = df['f_1_fighter_TD_Avg'] - \
            df['f_2_fighter_TD_Avg']  # Takedown average
        df['td_acc_adv'] = df['f_1_fighter_TD_Acc'] - \
            df['f_2_fighter_TD_Acc']  # Takedown accuracy
        df['td_def_adv'] = df['f_1_fighter_TD_Def'] - \
            df['f_2_fighter_TD_Def']  # Takedown defense
        df['sub_avg_adv'] = df['f_1_fighter_Sub_Avg'] - \
            df['f_2_fighter_Sub_Avg']  # Submission average

        # Betting odds features (if available)
        df['odds_diff'] = df['f_1_odds'] - df['f_2_odds']
        df['f_1_is_favorite'] = (df['f_1_odds'] < df['f_2_odds']).astype(int)

        return df

    def get_feature_columns(self) -> Dict[str, List[str]]:
        """Return feature column groups for modeling"""
        return {
            'physical': [
                'f_1_fighter_height_cm', 'f_2_fighter_height_cm',
                'f_1_fighter_reach_cm', 'f_2_fighter_reach_cm',
                'f_1_age', 'f_2_age',
            ],
            'record': [
                'f_1_fighter_w', 'f_1_fighter_l', 'f_1_fighter_d',
                'f_2_fighter_w', 'f_2_fighter_l', 'f_2_fighter_d',
                'f_1_win_rate', 'f_2_win_rate',
                'f_1_total_fights', 'f_2_total_fights',
            ],
            'striking': [
                'f_1_fighter_SlpM', 'f_2_fighter_SlpM',
                'f_1_fighter_Str_Acc', 'f_2_fighter_Str_Acc',
                'f_1_fighter_SApM', 'f_2_fighter_SApM',
                'f_1_fighter_Str_Def', 'f_2_fighter_Str_Def',
            ],
            'grappling': [
                'f_1_fighter_TD_Avg', 'f_2_fighter_TD_Avg',
                'f_1_fighter_TD_Acc', 'f_2_fighter_TD_Acc',
                'f_1_fighter_TD_Def', 'f_2_fighter_TD_Def',
                'f_1_fighter_Sub_Avg', 'f_2_fighter_Sub_Avg',
            ],
            'advantages': [
                'height_adv', 'reach_adv', 'age_adv',
                'wins_adv', 'losses_adv', 'win_rate_adv', 'exp_adv',
                'slpm_adv', 'str_acc_adv', 'str_def_adv', 'sapm_adv',
                'td_avg_adv', 'td_acc_adv', 'td_def_adv', 'sub_avg_adv',
            ],
            'categorical': [
                'f_1_stance_encoded', 'f_2_stance_encoded',
                'weight_class_encoded', 'title_fight',
            ],
            'odds': [
                'f_1_odds', 'f_2_odds', 'odds_diff', 'f_1_is_favorite',
            ],
        }

    def prepare_model_data(self, include_odds: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix and target for model training.

        Args:
            include_odds: Whether to include betting odds features (reduces sample size)

        Returns:
            X: Feature DataFrame
            y: Target Series
        """
        df = self.df_cleaned.copy()

        feature_cols = self.get_feature_columns()

        # Select features to use
        features = (
            feature_cols['physical'] +
            feature_cols['record'] +
            feature_cols['striking'] +
            feature_cols['grappling'] +
            feature_cols['advantages'] +
            feature_cols['categorical']
        )

        if include_odds:
            features += feature_cols['odds']

        # Drop rows with missing values in feature columns
        df_model = df[features + ['target']].dropna()

        X = df_model[features]
        y = df_model['target']

        print(
            f"\nModel-ready data: {len(X):,} samples, {len(features)} features")
        print(
            f"Class distribution: F1 wins={y.sum():,} ({y.mean()*100:.1f}%), F2 wins={(~y.astype(bool)).sum():,} ({(1-y.mean())*100:.1f}%)")

        return X, y

    def extract_fighters(self) -> pd.DataFrame:
        """Extract unique fighter profiles from fight data"""
        df = self.df_cleaned

        # Extract fighter 1 data
        f1_cols = [c for c in df.columns if c.startswith(
            'f_1_fighter_') or c == 'f_1_name']
        f1_df = df[f1_cols].copy()
        f1_df.columns = [c.replace('f_1_fighter_', '').replace(
            'f_1_', '') for c in f1_df.columns]

        # Extract fighter 2 data
        f2_cols = [c for c in df.columns if c.startswith(
            'f_2_fighter_') or c == 'f_2_name']
        f2_df = df[f2_cols].copy()
        f2_df.columns = [c.replace('f_2_fighter_', '').replace(
            'f_2_', '') for c in f2_df.columns]

        # Combine and dedupe (keep latest record for each fighter)
        fighters = pd.concat([f1_df, f2_df], ignore_index=True)
        fighters = fighters.drop_duplicates(subset=['name'], keep='last')
        fighters = fighters.sort_values('name').reset_index(drop=True)

        print(f"Extracted {len(fighters):,} unique fighters")
        self.fighters_df = fighters
        return fighters

    def save_processed_data(self):
        """Save all processed datasets"""
        # Save cleaned fights
        self.df_cleaned.to_csv(self.processed_path /
                               "fights_cleaned.csv", index=False)

        # Save model-ready data (without odds for max samples)
        X, y = self.prepare_model_data(include_odds=False)
        model_df = X.copy()
        model_df['target'] = y.values
        model_df.to_csv(self.processed_path / "model_ready.csv", index=False)

        # Save model-ready data with odds
        X_odds, y_odds = self.prepare_model_data(include_odds=True)
        model_odds_df = X_odds.copy()
        model_odds_df['target'] = y_odds.values
        model_odds_df.to_csv(self.processed_path /
                             "model_ready_with_odds.csv", index=False)

        # Save fighter profiles
        self.extract_fighters()
        self.fighters_df.to_csv(self.processed_path /
                                "fighters.csv", index=False)

        print(f"\nSaved processed data to {self.processed_path}/")
        print(f"  - fights_cleaned.csv ({len(self.df_cleaned):,} rows)")
        print(f"  - model_ready.csv ({len(model_df):,} rows)")
        print(f"  - model_ready_with_odds.csv ({len(model_odds_df):,} rows)")
        print(f"  - fighters.csv ({len(self.fighters_df):,} rows)")

    def generate_summary(self) -> Dict:
        """Generate summary statistics"""
        df = self.df_cleaned

        summary = {
            'total_fights': len(df),
            'date_range': f"{df['event_date'].min().date()} to {df['event_date'].max().date()}",
            'unique_fighters': len(set(df['f_1_name'].unique()) | set(df['f_2_name'].unique())),
            'f1_win_pct': df['target'].mean() * 100,
            'ko_pct': (df['result_type'] == 'KO/TKO').mean() * 100,
            'sub_pct': (df['result_type'] == 'Submission').mean() * 100,
            'dec_pct': (df['result_type'] == 'Decision').mean() * 100,
            'title_fights': df['title_fight'].sum(),
            'weight_classes': df['weight_class'].nunique(),
        }

        return summary


def main():
    """Run the full cleaning pipeline"""
    print("=" * 60)
    print("FightPredict - UFC Data Cleaning Pipeline")
    print("=" * 60)

    # Initialize and run pipeline
    cleaner = UFCDataCleaner(
        raw_data_path="data/raw",
        processed_data_path="data/processed"
    )

    cleaner.load_data()
    cleaner.clean_fights()
    cleaner.save_processed_data()

    # Print summary
    summary = cleaner.generate_summary()
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value}")

    print("\n✓ Data cleaning complete!")
    return cleaner


if __name__ == "__main__":
    main()
