"""
FightPredict - Unified Fighter Lookup Module
=============================================

Consolidates all fighter search and stats functionality into a single module.
Used by: app.py, rag_analyzer.py, faiss_search.py, generate_matchup_features.py, predict_fight.py

Usage:
    from fighters import FighterDatabase, FighterSnapshot
    
    db = FighterDatabase()
    db.load()
    
    # Search fighters
    matches = db.search("McGregor")
    
    # Get fighter stats
    stats = db.get_snapshot("Conor McGregor")
    print(stats.name, stats.wins, stats.slpm)
    
    # Get all fighter names (for autocomplete)
    names = db.get_all_names()
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class FighterSnapshot:
    """
    Fighter statistics snapshot from their most recent fight.
    Contains all stats needed for feature generation and display.
    """
    name: str

    # Physical
    height_cm: float
    reach_cm: float
    stance: str
    age: float
    dob: Optional[str]

    # Record
    wins: int
    losses: int
    draws: int

    # Striking stats
    slpm: float       # Significant strikes landed per minute
    str_acc: float    # Strike accuracy (0-1)
    sapm: float       # Significant strikes absorbed per minute
    str_def: float    # Strike defense (0-1)

    # Grappling stats
    td_avg: float     # Takedown average per 15 min
    td_acc: float     # Takedown accuracy (0-1)
    td_def: float     # Takedown defense (0-1)
    sub_avg: float    # Submission average per 15 min

    # Temporal
    ufc_fights: int
    current_streak: int
    days_since_last_fight: float
    last_fight_date: Optional[datetime]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'height_cm': self.height_cm,
            'reach_cm': self.reach_cm,
            'stance': self.stance,
            'age': self.age,
            'dob': self.dob,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'slpm': self.slpm,
            'str_acc': self.str_acc,
            'sapm': self.sapm,
            'str_def': self.str_def,
            'td_avg': self.td_avg,
            'td_acc': self.td_acc,
            'td_def': self.td_def,
            'sub_avg': self.sub_avg,
            'ufc_fights': self.ufc_fights,
            'current_streak': self.current_streak,
            'days_since_last_fight': self.days_since_last_fight,
            'last_fight_date': self.last_fight_date
        }

    @property
    def record(self) -> str:
        """Return formatted record string."""
        return f"{self.wins}-{self.losses}-{self.draws}"

    @property
    def total_fights(self) -> int:
        """Total professional fights."""
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        """Win rate as decimal."""
        total = self.total_fights
        return self.wins / total if total > 0 else 0.0


class FighterDatabase:
    """
    Unified fighter database for lookups and stats retrieval.

    Loads data from fights_cleaned.csv and extracts fighter stats
    from their most recent fight (matching feature_engineering.py approach).
    """

    def __init__(
        self,
        fights_path: str = "data/processed/fights_cleaned.csv",
        fighters_path: str = "data/processed/fighters.csv"
    ):
        self.fights_path = Path(fights_path)
        self.fighters_path = Path(fighters_path)

        self.fights_df: Optional[pd.DataFrame] = None
        self.fighters_df: Optional[pd.DataFrame] = None
        self._all_names: Optional[List[str]] = None
        self._cache: Dict[str, FighterSnapshot] = {}

    def load(self) -> bool:
        """
        Load fight and fighter data.

        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            # Load fights data (primary source)
            if self.fights_path.exists():
                self.fights_df = pd.read_csv(
                    self.fights_path,
                    parse_dates=['event_date']
                )
                self.fights_df = self.fights_df.sort_values(
                    'event_date').reset_index(drop=True)

                # Build fighter names list from fights
                f1_names = set(self.fights_df['f_1_name'].unique())
                f2_names = set(self.fights_df['f_2_name'].unique())
                self._all_names = sorted(list(f1_names | f2_names))

                print(
                    f"✓ Loaded {len(self.fights_df):,} fights, {len(self._all_names):,} fighters")
            else:
                print(f"✗ Fights file not found: {self.fights_path}")
                return False

            # Optionally load fighters.csv for additional data
            if self.fighters_path.exists():
                self.fighters_df = pd.read_csv(self.fighters_path)
                print(
                    f"✓ Loaded fighters database: {len(self.fighters_df):,} entries")

            return True

        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False

    def get_all_names(self) -> List[str]:
        """
        Get sorted list of all fighter names (for autocomplete).

        Returns:
            List of fighter names.
        """
        if self._all_names is None:
            if self.fights_df is not None:
                f1_names = set(self.fights_df['f_1_name'].unique())
                f2_names = set(self.fights_df['f_2_name'].unique())
                self._all_names = sorted(list(f1_names | f2_names))
            else:
                self._all_names = []
        return self._all_names

    def search(self, query: str, limit: int = 10) -> List[str]:
        """
        Search for fighters by partial name match (case-insensitive).

        Args:
            query: Search string.
            limit: Maximum results to return.

        Returns:
            List of matching fighter names.
        """
        if not query:
            return []

        all_names = self.get_all_names()
        query_lower = query.lower()

        matches = [name for name in all_names if query_lower in name.lower()]
        return sorted(matches)[:limit]

    def get_snapshot(self, name: str) -> Optional[FighterSnapshot]:
        """
        Get fighter stats snapshot from their most recent fight.

        This matches how feature_engineering.py extracts stats - from
        the f_1_fighter_* or f_2_fighter_* columns at fight time.

        Args:
            name: Fighter name (case-insensitive matching supported).

        Returns:
            FighterSnapshot or None if not found.
        """
        if self.fights_df is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        # Check cache
        cache_key = name.lower()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Find fights for this fighter
        mask_f1 = self.fights_df['f_1_name'] == name
        mask_f2 = self.fights_df['f_2_name'] == name
        fighter_fights = self.fights_df[mask_f1 | mask_f2].copy()

        # Try case-insensitive if no exact match
        if len(fighter_fights) == 0:
            mask_f1 = self.fights_df['f_1_name'].str.lower() == name.lower()
            mask_f2 = self.fights_df['f_2_name'].str.lower() == name.lower()
            fighter_fights = self.fights_df[mask_f1 | mask_f2].copy()

            if len(fighter_fights) == 0:
                return None

            # Get actual name with correct casing
            if mask_f1.any():
                name = self.fights_df.loc[mask_f1, 'f_1_name'].iloc[0]
            else:
                name = self.fights_df.loc[mask_f2, 'f_2_name'].iloc[0]

        # Sort by date descending (most recent first)
        fighter_fights = fighter_fights.sort_values(
            'event_date', ascending=False)

        # Get most recent fight
        latest = fighter_fights.iloc[0]
        is_f1 = (latest['f_1_name'] == name)
        prefix = 'f_1_fighter_' if is_f1 else 'f_2_fighter_'

        # Calculate age from DOB
        dob_str = latest.get(f'{prefix}dob')
        age = self._calculate_age(dob_str)

        # Calculate temporal features
        temporal = self._compute_temporal(name, fighter_fights)

        # Build snapshot
        snapshot = FighterSnapshot(
            name=name,
            height_cm=self._safe_float(
                latest.get(f'{prefix}height_cm'), 175.0),
            reach_cm=self._safe_float(latest.get(f'{prefix}reach_cm'), 180.0),
            stance=str(latest.get(f'{prefix}stance', 'Orthodox')),
            age=age,
            dob=dob_str if pd.notna(dob_str) else None,
            wins=int(self._safe_float(latest.get(f'{prefix}w'), 0)),
            losses=int(self._safe_float(latest.get(f'{prefix}l'), 0)),
            draws=int(self._safe_float(latest.get(f'{prefix}d'), 0)),
            slpm=self._safe_float(latest.get(f'{prefix}SlpM'), 3.0),
            str_acc=self._safe_float(latest.get(f'{prefix}Str_Acc'), 0.45),
            sapm=self._safe_float(latest.get(f'{prefix}SApM'), 3.0),
            str_def=self._safe_float(latest.get(f'{prefix}Str_Def'), 0.55),
            td_avg=self._safe_float(latest.get(f'{prefix}TD_Avg'), 1.0),
            td_acc=self._safe_float(latest.get(f'{prefix}TD_Acc'), 0.40),
            td_def=self._safe_float(latest.get(f'{prefix}TD_Def'), 0.55),
            sub_avg=self._safe_float(latest.get(f'{prefix}Sub_Avg'), 0.5),
            ufc_fights=temporal['ufc_fights'],
            current_streak=temporal['streak'],
            days_since_last_fight=temporal['days_since_fight'],
            last_fight_date=temporal['last_fight_date']
        )

        # Cache it
        self._cache[cache_key] = snapshot
        return snapshot

    def get_fighter_fights(self, name: str) -> List[Dict[str, Any]]:
        """
        Get all fights for a specific fighter.

        Args:
            name: Fighter name.

        Returns:
            List of fight dictionaries, most recent first.
        """
        if self.fights_df is None:
            return []

        mask_f1 = self.fights_df['f_1_name'].str.lower() == name.lower()
        mask_f2 = self.fights_df['f_2_name'].str.lower() == name.lower()

        fights = self.fights_df[mask_f1 | mask_f2].copy()
        fights = fights.sort_values('event_date', ascending=False)

        results = []
        for _, row in fights.iterrows():
            is_f1 = row['f_1_name'].lower() == name.lower()
            opponent = row['f_2_name'] if is_f1 else row['f_1_name']
            won = row['winner'].lower() == name.lower(
            ) if pd.notna(row['winner']) else None

            results.append({
                'event_name': row.get('event_name'),
                'event_date': row.get('event_date'),
                'opponent': opponent,
                'won': won,
                'result': row.get('result'),
                'result_details': row.get('result_details'),
                'weight_class': row.get('weight_class'),
                'title_fight': row.get('title_fight', 0) == 1
            })

        return results

    def _safe_float(self, value, default: float = 0.0) -> float:
        """Safely convert value to float."""
        if pd.isna(value):
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _calculate_age(self, dob_str) -> float:
        """Calculate current age from DOB string."""
        if pd.isna(dob_str):
            return 30.0
        try:
            dob = pd.to_datetime(dob_str)
            return (datetime.now() - dob).days / 365.25
        except:
            return 30.0

    def _compute_temporal(self, name: str, fights: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute temporal features (UFC fights, streak, days since fight).

        Args:
            name: Fighter name.
            fights: DataFrame of fighter's fights (sorted by date desc).

        Returns:
            Dict with temporal features.
        """
        if len(fights) == 0:
            return {
                'ufc_fights': 0,
                'streak': 0,
                'days_since_fight': 180.0,
                'last_fight_date': None
            }

        # UFC fights = total count
        ufc_fights = len(fights)

        # Calculate streak from most recent fights
        streak = 0
        for _, fight in fights.iterrows():
            won = (fight['winner'] == name)

            if streak == 0:
                streak = 1 if won else -1
            elif streak > 0 and won:
                streak += 1
            elif streak < 0 and not won:
                streak -= 1
            else:
                break

        # Days since last fight
        last_fight_date = fights.iloc[0]['event_date']
        if pd.isna(last_fight_date):
            days_since = 180.0
        else:
            days_since = (datetime.now() -
                          pd.to_datetime(last_fight_date)).days
            days_since = float(max(0, min(days_since, 1500)))

        return {
            'ufc_fights': ufc_fights,
            'streak': streak,
            'days_since_fight': days_since,
            'last_fight_date': last_fight_date if pd.notna(last_fight_date) else None
        }

    def clear_cache(self):
        """Clear the fighter snapshot cache."""
        self._cache.clear()


# =============================================================================
# Convenience functions for backward compatibility
# =============================================================================

_default_db: Optional[FighterDatabase] = None


def get_database(
    fights_path: str = "data/processed/fights_cleaned.csv",
    fighters_path: str = "data/processed/fighters.csv"
) -> FighterDatabase:
    """
    Get or create the default fighter database instance.

    This provides a singleton-like pattern for easy access.
    """
    global _default_db
    if _default_db is None:
        _default_db = FighterDatabase(fights_path, fighters_path)
        _default_db.load()
    return _default_db


def search_fighter(query: str, limit: int = 10) -> List[str]:
    """Convenience function to search fighters."""
    return get_database().search(query, limit)


def get_fighter_snapshot(name: str) -> Optional[FighterSnapshot]:
    """Convenience function to get fighter snapshot."""
    return get_database().get_snapshot(name)


def get_fighter_stats(name: str) -> Optional[FighterSnapshot]:
    """Alias for get_fighter_snapshot (backward compatibility)."""
    return get_fighter_snapshot(name)


def get_all_fighter_names() -> List[str]:
    """Convenience function to get all fighter names."""
    return get_database().get_all_names()


# =============================================================================
# CLI for testing
# =============================================================================

def main():
    """Test the fighter database."""
    import argparse

    parser = argparse.ArgumentParser(description="Fighter Database")
    parser.add_argument("--search", "-s", type=str, help="Search for fighters")
    parser.add_argument("--info", "-i", type=str, help="Get fighter info")
    parser.add_argument("--fights", "-f", type=str,
                        help="Get fighter's fight history")

    args = parser.parse_args()

    db = FighterDatabase()
    if not db.load():
        print("Failed to load data")
        return

    if args.search:
        results = db.search(args.search)
        print(f"\nFound {len(results)} fighters matching '{args.search}':")
        for name in results:
            print(f"  • {name}")

    elif args.info:
        stats = db.get_snapshot(args.info)
        if stats:
            print(f"\n{stats.name}")
            print(f"  Record: {stats.record}")
            print(f"  Age: {stats.age:.1f}")
            print(f"  Height: {stats.height_cm:.0f} cm")
            print(f"  Reach: {stats.reach_cm:.0f} cm")
            print(f"  Stance: {stats.stance}")
            print(f"  UFC Fights: {stats.ufc_fights}")
            print(f"  Current Streak: {stats.current_streak:+d}")
            print(
                f"  Striking: {stats.slpm:.2f} SLpM, {stats.str_acc:.1%} acc, {stats.str_def:.1%} def")
            print(
                f"  Grappling: {stats.td_avg:.2f} TD/15min, {stats.td_acc:.1%} acc, {stats.sub_avg:.2f} sub/15min")
        else:
            print(f"Fighter not found: {args.info}")

    elif args.fights:
        fights = db.get_fighter_fights(args.fights)
        if fights:
            print(f"\n{args.fights}'s fight history ({len(fights)} fights):")
            for fight in fights[:10]:
                won_str = "✓ W" if fight['won'] else "✗ L" if fight['won'] is False else "?"
                print(
                    f"  {won_str} vs {fight['opponent']} - {fight['result']} ({fight['event_name']})")
        else:
            print(f"No fights found for: {args.fights}")

    else:
        # Demo
        print(f"\nTotal fighters: {len(db.get_all_names())}")
        print("\nSample search for 'jones':")
        for name in db.search("jones", limit=5):
            print(f"  • {name}")


if __name__ == "__main__":
    main()
