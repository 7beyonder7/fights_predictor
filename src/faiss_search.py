"""
FightPredict - FAISS Similarity Search
=======================================
Find historical fights similar to upcoming matchups using vector similarity.

Features:
- Index all historical fights by feature vectors
- Query: "Find 10 most similar historical fights to Fighter A vs Fighter B"
- Return similar fights with outcomes for context
- Multiple distance metrics (L2, cosine, inner product)
- Feature weighting for domain-specific similarity

Usage:
    from faiss_search import FightSimilaritySearch
    
    # Initialize and build index
    searcher = FightSimilaritySearch()
    searcher.build_index()
    
    # Find similar fights
    similar = searcher.find_similar_fights(feature_vector, k=10)
    
    # Or search by fighter names (if metadata available)
    similar = searcher.search_by_fighters("Conor McGregor", "Dustin Poirier", k=10)

Output:
    - models/faiss_index.bin - FAISS index file
    - models/faiss_metadata.pkl - Fight metadata for display
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import json
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# FAISS import
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not installed. Run: pip install faiss-cpu")


class FightSimilaritySearch:
    """
    FAISS-based similarity search for UFC fights.

    Indexes historical fights by their feature vectors and enables
    fast nearest-neighbor queries to find similar matchups.
    """

    def __init__(
        self,
        data_path: str = "data/processed",
        models_path: str = "models",
        use_cosine: bool = True
    ):
        """
        Initialize the similarity search engine.

        Args:
            data_path: Path to processed data files
            models_path: Path to save/load index files
            use_cosine: Use cosine similarity (True) or L2 distance (False)
        """
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.use_cosine = use_cosine
        self.index = None
        self.scaler = None
        self.feature_names = None
        self.fight_metadata = None
        self.feature_vectors = None
        self.n_features = None

        # Feature groups for weighted similarity
        self.feature_groups = {
            'physical': ['height_diff', 'reach_diff', 'age_diff', 'ape_index_diff'],
            'experience': ['exp_diff', 'f_1_exp_level', 'f_2_exp_level', 'ufc_exp_diff',
                           'f_1_is_debut', 'f_2_is_debut', 'f_1_is_veteran', 'f_2_is_veteran'],
            'striking': ['f_1_slpm', 'f_2_slpm', 'slpm_diff', 'f_1_str_acc', 'f_2_str_acc',
                         'str_acc_diff', 'f_1_sapm', 'f_2_sapm', 'f_1_str_def', 'f_2_str_def',
                         'f_1_striker_score', 'f_2_striker_score', 'striking_battle'],
            'grappling': ['f_1_td_avg', 'f_2_td_avg', 'f_1_td_acc', 'f_2_td_acc',
                          'f_1_td_def', 'f_2_td_def', 'f_1_sub_avg', 'f_2_sub_avg',
                          'f_1_grappler_score', 'f_2_grappler_score', 'td_battle'],
            'record': ['f_1_win_rate', 'f_2_win_rate', 'win_rate_diff',
                       'f_1_record_score', 'f_2_record_score', 'record_score_diff'],
            'style': ['f_1_style_ratio', 'f_2_style_ratio', 'stance_clash', 'style_clash'],
            'momentum': ['f_1_streak', 'f_2_streak', 'streak_diff',
                         'f_1_on_win_streak', 'f_2_on_win_streak']
        }

        # Unified fighter database
        self._fighter_db = None

    def _get_fighter_db(self):
        """Get or create fighter database instance."""
        if self._fighter_db is None:
            try:
                from fighters import FighterDatabase
            except ImportError:
                from src.fighters import FighterDatabase
            self._fighter_db = FighterDatabase(
                fights_path=str(self.data_path / "fights_cleaned.csv"),
                fighters_path=str(self.data_path / "fighters.csv")
            )
            self._fighter_db.load()
        return self._fighter_db

    def load_fighters(self) -> bool:
        """Load fighters database for name lookups."""
        db = self._get_fighter_db()
        return db.fights_df is not None

    def get_fighter_names(self) -> List[str]:
        """Get list of all fighter names for autocomplete."""
        return self._get_fighter_db().get_all_names()

    def search_fighter(self, query: str) -> List[str]:
        """Search for fighters by partial name match."""
        return self._get_fighter_db().search(query)

    def get_fighter_stats(self, fighter_name: str) -> Optional[Dict]:
        """Get stats for a specific fighter."""
        snapshot = self._get_fighter_db().get_snapshot(fighter_name)
        return snapshot.to_dict() if snapshot else None

    def find_fights_by_fighter(self, fighter_name: str) -> List[Dict]:
        """Find all historical fights involving a specific fighter."""
        if self.fight_metadata is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Search in both fighter columns
        mask = (
            (self.fight_metadata['f_1_name'].str.lower() == fighter_name.lower()) |
            (self.fight_metadata['f_2_name'].str.lower()
             == fighter_name.lower())
        )

        matches = self.fight_metadata[mask]

        results = []
        for idx, row in matches.iterrows():
            fight_info = row.to_dict()
            fight_info['index'] = idx
            fight_info['similarity'] = 1.0  # Exact match
            results.append(fight_info)

        return results

    def load_data(self, include_odds: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load feature data and fight metadata.

        Args:
            include_odds: Whether to use the dataset with betting odds

        Returns:
            (features_df, metadata_df)
        """
        # Load features
        if include_odds:
            features_path = self.data_path / "features_model_ready_with_odds.csv"
        else:
            features_path = self.data_path / "features_model_ready.csv"

        features_df = pd.read_csv(features_path)

        # Try to load fight metadata (with fighter names, dates, etc.)
        metadata_path = self.data_path / "fights_cleaned.csv"
        if metadata_path.exists():
            metadata_df = pd.read_csv(metadata_path)

            # Select key columns for display
            key_columns = [
                'event_name', 'event_date', 'weight_class', 'title_fight',
                'f_1_name', 'f_2_name', 'winner', 'result', 'result_details',
                'finish_round', 'finish_time', 'referee',
                'f_1_fighter_stance', 'f_2_fighter_stance',
                'f_1_fighter_height_cm', 'f_2_fighter_height_cm',
                'f_1_fighter_reach_cm', 'f_2_fighter_reach_cm',
                'f_1_fighter_w', 'f_1_fighter_l', 'f_2_fighter_w', 'f_2_fighter_l'
            ]

            # Only keep columns that exist
            existing_columns = [
                c for c in key_columns if c in metadata_df.columns]
            metadata_df = metadata_df[existing_columns].copy()

            print(f"✓ Loaded fight metadata: {len(metadata_df):,} fights")
        else:
            # Create basic metadata from index if no detailed file exists
            print("⚠️ Fight metadata file not found. Using basic indexing.")
            metadata_df = pd.DataFrame({
                'fight_id': range(len(features_df)),
            })

        # Add target from features if available
        if 'target' in features_df.columns:
            metadata_df['target'] = features_df['target'].values

        print(
            f"✓ Loaded features: {len(features_df):,} fights, {len(features_df.columns)-1} features")

        return features_df, metadata_df

    def build_index(
        self,
        include_odds: bool = False,
        feature_weights: Optional[Dict[str, float]] = None,
        index_type: str = "flat"
    ) -> None:
        """
        Build FAISS index from fight feature vectors.

        Args:
            include_odds: Whether to include betting odds features
            feature_weights: Optional dict of feature group weights
            index_type: "flat" (exact) or "ivf" (approximate, faster for large datasets)
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is not installed. Run: pip install faiss-cpu")

        print("\n" + "=" * 60)
        print("Building FAISS Similarity Index")
        print("=" * 60)

        # Load data
        features_df, metadata_df = self.load_data(include_odds)

        # Separate features and target
        if 'target' in features_df.columns:
            X = features_df.drop('target', axis=1)
            targets = features_df['target'].values
        else:
            X = features_df
            targets = None

        self.feature_names = list(X.columns)
        self.n_features = len(self.feature_names)

        # Store metadata
        self.fight_metadata = metadata_df.copy()
        if targets is not None:
            self.fight_metadata['target'] = targets

        # Standardize features (important for distance calculations)
        print("\nStandardizing features...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X.values).astype(np.float32)

        # Ensure array is C-contiguous (required by FAISS)
        X_scaled = np.ascontiguousarray(X_scaled)

        # Apply feature weights if provided
        if feature_weights:
            X_scaled = self._apply_feature_weights(X_scaled, feature_weights)
            X_scaled = np.ascontiguousarray(X_scaled)

        # Normalize for cosine similarity
        if self.use_cosine:
            print("Normalizing vectors for cosine similarity...")
            faiss.normalize_L2(X_scaled)

        self.feature_vectors = X_scaled

        # Build FAISS index
        print(f"\nBuilding {index_type.upper()} index...")

        if index_type == "flat":
            # Exact search - best for datasets < 100k
            if self.use_cosine:
                self.index = faiss.IndexFlatIP(
                    self.n_features)  # Inner product for cosine
            else:
                self.index = faiss.IndexFlatL2(self.n_features)  # L2 distance

        elif index_type == "ivf":
            # Approximate search - faster for large datasets
            n_clusters = min(100, len(X_scaled) // 40)  # Rule of thumb
            quantizer = faiss.IndexFlatL2(self.n_features)

            if self.use_cosine:
                self.index = faiss.IndexIVFFlat(quantizer, self.n_features, n_clusters,
                                                faiss.METRIC_INNER_PRODUCT)
            else:
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.n_features, n_clusters)

            print(f"   Training IVF index with {n_clusters} clusters...")
            self.index.train(X_scaled)
            self.index.nprobe = 10  # Number of clusters to search

        # Add vectors to index
        self.index.add(X_scaled)

        print(f"\n✓ Index built successfully!")
        print(f"   Vectors indexed: {self.index.ntotal:,}")
        print(f"   Dimensions: {self.n_features}")
        print(
            f"   Similarity metric: {'Cosine' if self.use_cosine else 'L2 Distance'}")

    def _apply_feature_weights(
        self,
        X: np.ndarray,
        weights: Dict[str, float]
    ) -> np.ndarray:
        """Apply feature group weights to emphasize certain aspects."""
        X_weighted = X.copy()

        for group_name, weight in weights.items():
            if group_name in self.feature_groups:
                group_features = self.feature_groups[group_name]
                for feature in group_features:
                    if feature in self.feature_names:
                        idx = self.feature_names.index(feature)
                        X_weighted[:, idx] *= weight

        return X_weighted

    def find_similar_fights(
        self,
        query_features: Union[np.ndarray, pd.Series, Dict],
        k: int = 10,
        exclude_indices: Optional[List[int]] = None,
        min_similarity: float = 0.0
    ) -> List[Dict]:
        """
        Find k most similar historical fights to a query feature vector.

        Args:
            query_features: Feature vector for the query fight
            k: Number of similar fights to return
            exclude_indices: Fight indices to exclude from results
            min_similarity: Minimum similarity score to include

        Returns:
            List of dicts with fight info and similarity scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Convert query to numpy array
        if isinstance(query_features, dict):
            query = np.array([query_features.get(f, 0)
                             for f in self.feature_names])
        elif isinstance(query_features, pd.Series):
            query = query_features[self.feature_names].values
        else:
            query = np.array(query_features)

        # Ensure correct shape and type
        query = query.reshape(1, -1).astype(np.float32)

        # Scale query
        query_scaled = self.scaler.transform(query).astype(np.float32)

        # Ensure C-contiguous
        query_scaled = np.ascontiguousarray(query_scaled)

        # Normalize for cosine similarity
        if self.use_cosine:
            faiss.normalize_L2(query_scaled)

        # Search for more results to account for exclusions
        search_k = k + (len(exclude_indices) if exclude_indices else 0) + 5

        # Perform search
        distances, indices = self.index.search(query_scaled, search_k)

        # Process results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue

            if exclude_indices and idx in exclude_indices:
                continue

            # Convert distance to similarity score
            if self.use_cosine:
                similarity = float(dist)  # Inner product is already similarity
            else:
                # Convert L2 distance to similarity
                similarity = 1 / (1 + float(dist))

            if similarity < min_similarity:
                continue

            # Get fight metadata
            fight_info = self._get_fight_info(idx, similarity)
            results.append(fight_info)

            if len(results) >= k:
                break

        return results

    def _get_fight_info(self, idx: int, similarity: float) -> Dict:
        """Get detailed information about a fight by index."""
        info = {
            'index': int(idx),
            'similarity': round(similarity, 4)
        }

        # Add metadata if available
        if self.fight_metadata is not None and idx < len(self.fight_metadata):
            row = self.fight_metadata.iloc[idx]

            # Add all metadata columns
            for col in self.fight_metadata.columns:
                val = row[col]
                # Convert numpy types to Python types
                if pd.isna(val):
                    info[col] = None
                elif isinstance(val, (np.integer, np.int64)):
                    info[col] = int(val)
                elif isinstance(val, (np.floating, np.float64)):
                    info[col] = float(val)
                else:
                    info[col] = val

            # Add outcome interpretation - use actual winner name if available
            if 'target' in info:
                if info.get('winner') and info['winner'] not in ['Fighter 1', 'Fighter 2', None]:
                    # Already have winner name from metadata
                    pass
                else:
                    # Determine winner from target and fighter names
                    f1_name = info.get('f_1_name', 'Fighter 1')
                    f2_name = info.get('f_2_name', 'Fighter 2')
                    info['winner'] = f1_name if info['target'] == 1 else f2_name

        return info

    def search_by_fighters(
        self,
        fighter_1: str,
        fighter_2: str,
        k: int = 10,
        feature_vector: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Find similar fights for a matchup between two fighters.

        If feature_vector is provided, uses that for similarity search.
        Otherwise, requires a method to compute features from fighter names.

        Args:
            fighter_1: Name of first fighter
            fighter_2: Name of second fighter
            k: Number of similar fights to return
            feature_vector: Pre-computed feature vector for this matchup

        Returns:
            List of similar fights with outcomes
        """
        if feature_vector is None:
            raise ValueError(
                "Feature vector required. Use feature_engineering to compute features "
                "for the fighter matchup first, then pass the vector here."
            )

        print(f"\n🔍 Finding fights similar to: {fighter_1} vs {fighter_2}")

        results = self.find_similar_fights(feature_vector, k=k)

        return results

    def find_similar_by_index(self, fight_index: int, k: int = 10) -> List[Dict]:
        """
        Find fights similar to a specific historical fight by its index.

        Args:
            fight_index: Index of the fight in the dataset
            k: Number of similar fights to return

        Returns:
            List of similar fights (excluding the query fight itself)
        """
        if self.feature_vectors is None:
            raise ValueError("Index not built. Call build_index() first.")

        query_vector = self.feature_vectors[fight_index]

        # Exclude the query fight itself
        results = self.find_similar_fights(
            query_vector,
            k=k,
            exclude_indices=[fight_index]
        )

        return results

    def analyze_similar_fights(
        self,
        query_features: np.ndarray,
        k: int = 10
    ) -> Dict:
        """
        Analyze similar fights and provide statistical insights.

        Args:
            query_features: Feature vector for query fight
            k: Number of similar fights to analyze

        Returns:
            Dict with analysis results
        """
        similar_fights = self.find_similar_fights(query_features, k=k)

        if not similar_fights:
            return {'error': 'No similar fights found'}

        # Analyze outcomes
        outcomes = [f.get('target', f.get('winner')) for f in similar_fights]
        outcomes = [o for o in outcomes if o is not None]

        if not outcomes:
            return {
                'similar_fights': similar_fights,
                'analysis': 'No outcome data available'
            }

        # Calculate statistics
        if all(isinstance(o, (int, float)) for o in outcomes):
            # Numeric targets (0/1)
            f1_wins = sum(1 for o in outcomes if o == 1)
            f2_wins = sum(1 for o in outcomes if o == 0)
        else:
            # String outcomes
            f1_wins = sum(1 for o in outcomes if 'Fighter 1' in str(o))
            f2_wins = len(outcomes) - f1_wins

        total = len(outcomes)

        # Weighted by similarity
        weights = [f['similarity'] for f in similar_fights[:len(outcomes)]]
        if sum(weights) > 0:
            weighted_f1 = sum(w * (1 if o == 1 else 0)
                              for w, o in zip(weights, outcomes)) / sum(weights)
        else:
            weighted_f1 = f1_wins / total if total > 0 else 0.5

        analysis = {
            'similar_fights': similar_fights,
            'total_analyzed': total,
            'fighter_1_wins': f1_wins,
            'fighter_2_wins': f2_wins,
            'fighter_1_win_rate': f1_wins / total if total > 0 else 0,
            'fighter_2_win_rate': f2_wins / total if total > 0 else 0,
            'weighted_fighter_1_probability': weighted_f1,
            'weighted_fighter_2_probability': 1 - weighted_f1,
            'average_similarity': np.mean([f['similarity'] for f in similar_fights]),
            'prediction': 'Fighter 1' if weighted_f1 > 0.5 else 'Fighter 2',
            'confidence': abs(weighted_f1 - 0.5) * 2  # 0 to 1 scale
        }

        return analysis

    def save_index(self, filename: str = "faiss_index") -> None:
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save. Build index first.")

        # Save FAISS index
        index_path = self.models_path / f"{filename}.bin"
        faiss.write_index(self.index, str(index_path))
        print(f"✓ Index saved to: {index_path}")

        # Save metadata and scaler
        metadata_path = self.models_path / f"{filename}_metadata.pkl"
        metadata = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'fight_metadata': self.fight_metadata,
            'use_cosine': self.use_cosine,
            'n_features': self.n_features,
            'timestamp': datetime.now().isoformat()
        }

        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ Metadata saved to: {metadata_path}")

    def load_index(self, filename: str = "faiss_index") -> None:
        """Load FAISS index and metadata from disk."""
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is not installed. Run: pip install faiss-cpu")

        # Load FAISS index
        index_path = self.models_path / f"{filename}.bin"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        self.index = faiss.read_index(str(index_path))
        print(f"✓ Index loaded from: {index_path}")

        # Load metadata
        metadata_path = self.models_path / f"{filename}_metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            self.scaler = metadata['scaler']
            self.feature_names = metadata['feature_names']
            self.fight_metadata = metadata['fight_metadata']
            self.use_cosine = metadata['use_cosine']
            self.n_features = metadata['n_features']

            print(
                f"✓ Metadata loaded: {self.index.ntotal:,} fights, {self.n_features} features")
        else:
            print("⚠️ Metadata file not found. Some features may be limited.")

    def get_feature_importance_for_match(
        self,
        query_features: np.ndarray,
        similar_fight_idx: int
    ) -> pd.DataFrame:
        """
        Analyze which features make two fights similar.

        Args:
            query_features: Feature vector for query fight
            similar_fight_idx: Index of a similar fight

        Returns:
            DataFrame showing feature-by-feature comparison
        """
        if self.feature_vectors is None:
            raise ValueError("Index not built with feature vectors stored.")

        # Get scaled query
        query_scaled = self.scaler.transform(query_features.reshape(1, -1))[0]
        similar_scaled = self.feature_vectors[similar_fight_idx]

        # Calculate absolute differences
        differences = np.abs(query_scaled - similar_scaled)

        # Create comparison dataframe
        comparison = pd.DataFrame({
            'feature': self.feature_names,
            'query_value': query_scaled,
            'similar_value': similar_scaled,
            'difference': differences,
            # Higher = more similar
            'contribution_to_similarity': 1 / (1 + differences)
        })

        comparison = comparison.sort_values(
            'contribution_to_similarity', ascending=False)

        return comparison


def display_similar_fights(results: List[Dict], query_info: str = "") -> None:
    """Pretty print similar fight results."""
    print("\n" + "=" * 70)
    if query_info:
        print(f"Similar Fights to: {query_info}")
    else:
        print("Similar Historical Fights")
    print("=" * 70)

    for i, fight in enumerate(results, 1):
        similarity = fight.get('similarity', 0) * 100

        # Get fighter names
        f1 = fight.get('f_1_name', 'Fighter 1')
        f2 = fight.get('f_2_name', 'Fighter 2')
        winner = fight.get('winner', 'Unknown')

        # Get fight details
        event = fight.get('event_name', '')
        date = fight.get('event_date', '')
        weight_class = fight.get('weight_class', '')
        result = fight.get('result', '')
        result_details = fight.get('result_details', '')
        finish_round = fight.get('finish_round', '')
        finish_time = fight.get('finish_time', '')
        title_fight = fight.get('title_fight', 0)

        # Format title
        title_str = " 🏆 TITLE FIGHT" if title_fight == 1 else ""

        print(f"\n{i}. [{similarity:.1f}% similar]{title_str}")
        print(f"   🥊 {f1} vs {f2}")
        print(f"   🏅 Winner: {winner}")

        if result:
            finish_info = f"{result}"
            if result_details:
                finish_info += f" ({result_details})"
            if finish_round and finish_time:
                finish_info += f" - Round {finish_round}, {finish_time}"
            print(f"   📋 Result: {finish_info}")

        if event:
            print(f"   📅 {event}", end="")
            if date:
                print(f" | {date}", end="")
            if weight_class:
                print(f" | {weight_class}", end="")
            print()

        # Physical comparison if available
        f1_height = fight.get('f_1_fighter_height_cm')
        f2_height = fight.get('f_2_fighter_height_cm')
        f1_reach = fight.get('f_1_fighter_reach_cm')
        f2_reach = fight.get('f_2_fighter_reach_cm')

        if f1_height and f2_height:
            print(f"   📏 Height: {f1_height:.0f}cm vs {f2_height:.0f}cm | "
                  f"Reach: {f1_reach:.0f}cm vs {f2_reach:.0f}cm" if f1_reach and f2_reach else "")

    print("\n" + "=" * 70)


def main():
    """Demo FAISS similarity search."""
    print("\n" + "🔍" * 35)
    print("     FightPredict - FAISS Similarity Search")
    print("🔍" * 35)

    # Initialize searcher
    searcher = FightSimilaritySearch(
        data_path="data/processed",
        models_path="models",
        use_cosine=True
    )

    # Load fighters database
    searcher.load_fighters()

    # Build index
    searcher.build_index(include_odds=False)

    # Save index
    searcher.save_index("faiss_index")

    # Demo 1: Find fights similar to a specific historical fight
    print("\n" + "=" * 60)
    print("Demo 1: Finding fights similar to fight #100")
    print("=" * 60)

    # Get info about fight #100
    if searcher.fight_metadata is not None and len(searcher.fight_metadata) > 100:
        fight_100 = searcher.fight_metadata.iloc[100]
        f1 = fight_100.get('f_1_name', 'Fighter 1')
        f2 = fight_100.get('f_2_name', 'Fighter 2')
        query_info = f"{f1} vs {f2}"
    else:
        query_info = "Fight #100"

    similar = searcher.find_similar_by_index(100, k=5)
    display_similar_fights(similar, query_info)

    # Demo 2: Analyze outcomes of similar fights
    print("\n" + "=" * 60)
    print("Demo 2: Prediction based on similar fight outcomes")
    print("=" * 60)

    query_idx = 50
    query_vector = searcher.feature_vectors[query_idx]

    if searcher.fight_metadata is not None and len(searcher.fight_metadata) > query_idx:
        fight_info = searcher.fight_metadata.iloc[query_idx]
        f1 = fight_info.get('f_1_name', 'Fighter 1')
        f2 = fight_info.get('f_2_name', 'Fighter 2')
        actual_winner = fight_info.get('winner', 'Unknown')
        print(f"\nQuery fight: {f1} vs {f2}")
        print(f"Actual winner: {actual_winner}")

    analysis = searcher.analyze_similar_fights(query_vector, k=10)

    print(f"\n📊 Analysis of 10 most similar historical fights:")
    print(
        f"   Fighter 1 wins in similar fights: {analysis['fighter_1_wins']}/{analysis['total_analyzed']}")
    print(
        f"   Fighter 2 wins in similar fights: {analysis['fighter_2_wins']}/{analysis['total_analyzed']}")
    print(
        f"   Weighted F1 probability: {analysis['weighted_fighter_1_probability']:.1%}")
    print(
        f"   Weighted F2 probability: {analysis['weighted_fighter_2_probability']:.1%}")
    print(
        f"   📈 Prediction: {analysis['prediction']} (confidence: {analysis['confidence']:.0%})")
    print(f"   Average similarity: {analysis['average_similarity']:.3f}")

    # Demo 3: Search for a specific fighter's fights
    print("\n" + "=" * 60)
    print("Demo 3: Find all fights for a specific fighter")
    print("=" * 60)

    # Get a random fighter name
    fighter_names = searcher.get_fighter_names()
    if fighter_names:
        # Pick a well-known fighter if available
        test_fighters = ['Conor McGregor',
                         'Khabib Nurmagomedov', 'Jon Jones', 'Amanda Nunes']
        test_fighter = None
        for name in test_fighters:
            if name in fighter_names:
                test_fighter = name
                break

        if test_fighter is None:
            test_fighter = fighter_names[0]

        print(f"\nSearching for fights involving: {test_fighter}")
        fighter_fights = searcher.find_fights_by_fighter(test_fighter)

        if fighter_fights:
            print(f"Found {len(fighter_fights)} fights:")
            for fight in fighter_fights[:5]:  # Show first 5
                opponent = fight.get('f_2_name') if fight.get(
                    'f_1_name') == test_fighter else fight.get('f_1_name')
                result = fight.get('result', '')
                winner = fight.get('winner', '')
                won = "✓ WON" if winner == test_fighter else "✗ LOST"
                print(f"   vs {opponent}: {won} by {result}")

    print("\n" + "=" * 60)
    print("✓ FAISS Similarity Search Ready!")
    print("=" * 60)
    print("\nUsage for Streamlit:")
    print("   searcher = FightSimilaritySearch()")
    print("   searcher.load_index('faiss_index')")
    print("   similar = searcher.find_similar_fights(feature_vector, k=10)")

    return searcher


if __name__ == "__main__":
    searcher = main()
