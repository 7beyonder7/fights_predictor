"""
FightPredict - End-to-End Prediction Example
=============================================

This script demonstrates the complete prediction pipeline:
1. Look up two fighters
2. Generate matchup features
3. Get model predictions
4. Generate LLM-powered analysis

Usage:
    python predict_fight.py --fighter1 "Islam Makhachev" --fighter2 "Charles Oliveira"
    python predict_fight.py --fighter1 "Jon Jones" --fighter2 "Stipe Miocic" --title-fight
    python predict_fight.py --fighter1 "Alex Pereira" --fighter2 "Jamahal Hill" --f1-odds -180 --f2-odds 150
"""

import argparse
import json
import sys
from generate_matchup_features import MatchupFeatureGenerator
from rag_analyzer import FightAnalyzer


def predict_fight(
    fighter1: str,
    fighter2: str,
    weight_class: str = "Unknown",
    title_fight: bool = False,
    f1_odds: int = None,
    f2_odds: int = None,
    provider: str = "ollama",
    model: str = None,
    quick_only: bool = False
):
    """
    Run complete prediction pipeline for a UFC matchup.

    Args:
        fighter1: Fighter 1 name (red corner)
        fighter2: Fighter 2 name (blue corner)
        weight_class: Weight class
        title_fight: Whether it's a title fight
        f1_odds: American odds for fighter 1
        f2_odds: American odds for fighter 2
        provider: LLM provider (ollama, claude, openai)
        model: Model name (optional, uses defaults)
        quick_only: If True, skip LLM analysis
    """
    print("\n" + "="*70)
    print("🥊 FIGHTPREDICT - UFC FIGHT PREDICTION SYSTEM")
    print("="*70)

    # =========================================================================
    # STEP 1: Initialize components
    # =========================================================================
    print("\n📦 Loading components...")

    # Feature generator
    generator = MatchupFeatureGenerator()
    if not generator.load_data():
        print("❌ Failed to load fighter data")
        return

    # RAG analyzer
    analyzer = FightAnalyzer(provider=provider, model=model)
    analyzer.load_models(with_odds=(f1_odds is not None))

    # =========================================================================
    # STEP 2: Look up fighters and validate
    # =========================================================================
    print(f"\n🔍 Looking up fighters...")

    f1_stats = generator.get_fighter_stats(fighter1)
    f2_stats = generator.get_fighter_stats(fighter2)

    if f1_stats is None:
        print(f"❌ Fighter not found: {fighter1}")
        matches = generator.search_fighter(fighter1)
        if matches:
            print(f"   Did you mean: {', '.join(matches[:5])}")
        return

    if f2_stats is None:
        print(f"❌ Fighter not found: {fighter2}")
        matches = generator.search_fighter(fighter2)
        if matches:
            print(f"   Did you mean: {', '.join(matches[:5])}")
        return

    print(f"✓ Found: {f1_stats.name}")
    print(f"✓ Found: {f2_stats.name}")

    # =========================================================================
    # STEP 3: Display matchup summary
    # =========================================================================
    print("\n" + "-"*70)
    print("📊 MATCHUP OVERVIEW")
    print("-"*70)

    print(f"\n{'':20} {'FIGHTER 1':>20}  vs  {'FIGHTER 2':<20}")
    print(f"{'':20} {f1_stats.name:>20}  vs  {f2_stats.name:<20}")
    print()
    f1_record = f"{f1_stats.wins}-{f1_stats.losses}-{f1_stats.draws}"
    f2_record = f"{f2_stats.wins}-{f2_stats.losses}-{f2_stats.draws}"
    print(f"{'Record:':<20} {f1_record:>17}       {f2_record:<17}")
    print(f"{'Age:':<20} {f1_stats.age:>17.0f}       {f2_stats.age:<17.0f}")
    print(f"{'Height:':<20} {f1_stats.height_cm:>14.0f} cm       {f2_stats.height_cm:<14.0f} cm")
    print(f"{'Reach:':<20} {f1_stats.reach_cm:>14.0f} cm       {f2_stats.reach_cm:<14.0f} cm")
    print(f"{'Stance:':<20} {f1_stats.stance:>17}       {f2_stats.stance:<17}")
    print(f"{'UFC Fights:':<20} {f1_stats.ufc_fights:>17}       {f2_stats.ufc_fights:<17}")
    print(f"{'Current Streak:':<20} {f1_stats.current_streak:>17}       {f2_stats.current_streak:<17}")

    print(f"\n{'STRIKING':^70}")
    print(f"{'SLpM:':<20} {f1_stats.slpm:>17.2f}       {f2_stats.slpm:<17.2f}")
    print(f"{'Str Acc:':<20} {f1_stats.str_acc:>16.1%}       {f2_stats.str_acc:<16.1%}")
    print(f"{'Str Def:':<20} {f1_stats.str_def:>16.1%}       {f2_stats.str_def:<16.1%}")

    print(f"\n{'GRAPPLING':^70}")
    print(f"{'TD Avg:':<20} {f1_stats.td_avg:>17.2f}       {f2_stats.td_avg:<17.2f}")
    print(f"{'TD Acc:':<20} {f1_stats.td_acc:>16.1%}       {f2_stats.td_acc:<16.1%}")
    print(f"{'TD Def:':<20} {f1_stats.td_def:>16.1%}       {f2_stats.td_def:<16.1%}")
    print(f"{'Sub Avg:':<20} {f1_stats.sub_avg:>17.2f}       {f2_stats.sub_avg:<17.2f}")

    if f1_odds and f2_odds:
        print(f"\n{'BETTING ODDS':^70}")
        print(f"{'American Odds:':<20} {f1_odds:>+17}       {f2_odds:<+17}")

    # =========================================================================
    # STEP 4: Generate features
    # =========================================================================
    print("\n" + "-"*70)
    print("⚙️  GENERATING FEATURES")
    print("-"*70)

    try:
        features, feature_names = generator.generate_features(
            fighter1=f1_stats.name,
            fighter2=f2_stats.name,
            weight_class=weight_class,
            title_fight=title_fight,
            f1_odds=f1_odds,
            f2_odds=f2_odds,
            include_odds=(f1_odds is not None)
        )
        print(f"✓ Generated {len(features)} features")
    except Exception as e:
        print(f"❌ Feature generation failed: {e}")
        return

    # =========================================================================
    # STEP 5: Get model predictions - UPDATED OUTPUT
    # =========================================================================
    print("\n" + "-"*70)
    print("🤖 MODEL PREDICTIONS")
    print("-"*70)

    result = analyzer.quick_predict(features, f1_stats.name, f2_stats.name)

    if "predicted_winner" in result:
        winner = result["predicted_winner"]
        loser = f2_stats.name if winner == f1_stats.name else f1_stats.name
        confidence = result["confidence"]

        # Get win probabilities
        win_prob = result.get("win_probability", {})
        f1_prob = win_prob.get(f1_stats.name, 0.5)
        f2_prob = win_prob.get(f2_stats.name, 0.5)

        # Print prediction summary
        print(f"\n🏆 PREDICTED WINNER: {winner}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"")
        print(f"   Win Probabilities:")
        print(f"   • {f1_stats.name}: {f1_prob:.1%}")
        print(f"   • {f2_stats.name}: {f2_prob:.1%}")
        sys.stdout.flush()

        ensemble = result.get("ensemble", {})
        if ensemble.get("individual_predictions"):
            print(f"\n   Individual Model Breakdown:")
            print(
                f"   {'Model':<10} {'Prediction':<20} {'P(F1)':<10} {'Threshold':<10} {'Margin':<10} {'Confidence':<10}")
            print(f"   {'-'*70}")

            for pred in ensemble["individual_predictions"]:
                model_name = pred['model']
                pred_winner = f1_stats.name if pred["prediction"] == 1 else f2_stats.name
                prob_f1 = pred["probability_f1"]
                threshold = pred["threshold"]
                margin = pred["margin"]
                conf = pred["confidence"]

                print(
                    f"   {model_name:<10} {pred_winner:<20} {prob_f1:>8.1%}   {threshold:>8.3f}   {margin:>+8.1%}   {conf:>8.1%}")

            print()
            if ensemble.get("models_agree"):
                print(f"   ✓ Models AGREE on prediction")
            else:
                method = ensemble.get("method", "unknown")
                print(
                    f"   ⚠ Models DISAGREE - used {method} for final prediction")

        # Similar fights analysis
        if result.get("similar_fights_analysis"):
            sim_analysis = result["similar_fights_analysis"]
            if "f1_win_rate" in sim_analysis:
                print(
                    f"\n   📊 Similar Fights Analysis ({sim_analysis.get('k', 10)} historical matches):")
                print(
                    f"   • Fighter 1 win rate in similar matchups: {sim_analysis['f1_win_rate']:.1%}")
                sim_pred = sim_analysis.get('prediction', -1)
                ensemble_pred = ensemble.get('ensemble_prediction', -1)
                aligns = "Yes ✓" if sim_pred == ensemble_pred else "No ⚠"
                print(f"   • Similar fights align with prediction: {aligns}")

    # =========================================================================
    # STEP 6: LLM Analysis (optional)
    # =========================================================================
    if not quick_only:
        print("\n" + "-"*70)
        print("📝 LLM ANALYSIS")
        print("-"*70)

        if not analyzer.llm.is_available():
            print(f"\n⚠️  LLM provider ({provider}) not available")
            if provider == "ollama":
                print("   Make sure Ollama is running: ollama serve")
            else:
                print(
                    f"   Set {provider.upper()}_API_KEY environment variable")
        else:
            print(
                f"\nGenerating analysis with {provider} ({analyzer.config.model})...\n")

            try:
                analysis = analyzer.analyze_from_features(
                    features,
                    f1_stats.name,
                    f2_stats.name,
                    verbose=False
                )
                print(analysis)
            except Exception as e:
                print(f"❌ LLM analysis failed: {e}")

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70 + "\n")

    return result


def interactive_mode():
    """Interactive mode for exploring fighters and matchups."""
    print("\n" + "="*70)
    print("🥊 FIGHTPREDICT - INTERACTIVE MODE")
    print("="*70)

    generator = MatchupFeatureGenerator()
    if not generator.load_data():
        print("❌ Failed to load data")
        return

    analyzer = None  # Lazy load

    while True:
        print("\n" + "-"*40)
        print("Commands:")
        print("  search <name>  - Search for a fighter")
        print("  info <name>    - Get fighter info")
        print("  predict        - Predict a matchup")
        print("  provider       - Change LLM provider")
        print("  quit           - Exit")
        print("-"*40)

        cmd = input("\n> ").strip().lower()

        if cmd.startswith("search "):
            query = cmd[7:]
            results = generator.search_fighter(query)
            if results:
                print(f"\nFound {len(results)} fighters:")
                for name in results:
                    print(f"  • {name}")
            else:
                print("No fighters found.")

        elif cmd.startswith("info "):
            name = cmd[5:]
            stats = generator.get_fighter_stats(name)
            if stats:
                print(f"\n{stats.name}")
                print(f"  Record: {stats.wins}-{stats.losses}-{stats.draws}")
                print(f"  Age: {stats.age:.0f}")
                print(f"  Height: {stats.height_cm:.0f} cm")
                print(f"  Reach: {stats.reach_cm:.0f} cm")
                print(f"  Stance: {stats.stance}")
                print(f"  UFC Fights: {stats.ufc_fights}")
                print(
                    f"  Striking: {stats.slpm:.2f} SLpM, {stats.str_acc:.1%} acc")
                print(
                    f"  Grappling: {stats.td_avg:.2f} TD/15min, {stats.sub_avg:.2f} sub/15min")
            else:
                print("Fighter not found.")

        elif cmd == "predict":
            f1 = input("Fighter 1: ").strip()
            f2 = input("Fighter 2: ").strip()

            title = input("Title fight? (y/n): ").strip().lower() == 'y'

            odds_input = input("Enter odds? (y/n): ").strip().lower()
            f1_odds = f2_odds = None
            if odds_input == 'y':
                try:
                    f1_odds = int(input(f"  {f1} odds (e.g., -150): "))
                    f2_odds = int(input(f"  {f2} odds (e.g., +130): "))
                except:
                    print("Invalid odds, skipping.")

            quick = input(
                "Quick predict only (no LLM)? (y/n): ").strip().lower() == 'y'

            # Initialize analyzer if needed
            if analyzer is None and not quick:
                provider = input(
                    "LLM provider (ollama/claude/openai): ").strip() or "ollama"
                analyzer = FightAnalyzer(provider=provider)
                analyzer.load_models(with_odds=(f1_odds is not None))

            predict_fight(
                f1, f2,
                title_fight=title,
                f1_odds=f1_odds,
                f2_odds=f2_odds,
                provider=analyzer.config.provider if analyzer else "ollama",
                quick_only=quick
            )

        elif cmd == "provider":
            provider = input("New provider (ollama/claude/openai): ").strip()
            model = input("Model (press enter for default): ").strip() or None
            analyzer = FightAnalyzer(provider=provider, model=model)
            analyzer.load_models()
            print(f"Switched to {provider}")

        elif cmd in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        else:
            print("Unknown command. Try 'search', 'info', 'predict', or 'quit'.")


def main():
    parser = argparse.ArgumentParser(
        description="FightPredict - UFC Fight Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_fight.py -f1 "Islam Makhachev" -f2 "Charles Oliveira"
  python predict_fight.py -f1 "Jon Jones" -f2 "Stipe Miocic" --title-fight
  python predict_fight.py -f1 "Alex Pereira" -f2 "Jamahal Hill" --f1-odds -180 --f2-odds 150
  python predict_fight.py --interactive
        """
    )

    parser.add_argument("--fighter1", "-f1", type=str, help="Fighter 1 name")
    parser.add_argument("--fighter2", "-f2", type=str, help="Fighter 2 name")
    parser.add_argument("--weight-class", "-w", type=str,
                        default="Unknown", help="Weight class")
    parser.add_argument("--title-fight", "-t",
                        action="store_true", help="Title fight")
    parser.add_argument("--f1-odds", type=int, help="Fighter 1 American odds")
    parser.add_argument("--f2-odds", type=int, help="Fighter 2 American odds")
    parser.add_argument("--provider", "-p", type=str, default="ollama",
                        choices=["ollama", "claude", "openai"], help="LLM provider")
    parser.add_argument("--model", "-m", type=str, help="Model name")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick predict only (no LLM)")
    parser.add_argument("--interactive", "-i",
                        action="store_true", help="Interactive mode")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.fighter1 and args.fighter2:
        predict_fight(
            fighter1=args.fighter1,
            fighter2=args.fighter2,
            weight_class=args.weight_class,
            title_fight=args.title_fight,
            f1_odds=args.f1_odds,
            f2_odds=args.f2_odds,
            provider=args.provider,
            model=args.model,
            quick_only=args.quick
        )
    else:
        parser.print_help()
        print("\n💡 Try: python predict_fight.py --interactive")


if __name__ == "__main__":
    main()
