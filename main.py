"""
Euroleague Fantasy Optimizer - Main Pipeline
Run this to execute the full workflow from data collection to team optimization
"""

import sys
from pathlib import Path

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def run_step(step_name, script_name, description):
    print_header(f"STEP {step_name}: {description}")
    print(f"Running: {script_name}\n")

    try:
        with open(script_name) as f:
            exec(f.read(), {'__name__': '__main__'})
        print(f"\n✓ {step_name} completed successfully")
        return True
    except FileNotFoundError:
        print(f"✗ Error: {script_name} not found")
        return False
    except Exception as e:
        print(f"✗ Error in {step_name}: {str(e)}")
        return False

def main():
    print_header("EUROLEAGUE FANTASY OPTIMIZER")
    print("This pipeline will:")
    print("  1. Collect data from Euroleague & Eurocup APIs")
    print("  2. Engineer features (rolling averages, form indicators)")
    print("  3. Train ML model to predict next round performance")
    print("  4. Optimize team selection within budget constraints")
    print("\nEstimated time: 10-15 minutes")

    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Step 1: Data Collection
    if not run_step("1", "prepare_data.py", "Data Collection"):
        print("\nPipeline failed at Step 1")
        return

    # Step 2: Feature Engineering
    if not run_step("2", "build_features.py", "Feature Engineering"):
        print("\nPipeline failed at Step 2")
        return

    # Step 3: Model Training
    if not run_step("3", "train_model.py", "ML Model Training"):
        print("\nPipeline failed at Step 3")
        return

    # Step 4: Team Optimization
    if not run_step("4", "optimize_team_v2.py", "Team Optimization"):
        print("\nPipeline failed at Step 4")
        return

    # Success
    print_header("PIPELINE COMPLETED SUCCESSFULLY")
    print("Your optimal team has been saved to: optimal_team_ml.csv")
    print("\nNext steps:")
    print("  1. Check if positions fit 4G/4F/2C requirement")
    print("  2. Verify players are healthy (check injury reports)")
    print("  3. Add a coach to your roster")
    print("  4. Submit your team!")

    print("\nFiles generated:")
    print("  - data/processed/fantasy_dataset_2024.csv")
    print("  - data/processed/next_round_predictions.csv")
    print("  - models/fantasy_predictor.pkl")
    print("  - optimal_team_ml.csv")

if __name__ == "__main__":
    # Check required files exist
    required_files = [
        'prepare_data.py',
        'build_features.py',
        'train_model.py',
        'optimize_team_v2.py',
        'euroleague_prices.csv'
    ]

    missing = [f for f in required_files if not Path(f).exists()]

    if missing:
        print("Error: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)

    main()