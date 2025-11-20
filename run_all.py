import pandas as pd
from data import generate_multivariate_series
from evaluate import evaluate_all

def main():
    print("Generating data...")
    df = generate_multivariate_series(n_samples=3000, n_features=5, seed=42)
    df.to_csv("data.csv", index=False)
    print("Data saved to data.csv")

    print("Starting evaluation (training + baselines) ...")
    summary = evaluate_all(df, seq_len=60, results_dir="results")
    print("Done. Check results/ for models, plots and metrics_summary.csv")

if __name__ == "__main__":
    main()

