import numpy as np
import pandas as pd

def generate_multivariate_series(n_samples=3000, n_features=5, seed=42):

    np.random.seed(seed)
    t = np.arange(n_samples)
    data = np.zeros((n_samples, n_features))

    for f in range(n_features):
        trend = 0.0005 * (f+1) * t  # small trend varying by feature
        period = 50 + 10*f
        season = 0.5 * np.sin(2 * np.pi * t / period + f)
        noise = 0.1 * np.random.randn(n_samples)
        amplitude = 1.0 + 0.2 * f
        data[:, f] = amplitude * (trend + season) + noise

    coefs = np.linspace(0.2, 1.0, n_features)
    target = data.dot(coefs)  # linear comb
    target = target + 0.3 * np.sin(0.02 * t) + 0.2 * np.random.randn(n_samples)
    df = pd.DataFrame(data, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = target
    return df

if __name__ == "__main__":
    df = generate_multivariate_series(n_samples=3000, n_features=5)
    df.to_csv("data.csv", index=False)
    print("Saved data.csv (3000 rows, 6 columns).")

