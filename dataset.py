# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, data_df, seq_len=60, scaler=None, train=True):
        """
        data_df: pandas DataFrame with features and 'target' column
        seq_len: number of past timesteps to use
        scaler: sklearn scaler fitted on training data (if provided)
        """
        self.seq_len = seq_len
        values = data_df.values.astype(float)
        self.features = values[:, :-1]
        self.targets = values[:, -1].reshape(-1, 1)

        if scaler is None:
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
            # fit on all available samples (callers should ensure train split)
            self.feature_scaler.fit(self.features)
            self.target_scaler.fit(self.targets)
        else:
            self.feature_scaler, self.target_scaler = scaler

        self.features = self.feature_scaler.transform(self.features)
        self.targets = self.target_scaler.transform(self.targets)

        self.X, self.y = self._create_windows(self.features, self.targets, seq_len)

    def _create_windows(self, feats, targs, seq_len):
        X, y = [], []
        for i in range(len(feats) - seq_len):
            X.append(feats[i:i+seq_len])
            y.append(targs[i+seq_len])  # next-step target
        X = np.array(X)  # (N, seq_len, n_features)
        y = np.array(y)  # (N, 1)
        return X.astype(np.float32), y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

    def get_scalers(self):
        return (self.feature_scaler, self.target_scaler)
