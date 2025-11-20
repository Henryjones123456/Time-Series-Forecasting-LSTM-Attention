import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def save_plot(y_true, y_pred, fname, title="Forecast vs Actual", n_points=200):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.plot(y_true[:n_points], label="Actual")
    plt.plot(y_pred[:n_points], label="Predicted")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)

