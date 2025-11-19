# evaluate.py
import pandas as pd
import numpy as np
import torch
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
import os

from dataset import TimeSeriesDataset
from train import train_and_validate
from model import LSTMAttentionModel, VanillaLSTM
from utils import rmse, mae, save_plot, to_numpy

def split_data(df, test_size=0.2, val_size=0.1):
    n = len(df)
    test_n = int(n * test_size)
    val_n = int(n * val_size)
    train_df = df.iloc[: n - test_n - val_n].reset_index(drop=True)
    val_df = df.iloc[n - test_n - val_n : n - test_n].reset_index(drop=True)
    test_df = df.iloc[n - test_n :].reset_index(drop=True)
    return train_df, val_df, test_df

def evaluate_all(df, seq_len=60, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)

    # ----- Split dataset -----
    train_df = df.iloc[:-200]
    val_df   = df.iloc[-200:]

    # ----- Input dimension from dataframe -----
    input_dim = train_df.shape[1]

    # ----- Parameter sets -----
    lstm_params = {
        "hidden_dim": 64,
        "num_layers": 2
    }

    attn_params = {
        "hidden_dim": 64,
        "num_layers": 2
    }

    # ----- Train Standard LSTM -----
    lstm_res = train_and_validate(
        train_df, val_df,
        seq_len=seq_len,
        epochs=25,
        batch_size=64,
        model_type="lstm",
        params=lstm_params,
        device=None,
        save_path=os.path.join(results_dir, "lstm_best.pt")
    )

    # ----- Train Attention LSTM -----
    attn_res = train_and_validate(
        train_df, val_df,
        seq_len=seq_len,
        epochs=25,
        batch_size=64,
        model_type="attn",
        params=attn_params,
        device=None,
        save_path=os.path.join(results_dir, "attn_best.pt")
    )

    # ----- Summary -----
    summary = {
        "input_dim": input_dim,
        "sequence_length": seq_len,
        "models": {
            "LSTM": lstm_res,
            "Attention_LSTM": attn_res
        }
    }

