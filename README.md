# Time-Series-Forecasting-LSTM-Attention
Multivariate Time-Series Forecasting with LSTM/GRU + Attention

This project builds an advanced deep learning model to forecast future values of a multivariate time-series dataset using LSTM/GRU networks combined with an Attention mechanism.
It also compares performance with traditional models (ARIMA) and baseline LSTM models to evaluate improvements.

🔍 Project Overview

The goal is to create a complete forecasting pipeline that:

Generates or loads a synthetic multivariate time-series dataset (5+ features, 1000+ rows).

Preprocesses and transforms the data for supervised learning.

Builds deep learning models:

LSTM + Attention

GRU + Attention

Baseline LSTM

Trains and tunes model hyperparameters.

Compares all models using MAE, MSE, RMSE.

Produces clear plots of predictions vs actual values.

📌 Key Features

Fully automated dataset generation (trend + noise + seasonality).

Configurable architecture (layers, units, dropout, attention).

Clean modular code for training, predicting, and evaluating.

Side-by-side model comparison.

Easy to customize for any dataset.

🧠 Technologies Used

Python

TensorFlow / Keras

NumPy

Pandas

Matplotlib

Scikit-learn

Statsmodels (for ARIMA)

📊 Model Comparison

Each model is evaluated using:

MAE (Mean Absolute Error)

MSE (Mean Squared Error)

RMSE (Root Mean Squared Error)

Plots show:

Actual vs Predicted values
How to Run
pip install -r requirements.txt
python main.py

📜 Output

You will get:

Forecast plot

Model performance report

Best model summary

Saved model weights
Training & validation loss curves
