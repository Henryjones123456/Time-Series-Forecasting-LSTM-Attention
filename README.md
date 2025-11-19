# Time-Series-Forecasting-LSTM-Attention
Time-Series Forecasting Using LSTM/GRU with Attention Mechanism

This project focuses on predicting future values of a time-series using deep learning. The main goal is to build an advanced forecasting model using LSTM or GRU combined with an Attention mechanism, and compare it with traditional and baseline models.

A synthetic multivariate dataset (with more than 1000 records and 5+ features) is created to represent a realistic scenario with trend, seasonality, and noise. The dataset includes temperature, humidity, pressure, wind speed, and energy demand values. After preprocessing and scaling the data, a supervised structure is created for training the neural networks.

The core model uses LSTM/GRU layers to capture long-term patterns and the Attention layer to focus on the most important time steps. Hyperparameters like number of layers, hidden size, batch size, learning rate, and sequence length are tuned to improve performance.

Finally, the model is compared with:

Traditional ARIMA

Basic LSTM without attention

Our Attention-based LSTM/GRU

The comparison is done using MAE, MSE, and RMSE.
The project concludes with selecting the best model and listing its optimal hyperparameters.
