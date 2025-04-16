📈 Stock Market Prediction using LSTM
This project leverages Long Short-Term Memory (LSTM) neural networks to predict stock prices based on historical data. It demonstrates how deep learning can be applied to time-series forecasting tasks in the finance domain.

🚀 Features
Loads and visualizes historical stock data

Normalizes the data for neural network training

Builds an LSTM model using Keras/TensorFlow

Trains the model and evaluates its performance

Plots predicted vs actual stock prices

🛠️ Technologies Used
Python

NumPy

Pandas

Matplotlib

Scikit-learn

TensorFlow / Keras

📊 Dataset
The model uses historical stock price data (e.g., from Yahoo Finance).

The Close price is primarily used for prediction.

🔧 How It Works
Data Preprocessing:

Load and visualize historical stock data.

Normalize data using MinMaxScaler.

Prepare sequences for LSTM input.

Model Creation:

LSTM layers with dropout regularization.

Dense output layer for final prediction.

Training:

Train the model on training set.

Evaluate on the test set using Root Mean Square Error (RMSE).

Prediction & Visualization:

Compare predicted and actual values.

Visualize using matplotlib



🧠 Model Architecture
LSTM (50 units) → Dropout (0.2)
→ LSTM (50 units) → Dropout (0.2)
→ Dense (1 unit)

📈 Sample Output
Graph of actual vs predicted stock prices

RMSE score to indicate model accuracy

