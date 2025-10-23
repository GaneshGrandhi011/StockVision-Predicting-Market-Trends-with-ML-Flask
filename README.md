# 📈 Stock Price Prediction with LSTM

This project implements a time-series forecasting model using an **LSTM (Long Short-Term Memory)** neural network to predict future stock prices. 🧠

The model is built with `Keras` (TensorFlow) and trained on historical stock data downloaded from `yfinance`. The system includes two main components: a script to train and save the model, and a prediction script that loads the model to forecast prices for the next two consecutive days.

It also features a built-in currency conversion from **USD to INR** using the `forex-python` library.

---

## ✨ Key Features

* 🔮 **Time-Series Forecasting:** Utilizes a stacked LSTM model to capture long-term dependencies in stock price data.
* 📦 **Model Persistence:** Saves the trained model (`.h5`) and the data scaler (`.pkl`) for reuse, so you only need to train once.
* 🚀 **Two-Day Prediction:** Predicts the price for **tomorrow** and the **day after tomorrow** by feeding its own prediction back into the model.
* 🌍 **Currency Conversion:** Automatically fetches the live USD-to-INR exchange rate to provide prices in both currencies.
* 🛡️ **Robust Fallback:** If the live currency API fails, it gracefully falls back to a default rate to prevent crashes.
* 📊 **Data Handling:** Uses `yfinance` for data retrieval and `scikit-learn` for data preprocessing (MinMax scaling).

---

## 💻 Tech Stack

* **Python 3.x**
* **TensorFlow (Keras):** For building and training the LSTM model.
* **scikit-learn:** For data normalization.
* **yfinance:** For downloading historical stock data from Yahoo! Finance.
* **NumPy:** For numerical operations.
* **forex-python:** For currency conversion.
* **Pickle:** For saving and loading the scaler object.

---

## 🛠️ Installation

1.  Ensure you have Python 3 installed.
2.  Install all the required libraries using pip:

    ```bash
    pip install tensorflow numpy yfinance scikit-learn forex-python
    ```

---

## 🚀 How to Use

The project is divided into two main parts.

### 1. Train the Model

First, you must train the model using the first script (you can save it as `train.py`). This script downloads historical data (for "GOOG" by default), trains the model, and saves `stock_model.h5` and `scaler.pkl` to your directory.

```bash
python train.py