import numpy as np
import yfinance as yf
from keras.models import load_model
import pickle
from datetime import datetime, timedelta
from forex_python.converter import CurrencyRates

# --- Load Model and Scaler (Done once at startup) ---
print("Loading model and scaler...")
model = load_model("stock_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
print("Model and scaler loaded successfully.")

def predict_stock(ticker):
    # Download the latest data for the prediction
    end_date = datetime.today()
    start_date = end_date - timedelta(days=100)
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}. It may be an invalid symbol.")

    # Get the last 60 days of closing prices
    last_60_days = data['Close'][-60:].values.reshape(-1, 1)
    
    # Scale the data using the loaded scaler
    last_60_scaled = scaler.transform(last_60_days)
    
    # --- Predict Tomorrow ---
    X_test_tomorrow = np.array([last_60_scaled.flatten()])
    X_test_tomorrow = np.reshape(X_test_tomorrow, (X_test_tomorrow.shape[0], X_test_tomorrow.shape[1], 1))
    pred_tomorrow_scaled = model.predict(X_test_tomorrow)
    pred_tomorrow = scaler.inverse_transform(pred_tomorrow_scaled)[0, 0]

    # --- Predict Day After Tomorrow ---
    new_sequence = np.append(last_60_scaled[1:], pred_tomorrow_scaled)
    new_sequence = np.reshape(new_sequence, (1, new_sequence.shape[0], 1))
    pred_day_after_scaled = model.predict(new_sequence)
    pred_day_after = scaler.inverse_transform(pred_day_after_scaled)[0, 0]

    # --- Get Currency Rate with Fallback ---
    # ✅ This try/except block makes the code robust
    try:
        c = CurrencyRates()
        usd_to_inr = c.get_rate('USD', 'INR')
        print("Successfully fetched live currency rate.")
    except:
        # If it fails, use a default rate and print a warning to the terminal
        usd_to_inr = 83.50  # A recent approximate rate
        print("WARNING: Could not fetch live currency rate. Using default fallback rate.")


    # --- Return Results ---
    return {
        'ticker': ticker,
        'tomorrow_price_usd': float(pred_tomorrow),
        'day_after_price_usd': float(pred_day_after),
        'tomorrow_price_inr': float(pred_tomorrow * usd_to_inr),
        'day_after_price_inr': float(pred_day_after * usd_to_inr),
        'usd_to_inr_rate': float(usd_to_inr)
    }