import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime
import pickle

# Step 1: Download stock data
def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Step 2: Preprocess the data
def preprocess_data(data):
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    return scaled_data, scaler

# Step 3: Create the training dataset
def create_datasets(scaled_data, time_step=60):
    x_train, y_train = [], []
    for i in range(time_step, len(scaled_data)):
        x_train.append(scaled_data[i-time_step:i, 0])
        y_train.append(scaled_data[i, 0])
    return np.array(x_train), np.array(y_train)

# Step 4: Build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main training function
def main():
    ticker = "GOOG"  # You can train on a specific stock like Google
    start_date = "2015-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    print("Step 1: Downloading data...")
    data = download_stock_data(ticker, start_date, end_date)
    
    print("Step 2: Preprocessing data...")
    scaled_data, scaler = preprocess_data(data)
    
    print("Step 3: Creating training datasets...")
    time_step = 60
    x_train, y_train = create_datasets(scaled_data, time_step)
    
    # Reshape the data for LSTM [samples, time_steps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    print("Step 4: Building and training the LSTM model...")
    model = build_lstm_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    
    print("Step 5: Saving the model and scaler...")
    # Save the trained model
    model.save('stock_model.h5')
    
    # Save the scaler object
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print("\nTraining complete. 'stock_model.h5' and 'scaler.pkl' have been saved.")

if __name__ == "__main__":
    main()