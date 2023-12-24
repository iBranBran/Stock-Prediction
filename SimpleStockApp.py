import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import math

# Function to fetch stock data
def fetch_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data

# LSTM Model
def model(stock_data):
    closingData = stock_data.filter(['Close'])
    dataset = closingData.values
    training_data_len = math.ceil(len(dataset) * 0.8)
    #Scale Data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    #Create Training Dataset
    #Create Scale Training Dataset
    train_data = scaled_data[0:training_data_len,:]
    x_train =[]
    y_train =[]
    for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])
    #convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    #reshape dataset from 2D to 3D
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    #Build LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences= True, input_shape=(x_train.shape[1], 1) ))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    #Train the model
    hist = model.fit(x_train,y_train, batch_size = 8, epochs = 100)

    #create the testing dataset
    #create new array constaining scaled values
    test_data = scaled_data[training_data_len - 60: ,:]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range (60, len(test_data)):
        x_test.append(test_data[i-60: i, 0])

    #Convert Data to numpy array
    x_test = np.array(x_test)
    #Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #Get the predicted price values based on model
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    #Get model Accuracy
    accuracy = np.mean((y_test - abs(predictions - y_test))/ y_test)


    train = closingData[:training_data_len]
    valid = closingData[training_data_len:]
    valid['Predictions'] = predictions

    return train, valid, accuracy, scaler, model

# Streamlit app
def main():
    st.title("Stock Data and LSTM Model Visualization App")

    # Sidebar for user input
    st.sidebar.header("User Input")

    start_date = st.sidebar.date_input("Select Start Date", pd.to_datetime('2022-01-01'))
    end_date = st.sidebar.date_input("Select End Date", pd.to_datetime('2023-01-01'))

    selected_stock = st.sidebar.selectbox("Select Stock", ["AAPL", "MSFT", "TSLA"])

    # Fetch stock data
    stock_data = fetch_stock_data(selected_stock, start_date, end_date)

    # Display stock data
    st.subheader(f"Stock Data for {selected_stock}")
    st.dataframe(stock_data, width=800)  # Adjust width as needed

    # Plot stock data
    st.subheader(f"{selected_stock} Stock Price Plot")
    plt.figure(figsize=(20, 10))  # Adjust figure size here
    plt.plot(stock_data['Close'])
    plt.title(f"{selected_stock} Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    st.pyplot(plt)

    # Plot LSTM model
    st.subheader(f"LSTM Model Prediction")
    plt.figure(figsize=(14, 7))
    plt.title('LSTM Model Predictions')
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Close Price (USD)', fontsize=10)
    train, valid ,accuracy, scaler, lstm_model= model(stock_data)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Actual', 'Predictions'], loc='lower right')
    st.pyplot(plt)
    st.write(f'Model Accuracy: {accuracy}%')
    
    # Predict next day price
    # Create new dataframe
    newdf = stock_data.filter(['Close'])
    # Obtain the last 60 days closing price values and convert the dataframe to an array
    last_60_days = newdf[-60:].values
    # Scale the data to be values inbetween 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)
    # Create an empty list
    X_test = []
    # Append the past 60 days
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    # Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # Predict the scaled price
    predicted_price = lstm_model.predict (X_test)
    # Reverse the scaling
    pred_price = scaler.inverse_transform(predicted_price)
    st.write(f'The predicted next day price is: ${pred_price} USD')


if __name__ == "__main__":
    main()

