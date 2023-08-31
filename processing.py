import pandas as pd
import numpy as np
import os
import joblib
from keras.models import load_model
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


def load_history(epochs, units):
    history_filename = f'models/adam_optimizer_KRUPP/{epochs}epochs/history_units_{units}.joblib'
    return joblib.load(history_filename)

def krupp_pre_pro(file_path):
    
    df = pd.read_excel(file_path, sheet_name='Synthèse', header=2, skiprows=0)

    df = df.dropna()
    df.drop(columns=['poste 3', 'poste 1', 'poste 2', 'poste 3.1', 'poste 1.1', 'poste 2.1', 'poste 3.2', 'poste 1.2', 'poste 2.2', 'poste 3.3', 'poste 1.3', 'poste 2.3'], inplace=True)
    new_column_names = {'Journée': 'THE', 'Journée.1': 'THC', 'Journée.2': 'HM', 'Journée.3': 'Rendement'}
    df.rename(columns=new_column_names, inplace=True)
    df.set_index('Date', inplace=True)

    return df

def preforcast(app, df, station):
    
    if station == 'models/krupp.h5':
        time_steps = 30
    else:
        time_steps = 7
    
    data = df.filter(['THC'])


    model = load_model(os.path.join(app.static_folder, station))
    dataset = data.values

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Initialize lists to store the sequences and corresponding labels
    X_train, y_train = [], []
    X_test, y_test = [], []

    # Generate sequences for training data
    for i in range(len(scaled_data) - time_steps):
        X_train.append(scaled_data[i:i + time_steps])
        y_train.append(scaled_data[i + time_steps])

    # Convert lists to numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Calculate the index to split between train and test sets
    split_index = int(len(scaled_data) * 0.8)  # Use 80% of data for training

    # Split the data into train and test sets
    X_train, X_test = X_train[:split_index], X_train[split_index:]
    y_train, y_test = y_train[:split_index], y_train[split_index:]
    
    train = df[:-len(y_test)]
    valid = df[-len(y_test):]
    
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)

    valid.loc[:, 'Predictions'] = y_pred

    return train, valid

def forcast_peocessing(app, df, station):
    model = load_model(os.path.join(app.static_folder, station))

    # Number of days to predict
    if station == 'models/krupp.h5':
        future_time_steps = 30
    else:
        future_time_steps = 7

    # Get the last date in the original DataFrame
    last_date = df.index[-1]

    data = df.filter(['THC'])
    dataset = data.values

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Generate future dates based on the number of future predictions
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_time_steps, freq='D')
    future_input_data = scaled_data[-future_time_steps:].reshape(1, future_time_steps, 1)

    for i in range(future_time_steps):
        # Predict the next value
        future_prediction = model.predict(future_input_data)
        # Append the prediction to the input data for the next iteration
        future_input_data = np.append(future_input_data, future_prediction.reshape(1, 1, 1), axis=1)

    # Inverse transform the predictions to get the actual values
    future_predictions = scaler.inverse_transform(future_input_data[0])

    future_predictions = future_predictions.flatten()
    future_predictions = future_predictions[future_time_steps:]
    
    return future_dates, future_predictions

#------------------------------------------------------------------------

def koch_pre_pro(file_path):
    
    df = pd.read_excel('R0_KOCH_2023.xlsm', sheet_name='synthèse 312', header=2, skiprows=0)

    df = df.dropna()
    df.drop(columns=['poste 3', 'poste 1', 'poste 2', 'Unnamed: 4', 'poste 3.1', 'poste 1.1', 'poste 2.1', 'poste 3.2', 'poste 1.2', 'poste 2.2', 'Unnamed: 12', 'poste 3.3', 'poste1', 'poste 2.3','Unnamed: 16', 'poste 3.4', 'poste 1.3', 'poste 2.4'], inplace=True)
    new_column_names = {'Unnamed: 8': 'THC', 'Unnamed: 20': 'Rendement'}
    df.rename(columns=new_column_names, inplace=True)
    df.set_index('Date', inplace=True)

    return df