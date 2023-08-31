import pandas as pd
import os
from keras.models import load_model
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras.layers import Input
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def display_data(app, df, station):
    plt.figure(figsize=(10, 6))

    # Create the plot using Matplotlib
    if station == 'models/krupp.h5':
        plt.plot(df.index, df['THE'], label='THE')
    
    plt.plot(df.index, df['THC'], label='THC')
    plt.plot(df.index, df['Rendement'], label='Rendement')

    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title('Data visualization')

    plt.legend()
    if station == 'models/krupp.h5':
        graph_path = os.path.join(app.static_folder, 'graphs', 'krupp_exploration_graph.png')
    else:
        graph_path = os.path.join(app.static_folder, 'graphs', 'koch_exploration_graph.png')


    # Save the figure
    plt.savefig(graph_path, format='png')
    plt.close()

def ts_decomp_graph(app, df, station):

    # Ensure the DataFrame has a DateTimeIndex with daily frequency
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(date_range)
    df = df.fillna(0)

    # Interpolate missing values
    df['THC'] =  df['THC'].interpolate()

    # Perform Seasonal Decomposition using seasonal_decompose
    decomposition = seasonal_decompose(df['THC'], model='additive', extrapolate_trend='freq')

    # Get the components (trend, seasonal, and residuals)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residuals = decomposition.resid

    # Plot the original time series and the components
    plt.figure(figsize=(10, 8))
    plt.subplot(4, 1, 1)
    plt.plot(df['THC'], label='Original Time Series')
    plt.legend(loc='upper left')
    plt.title('Time Serie Decomposition')

    plt.subplot(4, 1, 2)
    plt.plot(trend, label='Trend')
    plt.legend(loc='upper left')
    plt.title('Trend Component')

    plt.subplot(4, 1, 3)
    plt.plot(seasonal, label='Seasonal')
    plt.legend(loc='upper left')
    plt.title('Seasonal Component')

    plt.subplot(4, 1, 4)
    plt.plot(residuals, label='Residuals')
    plt.legend(loc='upper left')
    plt.title('Residual Component')

    plt.tight_layout()
    if station == 'models/krupp.h5':
        graph_path = os.path.join(app.static_folder, 'graphs', 'krupp_tsdecomp_graph.png')
    else:
        graph_path = os.path.join(app.static_folder, 'graphs', 'koch_tsdecomp_graph.png')
    # Save the figure
    plt.savefig(graph_path, format='png')
    plt.close()

def rolling_stats(app, df, station):
    rolling_mean = df['THC'].rolling(window=30).mean()

    # Calculate Rolling Standard Deviation with a window size of 30 days
    rolling_std = df['THC'].rolling(window=30).std()

    # Plot the original time series data along with the Rolling Mean and Rolling Standard Deviation
    plt.figure(figsize=(10, 6))
    plt.plot(df['THC'], label='Original Data')
    plt.plot(rolling_mean, label='Rolling Mean (window=30 days)', color='orange')
    plt.plot(rolling_std, label='Rolling Std Dev (window=30 days)', color='red')

    plt.title('Rolling Statistics: Mean and Standard Deviation')
    plt.xlabel('Date')
    plt.ylabel('THC')
    plt.legend()
    plt.grid(True)

    
    if station == 'models/krupp.h5':
        graph_path = os.path.join(app.static_folder, 'graphs', 'krupp_rollingstats_graph.png')
    else:
        graph_path = os.path.join(app.static_folder, 'graphs', 'koch_rollingstats_graph.png')


    # Save the figure
    plt.savefig(graph_path, format='png')
    plt.close()

def model_test_graph(app, train, valid, station):
    
    plt.figure(figsize=(10, 6))
    plt.title('Model PerforMance on test set')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('THC Production', fontsize=18)
    plt.plot(train['THC'])
    plt.plot(valid[['THC', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

    if station == 'models/krupp.h5':
        graph_path = os.path.join(app.static_folder, 'graphs', 'krupp_test_graph.png')
    else:
        graph_path = os.path.join(app.static_folder, 'graphs', 'koch_test_graph.png')

    # Save the figure
    plt.savefig(graph_path, format='png')
    plt.close()


def forcast_graph(app, train, valid, future_dates, future_predictions, station):
    plt.figure(figsize=(10, 6))

    plt.plot(train['THC'], label='Original', color='blue')
    plt.plot(valid['THC'], label='Actual test', color='red')
    plt.plot(valid['Predictions'], label='Predicted test', color='yellow')

    # Plot the predicted values in red
    plt.plot(future_dates, future_predictions, label='Future values', color='green')
    
    plt.xlabel('Date')
    plt.ylabel('THC Production')
    plt.title('Actual vs. Predicted THC Production and future values Model')
    plt.legend()

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    plt.tight_layout()

    if station == 'models/krupp.h5':
        graph_path = os.path.join(app.static_folder, 'graphs', 'krupp_forcast_graph.png')
    else:
        graph_path = os.path.join(app.static_folder, 'graphs', 'koch_forcast_graph.png')

    plt.savefig(graph_path, format='png')
    plt.close()
