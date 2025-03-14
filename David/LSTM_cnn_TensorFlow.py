# -*- coding: utf-8 -*-
"""
rnn_cnn_v2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ClLJ1vHQsGSwmpj5rBeGdBqW43I2rUE-
"""

# Commented out IPython magic to ensure Python compatibility.
#from google.colab import drive
#drive.mount('/gdrive',force_remount=True)
# %cd /gdrive/MyDrive/DAVID/David

"""
# https://www.tensorflow.org/tutorials/structured_data/time_series
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,SimpleRNN
import optuna
from tensorflow.keras.layers import LSTM

import pickle
from itw_tools import get_demands, no_date_gaps, get_forecasts

# how many previous datapoint to check
lookback = 30

# get the data
with open("testdata.pkl", "rb") as f:
    d = pickle.load(f)
e = d[0]
#dvtest=d[1]
p, tp, d = get_demands(e)
pfc, tpfc, fc = get_forecasts(e)
timeseries = np.array(d).astype('float32')

# reshaping to get in the format (N rows, 1 Column )
# -1 means number of rows will be automatically decided by Puthon
timeseries = timeseries.reshape(-1,1)

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+lookback]  #dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return np.array(X), np.array(y)

# normalize the data to bring into common range of 0-1 to converse the model
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
timeseries = sc.fit_transform(timeseries)

# get X and y
X, y = create_dataset(timeseries, lookback=lookback)

# split data in test and training cases
train_size = int(len(y) * 0.80)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train.shape
y_train.shape

# reshaping to match RNN model input format
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Custom RMSE loss function
from tensorflow.keras import backend as K
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

from tensorflow.keras.callbacks import EarlyStopping
# Early stopping callback, to stop the model if its not improving after waiting for patience i.e 20 steps here
early_stopping = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)


"""
# CNN
- CNN visualize- https://poloclub.github.io/cnn-explainer/
"""

# Data processing

timestep = 30
X1= []
Y1=[]

raw_data=timeseries

for i in range(len(raw_data)- (timestep)):
    X1.append(raw_data[i:i+timestep])
    Y1.append(raw_data[i+timestep])

data=np.asanyarray(X1)
targets=np.asanyarray(Y1)

from sklearn.model_selection import train_test_split

# split into train/test
#X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)
# split data in test and training cases
train_size = int(len(y) * 0.80)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def optimize_cnn(trial):
    
    # Hyperparameters
    cnn_layers = trial.suggest_int('cnn_layers', 1, 3) 
    lstm_layers = trial.suggest_int('lstm_layers', 0, 3)
    lstm_units = [trial.suggest_categorical(f'lstm_units{i}', [64, 128]) for i in range(lstm_layers)]
    dropout = [trial.suggest_uniform(f'dropout{i}', 0.1, 0.5) for i in range(cnn_layers)]
    dropout1 = [trial.suggest_uniform(f'dropout1{i}', 0.1, 0.5) for i in range(lstm_layers)]
    filters_1 = [trial.suggest_int('filters_1{i}', 64, 128) for i in range(cnn_layers)]
    kernel_size_1 = [trial.suggest_int('kernel_size_1{i}', 5, 5) for i in range(cnn_layers)]
    strides_1 = [trial.suggest_int('strides_1{i}', 2, 2) for i in range(cnn_layers)]
    dense_units = trial.suggest_int('dense_units', 50,100)
    activation_function = trial.suggest_categorical('activation_function', ['relu', 'relu', 'relu'])
    # Build model
    model = Sequential()
    for i in range(cnn_layers):
    
    	# Restrict the combination of kernel_size and strides
        if kernel_size_1[i] > strides_1[i]:
            model.add(Conv1D(
                filters=filters_1[i],
                kernel_size=kernel_size_1[i],
                activation='relu',
                input_shape=(lookback, 1),
                padding='same',
                strides=strides_1[i],
            ))
        else:
            # Handle the case where kernel_size is not greater than strides (you can customize this based on your needs)
            model.add(Conv1D(
                filters=filters_1[i],
                kernel_size=kernel_size_1[i],
                activation='relu',
                input_shape=(lookback, 1),
                padding='same',
                strides=1,  # Default value or another appropriate value
            ))
            
        print(f"Conv1D Layer {i+1}:")
        print(f"  Kernel Size: {kernel_size_1[i]}")
        print(f"  Strides: {strides_1[i]}")
    	#print(f"Conv1D Layer {i+1}:")
        if model.output_shape[1] > 1:
            # Add try-except block to handle the exception
            try:
                model.add(MaxPooling1D(pool_size=2))
            except ValueError as e:
                print(f"Skipping pooling due to small dimensions: {e}")
        else:
            # Handle the case where dimensions are too small
            print("Skipping pooling due to small dimensions.")
        #model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout[i])) 

    for i in range(lstm_layers):    
        model.add(LSTM(units = lstm_units[i], return_sequences=True))
        model.add(Dropout(dropout1[i]))
    model.add(Dense(dense_units, activation=activation_function))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, callbacks=[early_stopping], validation_split=0.2 )
    ##################################
    # Evaluate the model on the test set
    #y_pred_scaled = model.predict(X_test)
    #y_pred = scaler.inverse_transform(y_pred_scaled)
    #y_test_orig = scaler.inverse_transform(y_test)
    ########################################
    # Get loss from the model
    rmse = history.history['loss'][-1]
    
    # New computer
    msee = model.evaluate(X_train, y_train, verbose=0)
    print('mse loss:', msee)
    
    return rmse

# CNN Optimization
study_cnn = optuna.create_study(direction='minimize')
study_cnn.optimize(optimize_cnn, n_trials=2)

# Get the best parameters for CNN
best_params_cnn = study_cnn.best_trial.params
print('Best CNN Parameters:', best_params_cnn)

# build CNN Model
model = Sequential()

for i in range(best_params_cnn['cnn_layers']):
    # Assuming 'lookback' is the number of time steps in the sequence
    model.add(Conv1D(filters = 128, kernel_size =  5, activation = 'relu', input_shape = (lookback,1), padding='same',strides=best_params_cnn['strides_1{i}']))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Dropout(best_params_cnn[f'dropout{i}']))
    if model.output_shape[1] > 1:
        # Add try-except block to handle the exception
        try:
            model.add(MaxPooling1D(pool_size=2))
        except ValueError as e:
            print(f"Skipping pooling due to small dimensions: {e}")
    else:
        # Handle the case where dimensions are too small
        print("Skipping pooling due to small dimensions.")
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(best_params_cnn[f'dropout{i}'])) 

for i in range(best_params_cnn['lstm_layers']):
    # First LSTM layer
    lstm_units_key = f'lstm_units'
    model.add(LSTM(units = best_params_cnn[f'lstm_units{i}'], return_sequences=True))
    model.add(Dropout(best_params_cnn[f'dropout1{i}'])) 
model.add(Flatten())
model.add(Dense(best_params_cnn['dense_units'], activation=best_params_cnn['activation_function']))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam',metrics=[tf.keras.metrics.RootMeanSquaredError()])

print('Model compiled:')
print(model.summary())

# Train the model
historyCNN = model.fit(X_train, y_train, epochs=5000, batch_size=32, verbose=0, callbacks=[early_stopping], validation_split=0.2 )

# store loss from CNN model
rmse = historyCNN.history['loss']
epochs = range(1, len(rmse) + 1)
plt.plot(epochs,rmse)
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.show()

# perform prediction.
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)
######################################
# Inverse transform the predictions to the original scale
# trainPredict = sc.inverse_transform(trainPredict)
# testPredict = sc.inverse_transform(testPredict)
# y = sc.inverse_transform(y)
# #####################################

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, testPredict))
print("Root Mean Squared Error on Test Set: %.4f" % rmse)

# Create train_plot and test_plot array and store respective prediction at their corresponding index.
tp = tp[lookback:]
dp = y[:,-1]

train_plot = np.ones_like(dp) * np.nan
test_plot = np.ones_like(dp) * np.nan
train_plot[:train_size] = trainPredict[:,-1]
test_plot[train_size:len(dp)] = testPredict[:,-1]

plt.plot(tp,dp,c='b', label = 'input data')
plt.plot(tp,train_plot, c='r',label='prediction on train set')
plt.plot(tp,test_plot, c='g',label='prediction on test set')
plt.xlabel('dfc')
plt.ylabel('amount')
plt.legend(loc = 'upper left')
plt.show()
print('Best CNN Parameters:', best_params_cnn)
"""
top_10_trials = sorted(study_cnn.trials, key=lambda x: x.value)[:10]

    # Print the parameters and objective values of the top 10 trials
for i, trial in enumerate(top_10_trials, 1):
        print(f"Rank {i}:")
        print(f"  Value: {trial.value}")
        print(f"  Params: {trial.params}")
        print("\n")
"""

