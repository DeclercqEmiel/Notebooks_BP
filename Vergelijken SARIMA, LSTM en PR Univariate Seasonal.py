#!/usr/bin/env python
# coding: utf-8

# In[40]:


# Voorspellen voor 1 jaar
# Of voorspellen op 3 jaar apart omdat voorspellen van 1 jaar misschien de algemene trend minder volgt
# SARIMA proberen anders gwn ARIMA


# In[41]:


import pandas as pd
import numpy as np
# matplotlib is the Python library for drawing diagrams
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# set the size of the diagrams
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15,5
import timeit
import warnings
from sklearn.model_selection import TimeSeriesSplit


# # General functions

# In[42]:


def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(36).mean()
    rolstd = timeseries.rolling(24).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
def full_graph(predicted, og_dataset, title):
    zerosArray = np.zeros(og_dataset.values.size-len(predicted.flatten()))
    cleanPrediction = pd.Series(np.concatenate((zerosArray,predicted))).replace(0,np.NaN)
    
    # plot
    plt.title(title)
    plt.plot(og_dataset.index, og_dataset.values,marker='o', color='blue',label='Actual values')
    plt.plot(og_dataset.index, cleanPrediction,marker='o', color='red',label='Last 2 year prediction')
    plt.ylim([0,20])
    plt.legend()

    plt.show()
    
def revert_diff(predicted_diff, og_data):
    last_value = og_data.iloc[-predicted_diff.size-1][0]
    predicted_actual = np.array([])
    for value_diff in predicted_diff:
        actual_value = last_value + value_diff
        predicted_actual = np.append(predicted_actual, actual_value)
        last_value = actual_value
    return predicted_actual

def revert_seasonal_diff_recursion(last_seasons_value, diff_value):
    return last_seasons_value + diff_value

def revert_diff_seasonal(predicted_diff, og_data):
    prediction_size = predicted_diff.size
    
    history = ts[:-prediction_size].values.flatten()
    for value_diff in predicted_diff[-prediction_size:]:
        new_value = revert_seasonal_diff_recursion(history[-12], value_diff)
        history = np.append(history,new_value)
    return history[-prediction_size:]


# ## Dataprep

# In[43]:


ts = pd.read_csv('./data/dataframe_monthly.csv', index_col=0, usecols=[0,1,3]).reset_index()


# In[44]:


ts


# In[45]:


ts['date'] = pd.to_datetime(ts['Month'].astype(str) + ts['Year'].astype(str), format='%m%Y', errors='ignore')


# In[46]:


ts


# In[47]:


ts = ts[['date','ice_extent']]
ts.set_index('date', inplace=True)
plt.plot(ts[['ice_extent']])


# In[48]:


test_stationarity(ts)


# ### Differencing

# In[49]:


ts_diff_seasonal = ts - ts.shift(12)
ts_diff_seasonal = ts_diff_seasonal.dropna()
test_stationarity(ts_diff_seasonal)


# In[50]:


ts_diff = ts - ts.shift(1)
ts_diff = ts_diff.dropna()
test_stationarity(ts_diff)


# ### Cross validation setup

# In[51]:


def display_cross_validation(dataset, n_splits):
    tscv = TimeSeriesSplit(n_splits = n_splits)
    
    for train_index, test_index in tscv.split(dataset):
        if train_index.size > 300:
            # initialize cross validation train and test sets
            cv_train, cv_test = dataset.iloc[train_index], dataset.iloc[test_index]

            print("TRAIN:", train_index.size) # visiualize cross_validation structure for reference
            print("TEST:", test_index.size)
            print()


# In[52]:


ts_diff = ts_diff[:-5] # need the -5 to get testsets for 24 months/2 years
display_cross_validation(ts_diff, 18)
tscv_diff = TimeSeriesSplit(n_splits = 18)


# In[53]:


display_cross_validation(ts_diff_seasonal, 18)
tscv_diff_seasonal = TimeSeriesSplit(n_splits = 18)


# # ARIMA

# ## random walk differencing

# ### Determine hyperparameters

# In[20]:


# ARIMA
from statsmodels.tsa.arima_model import ARIMA
import itertools
import warnings
import sys
from sklearn.metrics import mean_absolute_error



# Define the p, d and q parameters to take any value between 0 and 2
p = q = range(0, 5)
d = range(0,3)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
best_pdq = pdq
best_mean_mae = np.inf
warnings.filterwarnings("ignore") # specify to ignore warning messages
for param in pdq:
    print(param)
    try:   # some parametercombinations might lead to crash, so catch exceptions and continue
        maes = []
        for train_index, test_index in tscv_diff.split(ts_diff):
            if train_index.size > 300:
                # initialize cross validation train and test sets
                cv_train, cv_test = ts_diff.iloc[train_index], ts_diff.iloc[test_index]

                # build model
                model = ARIMA(cv_train, order=(param))
                model_fit = model.fit()

                # make predictions
                predictions =  model_fit.predict(start=len(cv_train), end=len(cv_train)+cv_test.size-1, dynamic=False)
                prediction_values = predictions.values
                true_values = cv_test.values
                # error calc
                #     print(true_values)
                #     print(predictions.values)
                maes.append(mean_absolute_error(true_values, prediction_values))

        
        mean_mae = np.mean(maes)
        print('MAE: ' + str(mean_mae))    

        if mean_mae < best_mean_mae:
            best_mean_mae = mean_mae
            best_maes = maes
            best_pdq = param
            best_predictions = prediction_values
    except Exception as e:
        print(e)
        continue
   
print()
print('Best MAE = ' + str(best_mean_mae))
print(best_pdq)


# best range(0,10)
# Best MAE = 0.23763938034311669
# (8, 0, 9)
# Wall time: 1h 32min 14s


# In[54]:


# best_pdq = (8, 0, 9) # range 10
best_pdq = (3,0,3)


# ### Final result

# In[55]:


from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error


start_time = timeit.default_timer()

warnings.filterwarnings("ignore") # specify to ignore warning messages

print("-------")

maes = []

for train_index, test_index in tscv_diff.split(ts_diff):
    if train_index.size > 300:
        # initialize cross validation train and test sets
        cv_train, cv_test = ts_diff.iloc[train_index], ts_diff.iloc[test_index]

        # build model
        model = ARIMA(cv_train, order=(best_pdq))
        model_fit = model.fit()

        # make predictions
        predictions =  model_fit.predict(start=len(cv_train), end=len(cv_train)+cv_test.size-1, dynamic=False)
        prediction_values = predictions.values
        true_values = cv_test.values

        # error calc
    #     print(true_values)
    #     print(predictions.values)
        maes.append(mean_absolute_error(true_values, prediction_values))

        print("I",end="")

time_ARIMA = timeit.default_timer() - start_time
mae_mean = np.mean(maes)
MAE_ARIMA = mae_mean
last_MAE_ARIMA = maes[-1]
last_prediction_ARIMA = prediction_values

print()
print('Mean MAE: %.3f x 1 000 000 km\u00b2' % MAE_ARIMA)
print('MAE of last prediction: %.3f x 1 000 000 km\u00b2' % last_MAE_ARIMA)
print('Execution time: %.3f seconds' % time_ARIMA)

reverted_prediction_values = revert_diff(last_prediction_ARIMA, ts[:-5])
full_graph(reverted_prediction_values, ts[:-5],'Last 2 year prediction ARIMA with regular differencing')
print(maes)


# ## Seasonal differencing

# ### Determine hyperparameters

# In[22]:


# ARIMA
from statsmodels.tsa.arima_model import ARIMA
import itertools
import warnings
import sys
from sklearn.metrics import mean_absolute_error



# Define the p, d and q parameters to take any value between 0 and 2
p = q = range(0, 5)
d = range(0,3)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
best_pdq = pdq
best_mean_mae = np.inf
warnings.filterwarnings("ignore") # specify to ignore warning messages
for param in pdq:
    print(param)
    try:   # some parametercombinations might lead to crash, so catch exceptions and continue
        maes = []
        for train_index, test_index in tscv_diff_seasonal.split(ts_diff_seasonal):
            if train_index.size > 300:
                # initialize cross validation train and test sets
                cv_train, cv_test = ts_diff_seasonal.iloc[train_index], ts_diff_seasonal.iloc[test_index]

                # build model
                model = ARIMA(cv_train, order=(param))
                model_fit = model.fit()

                # make predictions
                predictions =  model_fit.predict(start=len(cv_train), end=len(cv_train)+cv_test.size-1, dynamic=False)
                prediction_values = predictions.values
                true_values = cv_test.values
                # error calc
                #     print(true_values)
                #     print(predictions.values)
                maes.append(mean_absolute_error(true_values, prediction_values))

        
        mean_mae = np.mean(maes)
        print('MAE: ' + str(mean_mae))    

        if mean_mae < best_mean_mae:
            best_mean_mae = mean_mae
            best_maes = maes
            best_pdq = param
            best_predictions = prediction_values
    except Exception as e:
        print(e)
        continue
   
print()
print('Best MAE = ' + str(best_mean_mae))
print(best_pdq)



# ### Final result

# In[56]:


best_pdq=(3,0,3) # pq range 4
# best_pdq=(8, 0, 6) # pq range 10


# In[57]:


from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error


start_time = timeit.default_timer()

warnings.filterwarnings("ignore") # specify to ignore warning messages

print("------")

maes = []

for train_index, test_index in tscv_diff_seasonal.split(ts_diff_seasonal):
    if train_index.size > 300:
        # initialize cross validation train and test sets
        cv_train, cv_test = ts_diff_seasonal.iloc[train_index], ts_diff_seasonal.iloc[test_index]

        # build model
        model = ARIMA(cv_train, order=(best_pdq))
        model_fit = model.fit()

        # make predictions
        predictions =  model_fit.predict(start=len(cv_train), end=len(cv_train)+cv_test.size-1, dynamic=False)
        prediction_values = predictions.values
        true_values = cv_test.values

        # error calc
    #     print(true_values)
    #     print(predictions.values)
        maes.append(mean_absolute_error(true_values, prediction_values))

        print("I",end="")

time_ARIMA_seasonal = timeit.default_timer() - start_time
mae_mean = np.mean(maes)
MAE_ARIMA_seasonal = mae_mean
last_MAE_ARIMA_seasonal = maes[-1]
last_prediction_ARIMA_seasonal = prediction_values

print()
print('Mean MAE: %.3f x 1 000 000 km\u00b2' % MAE_ARIMA_seasonal)
print('MAE of last prediction: %.3f x 1 000 000 km\u00b2' % last_MAE_ARIMA_seasonal)
print('Execution time: %.3f seconds' % time_ARIMA_seasonal)

reverted_prediction_values = revert_diff_seasonal(last_prediction_ARIMA_seasonal, ts)
full_graph(reverted_prediction_values, ts,'Last 2 year prediction ARIMA with seasonal differencing')
print(maes)


# # SARIMAX

# ## Random walk differencing

# In[76]:


# SARIMAX

import itertools
import warnings
import sys
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Define the p, d and q parameters to take any value between 0 and 2
p = q = P = D = Q = range(0, 3)
d = D = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdqPDQ = list(itertools.product(p, d, q , P, D, Q))
best_pdqPDQ = pdqPDQ
best_mean_mae = np.inf
warnings.filterwarnings("ignore") # specify to ignore warning messages
for param in pdqPDQ:
    print(param)
    try:   # some parametercombinations might lead to crash, so catch exceptions and continue
        maes = []
        for train_index, test_index in tscv_diff.split(ts_diff):
            if train_index.size > 300:
                # initialize cross validation train and test sets
                cv_train, cv_test = ts_diff.iloc[train_index], ts_diff.iloc[test_index]

                # build model
                model = SARIMAX(cv_train, 
                order=param[:3], 
                seasonal_order=(12,)+param[3:])
                model_fit = model.fit()

                # make predictions
#                 predictions =  model_fit.predict(start=len(cv_train), end=len(cv_train)+cv_test.size-1, dynamic=False)
                predictions =  model_fit.forecast(steps=24)
                prediction_values = predictions.values
                true_values = cv_test.values
                # error calc
                #     print(true_values)
                #     print(predictions.values)
                maes.append(mean_absolute_error(true_values, prediction_values))

        
        mean_mae = np.mean(maes)
        print('MAE: ' + str(mean_mae))    

        if mean_mae < best_mean_mae:
            best_mean_mae = mean_mae
            best_maes = maes
            best_pdqPDQ = param
            best_predictions = prediction_values
    except Exception as e:
        print(e)
        continue

print(best_predictions.size)
print(data_test.index.size)

        
predictions_df = pd.DataFrame(best_predictions).set_index(keys=data_test.index)

# plot
print()
print('Best MAE = ' + str(best_mean_mae))
print(best_pdqPDQ)
plt.plot(data_test,color='blue')
plt.plot(predictions_df, color='red')
plt.show()

# best range(0,2):
# Best MAE = 0.2524024604742092
# (1, 0, 1, 1, 1, 1)
# Wall time: 14min 50s

# best range(0,2):

# Best MAE = 0.22780663319275937
# (1, 0, 2, 0, 1, 2)
# Wall time: 2h 6min 26s


# In[58]:


best_pdqPDQ = (1, 0, 2, 0, 1, 2)


# In[59]:


from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


start_time = timeit.default_timer()

warnings.filterwarnings("ignore") # specify to ignore warning messages

print("-------")

maes = []

for train_index, test_index in tscv_diff.split(ts_diff):
    if train_index.size > 300:

        # initialize cross validation train and test sets
        cv_train, cv_test = ts_diff.iloc[train_index], ts_diff.iloc[test_index]

        # build model
        model = SARIMAX(cv_train, 
                order=best_pdqPDQ[:3], 
                seasonal_order=(best_pdqPDQ[3:]+(12,)))
        model_fit = model.fit()

        # make predictions
        predictions =  model_fit.predict(start=len(cv_train), end=len(cv_train)+cv_test.size-1, dynamic=False)
        prediction_values = predictions.values
        true_values = cv_test.values

        # error calc
    #     print(true_values)
    #     print(predictions.values)
        maes.append(mean_absolute_error(true_values, prediction_values))
        print("I",end="")



time_SARIMA = timeit.default_timer() - start_time
mae_mean = np.mean(maes)
MAE_SARIMA = mae_mean
last_MAE_SARIMA = maes[-1]
last_prediction_SARIMA = prediction_values


print()
print('Mean MAE: %.3f x 1 000 000 km\u00b2' % MAE_SARIMA)
print('MAE of last prediction: %.3f x 1 000 000 km\u00b2' % last_MAE_SARIMA)
print('Execution time: %.3f seconds' % time_SARIMA)

reverted_prediction_values = revert_diff(last_prediction_SARIMA, ts[:-5])
full_graph(reverted_prediction_values, ts[:-5],'Last 2 year prediction SARIMAX')
print(maes)


# ## Seasonal differencing

# ### Determine hyperparameters

# In[79]:


# SARIMAX

import itertools
import warnings
import sys
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Define the p, d and q parameters to take any value between 0 and 2
p = q = P = D = Q = range(0, 3)
d = D = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdqPDQ = list(itertools.product(p, d, q , P, D, Q))
best_pdqPDQ = pdqPDQ
best_mean_mae = np.inf
warnings.filterwarnings("ignore") # specify to ignore warning messages
for param in pdqPDQ:
    print(param)
    try:   # some parametercombinations might lead to crash, so catch exceptions and continue
        maes = []
        for train_index, test_index in tscv_diff_seasonal.split(ts_diff_seasonal):
            if train_index.size > 300:
                # initialize cross validation train and test sets
                cv_train, cv_test = ts_diff_seasonal.iloc[train_index], ts_diff_seasonal.iloc[test_index]

                # build model
                model = SARIMAX(cv_train, 
                order=param[:3], 
                seasonal_order=(12,)+param[3:])
                model_fit = model.fit()

                # make predictions
#                 predictions =  model_fit.predict(start=len(cv_train), end=len(cv_train)+cv_test.size-1, dynamic=False)
                predictions =  model_fit.forecast(steps=24)
                prediction_values = predictions.values
                true_values = cv_test.values
                # error calc
                #     print(true_values)
                #     print(predictions.values)
                maes.append(mean_absolute_error(true_values, prediction_values))

        
        mean_mae = np.mean(maes)
        print('MAE: ' + str(mean_mae))    

        if mean_mae < best_mean_mae:
            best_mean_mae = mean_mae
            best_maes = maes
            best_pdqPDQ = param
            best_predictions = prediction_values
    except Exception as e:
        print(e)
        continue

print(best_predictions.size)
print(data_test.index.size)

        
predictions_df = pd.DataFrame(best_predictions).set_index(keys=data_test.index)

# plot
print()
print('Best MAE = ' + str(best_mean_mae))
print(best_pdqPDQ)
plt.plot(data_test,color='blue')
plt.plot(predictions_df, color='red')
plt.show()

# best range(0,2):
# Best MAE = 0.2524024604742092
# (1, 0, 1, 1, 1, 1)
# Wall time: 14min 50s

# best range(0,2):

# Best MAE = 0.22780663319275937
# (1, 0, 2, 0, 1, 2)
# Wall time: 2h 6min 26s


# In[60]:


best_pdqPDQ = (1, 0, 2, 0, 1, 2)


# In[61]:


from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


start_time = timeit.default_timer()

warnings.filterwarnings("ignore") # specify to ignore warning messages

print("------")

maes = []

for train_index, test_index in tscv_diff_seasonal.split(ts_diff_seasonal):
    if train_index.size > 300:

        # initialize cross validation train and test sets
        cv_train, cv_test = ts_diff_seasonal.iloc[train_index], ts_diff_seasonal.iloc[test_index]

        # build model
        model = SARIMAX(cv_train, 
                order=best_pdqPDQ[:3], 
                seasonal_order=(best_pdqPDQ[3:]+(12,)))
        model_fit = model.fit()

        # make predictions
        predictions =  model_fit.predict(start=len(cv_train), end=len(cv_train)+cv_test.size-1, dynamic=False)
        prediction_values = predictions.values
        true_values = cv_test.values

        # error calc
    #     print(true_values)
    #     print(predictions.values)
        maes.append(mean_absolute_error(true_values, prediction_values))
        print("I",end="")



time_SARIMA_seasonal = timeit.default_timer() - start_time
mae_mean = np.mean(maes)
MAE_SARIMA_seasonal = mae_mean
last_MAE_SARIMA_seasonal = maes[-1]
last_prediction_SARIMA_seasonal = prediction_values


print()
print('Mean MAE: %.3f x 1 000 000 km\u00b2' % MAE_SARIMA_seasonal)
print('MAE of last prediction: %.3f x 1 000 000 km\u00b2' % last_MAE_SARIMA_seasonal)
print('Execution time: %.3f seconds' % time_SARIMA_seasonal)

reverted_prediction_values = revert_diff_seasonal(last_prediction_SARIMA_seasonal, ts)
full_graph(reverted_prediction_values, ts,'Last 2 year prediction SARIMAX')
print(maes)


# # LSTM

# In[62]:


from keras.layers import Dropout
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
            
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
 
def build_model(raw_seq, n_steps_in, n_steps_out, n_features, n_neurons, dropout, batch_s):
    
    # split into samples
    X, y = split_sequence(raw_seq.values.flatten(), n_steps_in, n_steps_out)
    
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    # define model
    model = Sequential()
    model.add(LSTM(n_neurons, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mae')
    
    # fit model
    model.fit(X, y, batch_size=batch_s, epochs=100, verbose=0)
    
    return model


def predict(x_input, model, n_features):
    n_features = 1
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)
    return yhat


# ## Regular differencing 

# In[32]:


import timeit
import tensorflow as tf


start_time = timeit.default_timer()

# Disabled tf warning because of visual clutter
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# constant variables
n_steps_in = 24
n_steps_out = 24
n_features  = 1
maes = []
global_maes = []

# optimizable variables
n_neurons_array = [1,10,20]
dropout_array = [0,0.99]
batch_size_array = [1,2,8]

# n_neurons_array = [1,20]
# dropout_array = [0]
# batch_size_array = [1,8]



# initialize values
best_MAE = 100
best_n_neurons = 0
best_activation = 'none'
best_dropout = 0
best_batch_size = 0

for n_neurons in n_neurons_array:
    for dropout in dropout_array:
        for batch_size in batch_size_array:
            print("-----")
#             tscv = TimeSeriesSplit(n_splits = 17)
            for train_index, test_index in tscv_diff.split(ts_diff): 
                if train_index.size > 300:  
                    # initialize cross validation train and test sets
                    y_train, y_test = ts_diff.iloc[train_index], ts_diff.iloc[test_index]

                    # build model
                    lstm_model = build_model(y_train, n_steps_in, n_steps_out, n_features, n_neurons, dropout, batch_size)

                    # make predictions
                    x_input = array(y_test)
                    y_predicted = predict(x_input, lstm_model, n_features).flatten()
                    y_actual = y_test.values

                    # error calc
                    maes.append(mean_absolute_error(y_actual, y_predicted))

                    print("I",end="")

                    # last actual prediction 
                    last_prediction_LSTM = y_predicted

            time_LSTM = timeit.default_timer() - start_time
            MAE_LSTM = np.mean(maes)
            last_MAE_LSTM = maes[-1]
            global_maes.append(MAE_LSTM)

            if best_MAE > MAE_LSTM:
                best_n_neurons = n_neurons
                best_dropout = dropout
                best_batch_size = batch_size
                best_MAE = MAE_LSTM

            print()
            print(n_neurons)
            print(dropout)
            print(batch_size)
            print(MAE_LSTM)
            print()    

print('Best:')
print('N neurons')
print(best_n_neurons)
print('Dropout rate')
print(best_dropout)
print('Batch size')
print(best_batch_size)
print('MAE')
print(best_MAE)
plt.bar(range(0,len(global_maes)), global_maes)


# In[63]:


best_n_neurons, best_dropout, best_batch_size = 1, 0, 1 # actual best params


# In[64]:


import timeit
import tensorflow as tf


start_time = timeit.default_timer()

# Disabled tf warning because of visual clutter
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# constant variables
n_steps_in = 24
n_steps_out = 24
n_features  = 1
maes = []
global_maes = []

print("------")
tscv = TimeSeriesSplit(n_splits = 18)
for train_index, test_index in tscv.split(ts_diff): 
    if train_index.size > 300:  
        # initialize cross validation train and test sets
        y_train, y_test = ts_diff.iloc[train_index], ts_diff.iloc[test_index]

        # build model
        lstm_model = build_model(y_train, n_steps_in, n_steps_out, n_features, best_n_neurons, best_dropout, best_batch_size)

        # make predictions
        x_input = array(y_test)
        y_predicted = predict(x_input, lstm_model, n_features).flatten()
        y_actual = y_test.values

        # error calc
        maes.append(mean_absolute_error(y_actual, y_predicted))

        print("I",end="")

time_LSTM = timeit.default_timer() - start_time
MAE_LSTM = np.mean(maes)
last_MAE_LSTM = maes[-1]
global_maes.append(MAE_LSTM)
last_prediction_LSTM = y_predicted

# print('Best:')
# print('N neurons')
# print(best_n_neurons)
# print('Dropout rate')
# print(best_dropout)
# print('Batch size')
# print(best_batch_size)
# print('MAE')
# print(best_MAE)
# plt.bar(range(0,len(global_maes)), global_maes)

# time_ARIMA = timeit.default_timer() - start_time
# mae_mean = np.mean(maes)
# MAE_ARIMA = mae_mean
# last_MAE_ARIMA = maes[-1]

print()
print('Mean MAE: %.3f x 1 000 000 km\u00b2' % MAE_LSTM)
print('MAE of last prediction: %.3f x 1 000 000 km\u00b2' % last_MAE_LSTM)
print('Execution time: %.3f seconds' % time_LSTM)

reverted_prediction_values = revert_diff_seasonal(last_prediction_LSTM, ts)
full_graph(reverted_prediction_values, ts,'Last 2 year prediction LSTM random walk differencing')

print(maes)


# ## Seasonal differencing

# In[99]:


# initialize values
best_n_neurons = 1
best_dropout = 0
best_batch_size = 1


# In[100]:


import timeit
import tensorflow as tf


start_time = timeit.default_timer()

# Disabled tf warning because of visual clutter
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# constant variables
n_steps_in = 24
n_steps_out = 24
n_features  = 1
maes = []
global_maes = []

# optimizable variables
n_neurons_array = [1,5,10,20]
dropout_array = [0,0.5,0.99]
batch_size_array = [1,2,4,8]
## set#2
# n_neurons_array = [1,20]2
# dropout_array = [0]
# batch_size_array = [1,8]



# initialize values
best_MAE = 100
best_n_neurons = 0
best_activation = 'none'
best_dropout = 0
best_batch_size = 0

for n_neurons in n_neurons_array:
    for dropout in dropout_array:
        for batch_size in batch_size_array:
            print("------")
#             tscv = TimeSeriesSplit(n_splits = 17)
            for train_index, test_index in tscv_diff.split(ts_diff_seasonal): 
                if train_index.size > 300:  
                    # initialize cross validation train and test sets
                    y_train, y_test = ts_diff_seasonal.iloc[train_index], ts_diff_seasonal.iloc[test_index]

                    # build model
                    lstm_model = build_model(y_train, n_steps_in, n_steps_out, n_features, n_neurons, dropout, batch_size)

                    # make predictions
                    x_input = array(y_test)
                    y_predicted = predict(x_input, lstm_model, n_features).flatten()
                    y_actual = y_test.values

                    # error calc
                    maes.append(mean_absolute_error(y_actual, y_predicted))

                    print("I",end="")

                    # last actual prediction 
                    last_prediction_LSTM = y_predicted

            time_LSTM = timeit.default_timer() - start_time
            MAE_LSTM = np.mean(maes)
            last_MAE_LSTM = maes[-1]
            global_maes.append(MAE_LSTM)

            if best_MAE > MAE_LSTM:
                best_n_neurons = n_neurons
                best_dropout = dropout
                best_batch_size = batch_size
                best_MAE = MAE_LSTM

            print()
            print(n_neurons)
            print(dropout)
            print(batch_size)
            print(MAE_LSTM)
            print()    

print('Best:')
print('N neurons')
print(best_n_neurons)
print('Dropout rate')
print(best_dropout)
print('Batch size')
print(best_batch_size)
print('MAE')
print(best_MAE)
plt.bar(range(0,len(global_maes)), global_maes)


# In[65]:


best_n_neurons, best_dropout, best_batch_size = 1, 0.99, 8 # ran for 5 hours


# In[66]:


import timeit
import tensorflow as tf


start_time = timeit.default_timer()

# Disabled tf warning because of visual clutter
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# constant variables
n_steps_in = 24
n_steps_out = 24
n_features  = 1
maes = []
global_maes = []

print("------")
tscv = TimeSeriesSplit(n_splits = 18)
for train_index, test_index in tscv.split(ts_diff_seasonal): 
    if train_index.size > 300:  
        # initialize cross validation train and test sets
        y_train, y_test = ts_diff_seasonal.iloc[train_index], ts_diff_seasonal.iloc[test_index]

        # build model
        lstm_model = build_model(y_train, n_steps_in, n_steps_out, n_features, best_n_neurons, best_dropout, best_batch_size)

        # make predictions
        x_input = array(y_test)
        y_predicted = predict(x_input, lstm_model, n_features).flatten()
        y_actual = y_test.values

        # error calc
        maes.append(mean_absolute_error(y_actual, y_predicted))

        print("I",end="")

        

time_LSTM_seasonal = timeit.default_timer() - start_time
MAE_LSTM_seasonal = np.mean(maes)
last_MAE_LSTM_seasonal = maes[-1]
global_maes.append(MAE_LSTM_seasonal)
last_prediction_LSTM_seasonal = y_predicted
# print('Best:')
# print('N neurons')
# print(best_n_neurons)
# print('Dropout rate')
# print(best_dropout)
# print('Batch size')
# print(best_batch_size)
# print('MAE')
# print(best_MAE)
# plt.bar(range(0,len(global_maes)), global_maes)

# time_ARIMA = timeit.default_timer() - start_time
# mae_mean = np.mean(maes)
# MAE_ARIMA = mae_mean
# last_MAE_ARIMA = maes[-1]

print()
print('Mean MAE: %.3f x 1 000 000 km\u00b2' % MAE_LSTM_seasonal)
print('MAE of last prediction: %.3f x 1 000 000 km\u00b2' % last_MAE_LSTM_seasonal)
print('Execution time: %.3f seconds' % time_LSTM_seasonal)

reverted_prediction_values = revert_diff_seasonal(last_prediction_LSTM_seasonal, ts)
full_graph(reverted_prediction_values, ts,'Last 2 year prediction LSTM Seasonal')

print(maes)


# # Prophet

# In[67]:


ts_diff.reset_index()


# In[68]:


# formatting dataframe
ts_formated_prophet = ts_diff.reset_index().rename(columns = {'date' : 'ds', 'ice_extent' : 'y'})
ts_formated_prophet['ds'] = pd.DataFrame(pd.to_datetime(ts_formated_prophet['ds'].astype(str), format='%Y-%m-%d'))


# In[69]:


# initialize TimeSeriesSplit object
tscv_prophet = TimeSeriesSplit(n_splits = 18)

# loop trough all split time series that have a trainingsset with more than 20 values
for train_index, test_index in tscv_prophet.split(ts_formated_prophet):
    if train_index.size > 300:

        # initialize cross validation train and test sets
        cv_train, cv_test = ts_diff.iloc[train_index], ts_diff.iloc[test_index]
        
         # visiualize cross_validation structure for reference
        print("TRAIN:", train_index.size)
        print("TEST:", test_index.size)
        print()


# In[39]:


# Python
import itertools
import numpy as np
import pandas as pd
from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error


# define dataframe
df = ts_formated_prophet

param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5, 0.75, 1],
    'seasonality_prior_scale': [0.001, 0.01, 0.1, 1, 2, 5, 10],
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

# initialize variables
maes = []  
global_maes = []
best_MAE_prophet = np.inf

# Use cross validation to evaluate all parameters
for params in all_params:

    # loop trough all split time series that have a trainingsset with more than 20 values
    for train_index, test_index in tscv_prophet.split(ts_formated_prophet):    
        if train_index.size > 300:  
            
            # initialize cross validation train and test sets
            train  = ts_formated_prophet.iloc[train_index]
            y_test = ts_formated_prophet.iloc[test_index][['y']].values.flatten()
            X_test = ts_formated_prophet.iloc[test_index][['ds']]

            # Fit model with given params
            model = Prophet(**params, weekly_seasonality=False, daily_seasonality=False)
            model = model.fit(train)
            
            # make predictions
            forecast = model.predict(X_test)
            y_pred = forecast['yhat'].values
            
            # last actual prediction 
            last_prediction_prophet = y_pred
            
            # error calculation this part of the cross validation
            maes.append(mean_absolute_error(y_test, y_pred))
            
    # error calculation for this parameter combination
    MAE_prophet = np.mean(maes)
    last_MAE_prophet = maes[-1]
    global_maes.append(MAE_prophet)
    
    # logging
    print('changepoint_prior_scale: ' + str(params['changepoint_prior_scale']))
    print('seasonality_prior_scale: ' + str(params['seasonality_prior_scale']))
    print(MAE_prophet)
    
    # store parameters resulting in the lowest mean MAE
    if best_MAE_prophet > MAE_prophet:
        best_params = params
        best_MAE_prophet = MAE_prophet

# log optimal result          
print('changepoint_prior_scale: ' + str(best_params['changepoint_prior_scale']))
print('seasonality_prior_scale: ' + str(best_params['seasonality_prior_scale']))
print(best_MAE_prophet)
            


# In[77]:


best_params = {'changepoint_prior_scale': 0.001, 'seasonality_prior_scale': 10}


# In[78]:


from fbprophet import Prophet
# Disabled tf warning because of clutter
warnings.filterwarnings("ignore") # specify to ignore warning messages

start_time = timeit.default_timer()

# initialize variables
maes = []

for train_index, test_index in tscv_prophet.split(ts_formated_prophet):
    if train_index.size > 300:  
        
        # initialize cross validation train and test sets
        train  = ts_formated_prophet.iloc[train_index]
        y_test = ts_formated_prophet.iloc[test_index][['y']].values.flatten()
        X_test = ts_formated_prophet.iloc[test_index][['ds']]

        # build model
        model = Prophet(**best_params, weekly_seasonality=False, daily_seasonality=False)
        model.fit(train)

        # make predictions
        forecast = model.predict(X_test)
        y_pred = forecast['yhat'].values

        # error calc
        maes.append(mean_absolute_error(y_test, y_pred))

        


# store results
time_Prophet = timeit.default_timer() - start_time
MAE_Prophet = np.mean(maes)
last_MAE_Prophet = maes[-1]
last_prediction_prophet = y_pred

# visualize results
print()
print('Mean MAE: %.3f x 1 000 000 km\u00b2' % MAE_Prophet)
print('MAE of last prediction: %.3f x 1 000 000 km\u00b2' % last_MAE_Prophet)
print('Execution time: %.3f seconds' % time_Prophet)

reverted_prediction_values = revert_diff_seasonal(last_prediction_prophet, ts)
full_graph(reverted_prediction_values, ts, "Last 2 year prediction prophet")
print('Mean average errors')
print(maes)


# In[ ]:


Mean MAE: 0.250 x 1 000 000 km²


# # Evaluation

# In[79]:


# formatting
results = [[MAE_ARIMA, time_ARIMA, last_MAE_ARIMA],
           [MAE_ARIMA_seasonal, time_ARIMA_seasonal, last_MAE_ARIMA_seasonal],
           [MAE_SARIMA, time_SARIMA, last_MAE_SARIMA],
           [MAE_SARIMA_seasonal, time_SARIMA_seasonal, last_MAE_SARIMA_seasonal],
           [MAE_LSTM, time_LSTM, last_MAE_LSTM],
           [MAE_LSTM_seasonal, time_LSTM_seasonal, last_MAE_LSTM_seasonal],
           [MAE_Prophet, time_Prophet, last_MAE_Prophet]]

# display results
results = pd.DataFrame(results, columns=['Mean MAE (x 1 000 000 km\u00b2)','Execution time (s)','Last MAE (x 1 000 000 km\u00b2)']
             ,index=['ARIMA','ARIMA_seasonal_differencing','SARIMA','SARIMA_seasonal_differncing','LSTM','LSTM_seasonal_differencing','Prophet']).round(decimals=3)
results


# In[84]:


reverted_prediction_values = revert_diff(last_prediction_ARIMA, ts[:-5])
full_graph(reverted_prediction_values, ts[:-5],'Last prediction ARIMA')

reverted_prediction_values = revert_diff_seasonal(last_prediction_ARIMA_seasonal, ts)
full_graph(reverted_prediction_values, ts,'Last prediction ARIMA with seasonal differencing')

reverted_prediction_values = revert_diff(last_prediction_SARIMA, ts[:-5])
full_graph(reverted_prediction_values, ts[:-5],'Last prediction SARIMAX')

reverted_prediction_values = revert_diff_seasonal(last_prediction_SARIMA_seasonal, ts)
full_graph(reverted_prediction_values, ts,'Last prediction SARIMAX')


# In[83]:



reverted_prediction_values = revert_diff_seasonal(last_prediction_LSTM, ts)
full_graph(reverted_prediction_values, ts,'Last prediction LSTM')

reverted_prediction_values = revert_diff_seasonal(last_prediction_LSTM_seasonal, ts)
full_graph(reverted_prediction_values, ts,'Last prediction LSTM Seasonal')

reverted_prediction_values = revert_diff_seasonal(last_prediction_prophet, ts)
full_graph(reverted_prediction_values, ts, "Last 2 year prediction Prophet")

