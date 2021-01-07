#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
# matplotlib is the Python library for drawing diagrams
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# set the size of the diagrams
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15,5
from sklearn.model_selection import TimeSeriesSplit


# # General functions

# In[100]:


reverted_prediction_values


# In[104]:


ts[['ice_extent']].max()


# In[119]:





# In[112]:


plt.plot(last_predictions_VARMAX_seasonal)


# In[162]:



# reverted_prediction_values = revert_diff_seasonal(last_predictions_VARMAX_seasonal, ts[['ice_extent']])
full_graph_seasonal(last_predictions_VARMAX_seasonal, 'Last 2 year prediction ARIMA with seasonal differencing')


# In[171]:


def test_stationarity(timeseries, title):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
#     rolstd = timeseries.rolling(24).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
#     std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation from column: ' + title)
    plt.show(block=False)
    
def full_graph(predicted_diff, title):
    dataset_ie = ts[['ice_extent']][:-5]
    predictionsArray = np.asarray(revert_diff(predicted_diff, dataset_ie))
    zerosArray = np.zeros(dataset_ie.values.size-len(predictionsArray.flatten()))
    cleanPrediction = pd.Series(np.concatenate((zerosArray,predictionsArray))).replace(0,np.NaN)
    
    # plot
    plt.title(title)
    plt.plot(dataset_ie.index, dataset_ie.values,marker='o', color='blue',label='Actual values')
    plt.plot(dataset_ie.index, cleanPrediction,marker='o', color='red',label='Last 2 year prediction')
    plt.ylim([0,20])
    plt.legend()

    plt.show()
    
def full_graph_seasonal(predicted, title):
    predicted = predicted.flatten() + ts[['ice_extent']].values.flatten()[-48:-24]
    zerosArray = np.zeros(ts[['ice_extent']].values.size-len(predicted.flatten()))
    cleanPrediction = pd.Series(np.concatenate((zerosArray,predicted.flatten()))).replace(0,np.NaN)
    
    # plot
    plt.title(title)
    plt.plot(ts.index, ts[['ice_extent']].values,marker='o', color='blue',label='Actual values')
    plt.plot(ts.index, cleanPrediction,marker='o', color='red',label='Last 2 year prediction')
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


# # Dataprep

# In[56]:


ts = pd.read_csv('./data/dataframe_monthly.csv', index_col=0).reset_index()


# In[57]:


ts


# In[58]:


ts['date'] = pd.to_datetime(ts['Month'].astype(str) + ts['Year'].astype(str), format='%m%Y', errors='ignore')


# In[59]:


ts


# In[60]:


ts = ts[['date','ice_extent','mean_temp']]
ts.set_index('date', inplace=True)
plt.plot(ts[['ice_extent']])
plt.title('ice_extent')
plt.show()
plt.plot(ts[['mean_temp']])
plt.title('mean_temp')


# In[61]:


test_stationarity(ts[['ice_extent']], 'ice_extent')
test_stationarity(ts[['mean_temp']], 'mean_temp')


# # Differencing

# In[62]:


ts_diff = ts - ts.shift(1)
ts_diff = ts_diff.dropna()
test_stationarity(ts_diff[['ice_extent']], 'ice_extent')
test_stationarity(ts_diff[['mean_temp']], 'mean_temp')


# In[63]:


ts_diff_seasonal = ts - ts.shift(12)
ts_diff_seasonal = ts_diff_seasonal.dropna()
test_stationarity(ts_diff_seasonal[['ice_extent']], 'ice_extent')
test_stationarity(ts_diff_seasonal[['mean_temp']], 'mean_temp')


# In[64]:


tscv = TimeSeriesSplit(n_splits = 18)
dataset = ts_diff[:-5] # need the -5 to get testsets for 24 months/2 years

for train_index, test_index in tscv.split(dataset):
    if train_index.size > 300:

        # initialize cross validation train and test sets
        cv_train, cv_test = dataset.iloc[train_index], dataset.iloc[test_index]

        print("TRAIN:", train_index.size) # visiualize cross_validation structure for reference
        print("TEST:", test_index.size)
        print()


# In[65]:


tscv = TimeSeriesSplit(n_splits = 18)

for train_index, test_index in tscv.split(ts_diff_seasonal):
    if train_index.size > 300:

        # initialize cross validation train and test sets
        cv_train, cv_test = ts_diff_seasonal.iloc[train_index], ts_diff_seasonal.iloc[test_index]

        print("TRAIN:", train_index.size) # visiualize cross_validation structure for reference
        print("TEST:", test_index.size)
        print()


# In[66]:


# og_data = dataset[:-6]
# seasonal_entries = 12
# n_sets = int(og_data.shape[0]/seasonal_entries)
# split_sets = np.array_split(dataset[:-6], n_sets)
# ts_diff = pd.DataFrame(columns = og_data.columns)

# i = 0 

# for year in split_sets[:-1]:
#     ## swap 0 and 1 around
#     # take difference
#     diff = split_sets[i+1].reset_index() - split_sets[i].reset_index()
#     diff = diff.iloc[:,1:]

#     # append to dataframe
#     ts_diff = ts_diff.append(diff[['ice_extent','mean_temp']], ignore_index=True)
#     i = i+1


# In[67]:


# wait 
ts_seasonal_diff = (dataset - dataset.shift(12)).dropna()


# ## VARMAX

# ## Random walk differencing

# In[53]:


from statsmodels.tsa.statespace.varmax import VARMAX
import itertools
import warnings
import sys
from sklearn.metrics import mean_absolute_error



# Define the p, d and q parameters to take any value between 0 and 2
p = q = range(0, 5)

# Generate all different combinations of p, q and q triplets
pq = list(itertools.product(p, q))
best_pq = pq
best_mean_mae = np.inf
warnings.filterwarnings("ignore") # specify to ignore warning messages
for param in pq:
    print(param)
    try:   # some parametercombinations might lead to crash, so catch exceptions and continue
        maes = []
        for train_index, test_index in tscv.split(dataset):
            if train_index.size > 300:
                # initialize cross validation train and test sets
                cv_train, cv_test = dataset.iloc[train_index], dataset.iloc[test_index]

                # build model
                model = VARMAX(cv_train, order=(param))
                model_fit = model.fit()

                # make predictions
                predictions =  model_fit.forecast(steps=24, dynamic=False)
                prediction_values = predictions[['ice_extent']].values
                true_values = cv_test[['ice_extent']].values
                # error calc
                maes.append(mean_absolute_error(true_values, prediction_values))

        
        mean_mae = np.mean(maes)
        print('MAE: ' + str(mean_mae))    

        if mean_mae < best_mean_mae:
            best_mean_mae = mean_mae
            best_maes = maes
            best_pq = param
            best_predictions = prediction_values
    except Exception as e:
        print(e)
        continue
   
# plot
print()
print('Best MAE = ' + str(best_mean_mae))
print(best_pq)

# best range(0,5)
# Best MAE = 0.1772618201196872
# (1, 1)
# Wall time: 7min 53s


# In[175]:


best_pq = (4,1)


# In[176]:


from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_absolute_error
import timeit


start_time = timeit.default_timer()

warnings.filterwarnings("ignore") # specify to ignore warning messages

print("-------")

maes = []

for train_index, test_index in tscv.split(dataset):
    if train_index.size > 300:
        # initialize cross validation train and test sets
        cv_train, cv_test = dataset.iloc[train_index], dataset.iloc[test_index]

        # build model
        model = VARMAX(cv_train, order=(best_pq))
        model_fit = model.fit()

        # make predictions
        predictions =  model_fit.forecast(steps=24, dynamic=False)
        prediction_values = predictions[['ice_extent']].values
        true_values = cv_test[['ice_extent']].values
        # error calc
        maes.append(mean_absolute_error(true_values, prediction_values))

        print("I",end="")
    

time_VARMAX = timeit.default_timer() - start_time
mae_mean = np.mean(maes)
MAE_VARMAX = mae_mean
last_MAE_VARMAX = maes[-1]
last_predictions_VARMAX = prediction_values

print()
print('Mean MAE: %.3f x 1 000 000 km\u00b2' % MAE_VARMAX)
print('MAE of last prediction: %.3f x 1 000 000 km\u00b2' % last_MAE_VARMAX)
print('Execution time: %.3f seconds' % time_VARMAX)
full_graph(prediction_values, 'Last prediction VARMAX')
print(maes)


# ## Seasonal differencing

# In[13]:


from statsmodels.tsa.statespace.varmax import VARMAX
import itertools
import warnings
import sys
from sklearn.metrics import mean_absolute_error



# Define the p, d and q parameters to take any value between 0 and 2
p = q = range(0, 5)

# Generate all different combinations of p, q and q triplets
pq = list(itertools.product(p, q))
best_pq = pq
best_mean_mae = np.inf
warnings.filterwarnings("ignore") # specify to ignore warning messages
for param in pq:
    print(param)
    try:   # some parametercombinations might lead to crash, so catch exceptions and continue
        maes = []
        for train_index, test_index in tscv.split(ts_diff_seasonal):
            if train_index.size > 300:
                # initialize cross validation train and test sets
                cv_train, cv_test = ts_diff_seasonal.iloc[train_index], ts_diff_seasonal.iloc[test_index]

                # build model
                model = VARMAX(cv_train, order=(param))
                model_fit = model.fit()

                # make predictions
                predictions =  model_fit.forecast(steps=24, dynamic=False)
                prediction_values = predictions[['ice_extent']].values
                true_values = cv_test[['ice_extent']].values
                # error calc
                maes.append(mean_absolute_error(true_values, prediction_values))

        
        mean_mae = np.mean(maes)
        print('MAE: ' + str(mean_mae))    

        if mean_mae < best_mean_mae:
            best_mean_mae = mean_mae
            best_maes = maes
            best_pq = param
            best_predictions = prediction_values
    except Exception as e:
        print(e)
        continue
   
# plot
print()
print('Best MAE = ' + str(best_mean_mae))
print(best_pq)

# best range(0,5) = (0,1)


# In[70]:


best_pq = (0,1)


# In[163]:


from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_absolute_error
import timeit


start_time = timeit.default_timer()

warnings.filterwarnings("ignore") # specify to ignore warning messages

print("-------")

maes = []

for train_index, test_index in tscv.split(ts_diff_seasonal):
    if train_index.size > 300:
        # initialize cross validation train and test sets
        cv_train, cv_test = ts_diff_seasonal.iloc[train_index], ts_diff_seasonal.iloc[test_index]

        # build model
        model = VARMAX(cv_train, order=(best_pq))
        model_fit = model.fit()

        # make predictions
        predictions =  model_fit.forecast(steps=24, dynamic=False)
        prediction_values = predictions[['ice_extent']].values
        true_values = cv_test[['ice_extent']].values
        # error calc
        maes.append(mean_absolute_error(true_values, prediction_values))

        print("I",end="")
    

time_VARMAX_seasonal = timeit.default_timer() - start_time
mae_mean_seasonal = np.mean(maes)
MAE_VARMAX_seasonal = mae_mean
last_MAE_VARMAX_seasonal = maes[-1]
last_predictions_VARMAX_seasonal = prediction_values

print()
print('Mean MAE: %.3f x 1 000 000 km\u00b2' % MAE_VARMAX_seasonal)
print('MAE of last prediction: %.3f x 1 000 000 km\u00b2' % last_MAE_VARMAX_seasonal)
print('Execution time: %.3f seconds' % time_VARMAX_seasonal)
full_graph_seasonal(last_predictions_VARMAX_seasonal, 'Last prediction VARMAX')
print(maes)


# # SARIMAX

# In[171]:


# setting up values for displaying prediction of the last 2 years
data = dataset
test_size = 24
data_train = data[:-test_size]
data_test = data[-test_size:]


# In[194]:


# singular test

import itertools
import warnings
import sys
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore") # specify to ignore warning messages

# Variables
endog = data_train[['ice_extent']]
exog = sm.add_constant(data_train[['mean_temp']])
exog_test = sm.add_constant(data_test[['mean_temp']])

# Fit the model
mod = sm.tsa.statespace.SARIMAX(endog, exog)

# fit model
model_fit = model.fit()

yhat = model_fit.forecast(steps = 24, exog=exog_test)

mae = mean_absolute_error(yhat, data_test[['ice_extent']])

# plot
print(mae)
plt.plot(data_test[['ice_extent']], color='blue')
plt.plot(yhat, color='red')
plt.show()


# Adding exogenous variables with SARIMAX requires to give the mean temperatures from the testset for forecasting, which is not in line with the other usecases, therefore it won't be further explored
# (extra research!)

# # LSTM

# ## Random walk diff

# In[72]:


ts


# In[73]:


# multivariate multi-step encoder-decoder lstm example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import warnings

warnings.filterwarnings("ignore") # specify to ignore warning messages


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def predict_LSTM(train, test, n_neurons, n_epochs):
    test['sum'] = test['mean_temp'] + test['ice_extent']


    # define input sequence
    in_seq1 = train.values[:,0]
    in_seq2 = train.values[:,1]
    out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

    # convert to [rows, columns] structure
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))
    
    # horizontally stack columns
    dataset = hstack((in_seq1, in_seq2, out_seq))
    
    # choose a number of time steps
    n_steps_in, n_steps_out = 24, 24
    
    # covert into input/output
    X, y = split_sequences(dataset, n_steps_in, n_steps_out)
    
    # the dataset knows the number of features, e.g. 2
    n_features = X.shape[2]
    
    # define model
    model = Sequential()
    model.add(LSTM(n_neurons, activation='relu', input_shape=(n_steps_in, n_features)))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(n_neurons, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(optimizer='adam', loss='mae')
    
    # fit model
    model.fit(X, y, epochs=n_epochs, verbose=0)
    
    # demonstrate prediction
    x_input = test.values
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)
    return yhat
    


# In[18]:


from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_absolute_error
import timeit
import tensorflow as tf

start_time = timeit.default_timer()

# warnings.filterwarnings("ignore") # specify to ignore warning messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

maes = []
global_maes = []
best_MAE = np.inf


# n_neurons_array = [1,5,10,20]
# n_epochs_array = [100,200,300]

n_neurons_array = [1, 5, 10, 20, 30]
n_epochs_array = [100,200,300]


print("-------")

maes = []
for n_neurons in n_neurons_array:
    for n_epochs in n_epochs_array:
        for train_index, test_index in tscv.split(dataset):
            if train_index.size > 300:
                # initialize cross validation train and test sets
                cv_train, cv_test = dataset.iloc[train_index], dataset.iloc[test_index]

                yhat = predict_LSTM(cv_train, cv_test, n_neurons, n_epochs)


                prediction_values = yhat[0][:,0]
                true_values = cv_test[['ice_extent']].values

                # error calc
                maes.append(mean_absolute_error(true_values, prediction_values))

                print("I",end="")
        time_LSTM = timeit.default_timer() - start_time
        MAE_LSTM = np.mean(maes)
        last_MAE_LSTM = maes[-1]
        global_maes.append(MAE_LSTM)

        if best_MAE > MAE_LSTM:
            best_n_neurons = n_neurons
            best_n_epochs = n_epochs
            best_MAE = MAE_LSTM

        print()
        print(n_neurons)
        print(n_epochs)
        print(MAE_LSTM)
        print()    

print('Best:')
print('N neurons')
print(best_n_neurons)
print('Epochs size')
print(best_n_epochs)
print('MAE')
print(best_MAE)
plt.bar(range(0,len(global_maes)), global_maes)


# In[177]:


best_n_neurons, best_n_epochs = 30, 300


# In[178]:


from sklearn.metrics import mean_absolute_error
import timeit
import tensorflow as tf

start_time = timeit.default_timer()

# warnings.filterwarnings("ignore") # specify to ignore warning messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


print("-------")

maes = []

for train_index, test_index in tscv.split(dataset):
    if train_index.size > 300:
        # initialize cross validation train and test sets
        cv_train, cv_test = dataset.iloc[train_index], dataset.iloc[test_index]
        yhat = predict_LSTM(cv_train, cv_test, best_n_neurons, best_n_epochs)


        prediction_values = yhat[0][:,0]
        true_values = cv_test[['ice_extent']].values

        # error calc
        maes.append(mean_absolute_error(true_values, prediction_values))

        print("I",end="")
    

time_LSTM = timeit.default_timer() - start_time
mae_mean = np.mean(maes)
MAE_LSTM = mae_mean
last_MAE_LSTM = maes[-1]
last_predictions_LSTM = prediction_values

print()
print('Mean MAE: %.3f x 1 000 000 km\u00b2' % MAE_LSTM)
print('MAE of last prediction: %.3f x 1 000 000 km\u00b2' % last_MAE_LSTM)
print('Execution time: %.3f seconds' % time_LSTM)
full_graph(last_predictions_LSTM, 'Last prediction LSTM')
print(maes)


# ## Seasonal differencing

# In[76]:


from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_absolute_error
import timeit
import tensorflow as tf

start_time = timeit.default_timer()

# warnings.filterwarnings("ignore") # specify to ignore warning messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

maes = []
global_maes = []
best_MAE = np.inf


# n_neurons_array = [1,5,10,20]
# n_epochs_array = [100,200,300]

n_neurons_array = [1, 5, 10, 20, 30]
n_epochs_array = [100,200,300]


print("-------")

maes = []
for n_neurons in n_neurons_array:
    for n_epochs in n_epochs_array:
        for train_index, test_index in tscv.split(ts_diff_seasonal):
            if train_index.size > 300:
                # initialize cross validation train and test sets
                cv_train, cv_test = ts_diff_seasonal.iloc[train_index], ts_diff_seasonal.iloc[test_index]

                yhat = predict_LSTM(cv_train, cv_test, n_neurons, n_epochs)


                prediction_values = yhat[0][:,0]
                true_values = cv_test[['ice_extent']].values

                # error calc
                maes.append(mean_absolute_error(true_values, prediction_values))

                print("I",end="")
        time_LSTM = timeit.default_timer() - start_time
        MAE_LSTM = np.mean(maes)
        last_MAE_LSTM = maes[-1]
        global_maes.append(MAE_LSTM)

        if best_MAE > MAE_LSTM:
            best_n_neurons = n_neurons
            best_n_epochs = n_epochs
            best_MAE = MAE_LSTM

        print()
        print(n_neurons)
        print(n_epochs)
        print(MAE_LSTM)
        print()    

print('Best:')
print('N neurons')
print(best_n_neurons)
print('Epochs size')
print(best_n_epochs)
print('MAE')
print(best_MAE)
plt.bar(range(0,len(global_maes)), global_maes)


# In[164]:


best_n_neurons, best_n_epochs = 1, 200


# In[165]:


from sklearn.metrics import mean_absolute_error
import timeit
import tensorflow as tf

start_time = timeit.default_timer()

# warnings.filterwarnings("ignore") # specify to ignore warning messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


print("-------")

maes = []

for train_index, test_index in tscv.split(dataset):
    if train_index.size > 300:
        # initialize cross validation train and test sets
        cv_train, cv_test = dataset.iloc[train_index], dataset.iloc[test_index]
        yhat = predict_LSTM(cv_train, cv_test, best_n_neurons, best_n_epochs)


        prediction_values = yhat[0][:,0]
        true_values = cv_test[['ice_extent']].values

        # error calc
        maes.append(mean_absolute_error(true_values, prediction_values))

        print("I",end="")
    

time_LSTM_seasonal = timeit.default_timer() - start_time
mae_mean_seasonal = np.mean(maes)
MAE_LSTM_seasonal = mae_mean
last_MAE_LSTM_seasonal = maes[-1]
last_predictions_LSTM_seasonal = prediction_values

print()
print('Mean MAE: %.3f x 1 000 000 km\u00b2' % MAE_LSTM_seasonal)
print('MAE of last prediction: %.3f x 1 000 000 km\u00b2' % last_MAE_LSTM_seasonal)
print('Execution time: %.3f seconds' % time_LSTM_seasonal)
full_graph_seasonal(last_predictions_LSTM_seasonal, 'Last prediction LSTM')
print(maes)


# # Evaluation

# In[181]:


# formatting
results = [[MAE_VARMAX, time_VARMAX, last_MAE_VARMAX],
           [MAE_VARMAX_seasonal, time_VARMAX_seasonal, last_MAE_VARMAX_seasonal],
           [MAE_LSTM, time_LSTM, last_MAE_LSTM],
           [MAE_LSTM_seasonal, time_LSTM_seasonal, last_MAE_LSTM_seasonal]]

# display results
results = pd.DataFrame(results, columns=['Mean MAE (x 1 000 000 km\u00b2)','Execution time (s)','Last MAE (x 1 000 000 km\u00b2)']
             ,index=['VARMAX','VARMAX_seasonal','LSTM','LSTM_seasonal']).round(decimals=3)
results


# In[182]:


full_graph(last_predictions_VARMAX, 'Last prediction VARMAX with random walk differencing')
full_graph_seasonal(last_predictions_VARMAX_seasonal, 'Last prediction VARMAX with seasonal differencing')
full_graph(last_predictions_LSTM, 'Last prediction LSTM with random walk differencing')
full_graph_seasonal(last_predictions_LSTM_seasonal, 'Last prediction LSTM with seasonal differencing')

