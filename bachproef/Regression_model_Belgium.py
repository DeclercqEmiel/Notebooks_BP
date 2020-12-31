#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import datetime

Last check 5/14/20 last value: 307

predicted values for next days:
62	63.0	370.334145
63	64.0	338.834148
64	65.0	319.142565
65	66.0	295.600054
66	67.0	275.961111
67	68.0	255.803816
68	69.0	235.955925
69	70.0	219.198490
70	71.0	213.190145Time based

regressietechnieken
polynomial regression
Probleemstelling(en)
Focus op 1 om een eenvoudig model op te stellen
features en output identificeren
Lopend gemiddelde van de vorige 7 dagen op pieken uit te vlakken
Datums per row, dagen relatief nummeren
Alles onder de 10 gevallen niet gebruiken
Per dag een nieuwe waarde dagen sinds lockdown, -1 of 0 voor de dagen voor lockdown
Population density
Total population

Mean age of the population
Literatuurstudie:
    Recurrente neurale netwerken, keras tijdsreeksen
    https://hogent-my.sharepoint.com/personal/johan_decorte_hogent_be/Documents/Chatbestanden%20van%20Microsoft%20Teams/RNN_Sentiment_Analysis.ipynb
    
https://www.vrt.be/vrtnws/nl/2020/04/22/belgische-corona-aanpak-door-de-ogen-van-de-internationale-pers/?fbclid=IwAR2f8RG09CeLFgiE0dxSdWDWdaOH9k4l-W1UX03Ox1o8HnVV-ovF7abitXM
https://www.demorgen.be/nieuws/oversterfte-door-covid-19-groter-dan-tijdens-voorbije-griepseizoenen-of-hittegolven~bee1a2ea/?fbclid=IwAR23Cff9T6ChYHM_7OmTVvV4NsdZiFA_-BnjibstU1vuIZzylhLdDuMFfng
landen voorbij de piek gebruiken als trainingsdata


# # Intresting parameters
-Mean of how many values
-Polynomial degree
# In[2]:


# Minimum voor bp:
# Model en evaluatie van het model


# In[3]:


pd.set_option('display.max_columns', 500)


# In[4]:


# # Using github
# confirmed2 = pd.read_csv("https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
# deaths2 = pd.read_csv("https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv",index_col=0,parse_dates=[0])
# recovered2 = pd.read_csv("https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_active_global.csv",index_col=0,parse_dates=[0])


# # Retrieval

# In[5]:


confirmed = pd.read_csv("C:\\Users\\Emiel\\0BP\\input\\COVID-19\\csse_covid_19_data\\csse_covid_19_time_series\\time_series_covid19_confirmed_global.csv")
deaths = pd.read_csv("C:\\Users\\Emiel\\0BP\\input\\COVID-19\\csse_covid_19_data\\csse_covid_19_time_series\\time_series_covid19_deaths_global.csv")
recovered = pd.read_csv("C:\\Users\\Emiel\\0BP\\input\\COVID-19\\csse_covid_19_data\\csse_covid_19_time_series\\time_series_covid19_recovered_global.csv")


# # Formatting

# ### Select most reliable countries

# In[6]:


eu = ['Belgium', 
      'Netherlands', 
      'Spain',
      'Italy',
      'Sweden',
      'United Kingdom',
      'Germany',
      'France',
      'Switzerland',
      'Portugal',
      'Austria',
      'Ireland',
      'Norway',
      'Denmark',
      'Slovenia',
      'Czechia',
      'Belarus',
     ]

confirmedDataEu = confirmed[(confirmed['Country/Region'].isin(eu)) & (confirmed['Province/State'].isnull()) ]

confirmedDataEu = confirmedDataEu.drop('Province/State', axis =1)
# deathsData = deaths.drop(['Province/State','Lat','Long'], axis=1)
# deathsDataBe = deathsData[deathsData['Country/Region']=='Belgium']

# recoveredData = recovered.drop(['Province/State','Lat','Long'], axis=1)
# recoveredDataBe = recoveredData[recoveredData['Country/Region']=='Belgium']


# In[7]:


confirmedDataEu.columns[4:]
dates = confirmedDataEu.columns[3:len(confirmedDataEu.columns)]


# In[8]:


confirmedDataEu


# In[9]:


# Replace values bellow 10 with -1
def to_min_1(v): 
    if v < 10 :
        return -1
    return v


# In[10]:


for date in dates:
    confirmedDataEu[date] = confirmedDataEu[date].map(to_min_1)


# In[11]:


confirmedDataEu = confirmedDataEu.reset_index()
confirmedDataEu = confirmedDataEu.drop('index', axis=1)


# In[12]:


confirmedDataEu['Long']


# In[13]:


confirmedDataEu = confirmedDataEu.set_index('Country/Region').drop(['Lat','Long'],axis=1)


# In[14]:


population = [9006398, 9449323,11589623,10708981,5792202,65273511,
             83783942,4937786,60461826,17134872,5421241,10196709,
             2078938,46754778,10099265,8654622,67886011]
confirmedDataEuPM = confirmedDataEu.T / population * 1000000
confirmedDataEuPM = confirmedDataEuPM.T


# In[15]:


confirmedDataEu


# In[16]:


confirmedDataEuPM


# In[17]:


for date in dates:
    confirmedDataEuPM[date] = confirmedDataEuPM[date].map(to_min_1)


# ### Shift starting point to minimum of 10 cases

# In[18]:


confirmedDataEu.T['Belgium'].value_counts()[-1]


# In[19]:


for countryLoc in confirmedDataEuPM.index:
    
    coronaFreeDays = confirmedDataEuPM.T[countryLoc].value_counts()[-1]
    confirmedDataEuPM.loc[countryLoc, dates] = confirmedDataEuPM.loc[countryLoc, dates].shift(periods=-coronaFreeDays)


# In[20]:


confirmedDataEuPM


# In[21]:


# # Rename columns
# seq = np.arange(1, confirmedDataEu.columns[3:].size-1)
# seq = [ 'day_' + str(s) for s in seq]
# confirmedDataEu.rename(columns=dict(zip(confirmedDataEu.columns[3:], seq)),inplace=True)


# In[22]:


# Remove unnecessary days
confirmedDataEuPM.dropna(axis = 1, how = 'all', inplace = True)


# In[23]:


confirmedDataEuPMT = confirmedDataEuPM.T


# In[24]:


confirmedDataEuPMT


# ### Replace values with mean of value from last 7 days

# In[25]:


# confirmedDataEuT.ewm(span = 7, min_periods = 7).mean().head(50)
confirmedDataEuPMTFlattened = confirmedDataEuPMT.rolling(7).mean()


# In[26]:


confirmedDataEuPMT


# In[27]:


confirmedDataEuPMTFlattened = confirmedDataEuPMTFlattened - confirmedDataEuPMTFlattened.shift(1)
confirmedDataEuPMT = confirmedDataEuPMT - confirmedDataEuPMT.shift(1)


# In[28]:


confirmedDataEuPMTFlattened.columns


# In[ ]:





# In[29]:


confirmedDataEuPMTFlattened.tail(50)


# In[30]:


confirmedDataEuPMTFlattened


# In[31]:



plt.plot(confirmedDataEuPMTFlattened.index, confirmedDataEuPMTFlattened['Belgium'], label = "Belgium")
plt.plot(confirmedDataEuPMTFlattened.index, confirmedDataEuPMTFlattened['Italy'], label = "Italy")
# plt.plot(confirmedDataEuPMTFlattened.index, confirmedDataEuPMTFlattened['Switzerland'], label = "Switzerland")
# plt.plot(confirmedDataEuPMTFlattened.index, confirmedDataEuPMTFlattened['Netherlands'], label = "Netherlands")
plt.xlabel('Dates')
plt.ylabel('Amount')
plt.title('Combined cumulative chart per million inhabitants')

plt.legend()
plt.show()

fig=plt.figure(figsize=(20, 20))


# In[32]:


confirmedDataEuPMTFlattened.columns


# # Modelling 

# In[33]:


# 1 vgl tussen verschillende landen
# 2 Curve per land individueel doortrekken
#   veeltermvgl
#   times series


# ### Polynomial

# #### Example
https://towardsdatascience.com/machine-learning-polynomial-regression-with-python-5328e4e8a386
# In[34]:


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# # Importing the dataset
# dataset = pd.read_csv('https://s3.us-west-2.amazonaws.com/public.gamelab.fun/dataset/position_salaries.csv')
# X = dataset.iloc[:, 1:2].values
# y = dataset.iloc[:, 2].values

# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # Fitting Linear Regression to the dataset
# from sklearn.linear_model import LinearRegression
# lin_reg = LinearRegression()
# lin_reg.fit(X, y)

# # Visualizing the Linear Regression results
# def viz_linear():
#     plt.scatter(X, y, color='red')
#     plt.plot(X, lin_reg.predict(X), color='blue')
#     plt.title('Truth or Bluff (Linear Regression)')
#     plt.xlabel('Position level')
#     plt.ylabel('Salary')
#     plt.show()
#     return
# viz_linear()

# # Fitting Polynomial Regression to the dataset
# from sklearn.preprocessing import PolynomialFeatures
# poly_reg = PolynomialFeatures(degree=4)
# X_poly = poly_reg.fit_transform(X)
# pol_reg = LinearRegression()
# pol_reg.fit(X_poly, y)

# # Visualizing the Polymonial Regression results
# def viz_polymonial():
#     plt.scatter(X, y, color='red')
#     plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
#     plt.title('Truth or Bluff (Linear Regression)')
#     plt.xlabel('Position level')
#     plt.ylabel('Salary')
#     plt.show()
#     return
# viz_polymonial()

# # Predicting a new result with Linear Regression
# lin_reg.predict([[5.5]])
# #output should be 249500

# # Predicting a new result with Polymonial Regression
# pol_reg.predict(poly_reg.fit_transform([[5.5]]))
# #output should be 132148.43750003


# #### Actual

# In[35]:


BEPrep = confirmedDataEuPMTFlattened.Belgium.shift(-7).dropna().reset_index().reset_index()
ITPrep = confirmedDataEuPMTFlattened.Italy.shift(-7).dropna().reset_index().reset_index()

BEPrep.columns = ['day_nr','day','confirmed_cases']
BEPrep['day_nr'] += 1

ITPrep.columns = ['day_nr','day','confirmed_cases']
ITPrep['day_nr'] += 1

BEPrep = BEPrep.drop('day',axis=1)
ITPrep = ITPrep.drop('day',axis=1)

y = BEPrep.iloc[:, 1].values
X = BEPrep.iloc[:, 0].values


# In[36]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[37]:


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


poly_reg = PolynomialFeatures(degree=8)

X_poly = poly_reg.fit_transform(X.reshape(-1,1))
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

# Visualizing the Polymonial Regression results
def viz_polymonial():
    plt.scatter(X, y, color='red')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X.reshape(-1,1))), color='blue')
    plt.title('')
    plt.xlabel('Day')
    plt.ylabel('New Cases')
    plt.ylim(bottom=0)
    plt.show()
    return
viz_polymonial()


# In[38]:


# Predicting a new result with Polymonial Regression
pol_reg.predict(poly_reg.fit_transform([[12]]))[0]


# In[39]:


i = BEPrep.shape[0]+1
BEPred = BEPrep
while i <= 100 :
    BEPred = BEPred.append({'day_nr':i,'confirmed_cases': pol_reg.predict(poly_reg.fit_transform([[i]]))[0]}, ignore_index=True)
    i += 1


# In[40]:


BEPred.tail(50)


# In[41]:


y = BEPrep.iloc[:, 1].values
X = BEPrep.iloc[:, 0].values

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


poly_reg = PolynomialFeatures(degree=8)
X_poly = poly_reg.fit_transform(X.reshape(-1,1))
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

# Visualizing the Polymonial Regression results
def viz_polymonial():
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X.reshape(-1,1))), color='blue')
    plt.title('')
    plt.xlabel('Day')
    plt.ylabel('New Cases')
    plt.ylim(bottom=0, top=130)
    plt.show()
    return
viz_polymonial()

Curve
https://www.desmos.com/calculator/kreo2ssqj8
y=2^{\left(-\frac{\left(x\ -37\right)}{14}^{2}\ \ \right)}\cdot1413
How to skew this?Volgens het normaalverloop zouden we op dag 74 weer onder te 10 moeten zakken, aangezien de piek op dag 35, dit zal echter later zijn omdat de maatregels van de lockdown op 6 mei, dag 58 zijn aangepast zodat nu elk gezin maximaal 4 personen mag ontvangen. Zo is er ook weer een lichte stijging merkbaar sinds dag 59.
Hierdoor we op dag 66 pas weer op het verwachte niveau zitten, dit zal voor een week vertraging zorgen voor het dalen van  het aantal nieuwe besmettingen tot 10. Dit zou dus in principe moeten plaatsvinden op dag 81 maar dit hangt natuurlijk af van hoelang en hoe strikt men de lockdown nog zal doorvoeren.

Dit zou het geval zijn moest de curve parabolisch verlopen, maar wanneer we kijken naar de curves van andere landen zien we dat deze een pak trager dalen dan dat ze stijgen.

Ligt de oorzaak hiervan bij het afbouwen van de maatregelen of is dit eigen aan een virus?
Dus om dit correct te kunnen interpreteren moeten we een factor achterhalen

lockdown op dag 10
Mischien model maken waarbij de dag/het aantal besmettingen kan ingegeven worden waar


Incubatietijd van 4 tot 13 dagen
2 tot 12, 14 voor de zekerheid
5.2 dagen gemiddeld
https://www.nhg.org/veelgestelde-vragen/wat-de-incubatietijd-van-het-virus

Puur theoretisch zouden er dan geen nieuwe gevallen meer mogen zijn na 14 dagen strikte lockdown, als iedereen voldoende hygiënisch is en geen contact meer heeft met anderen.

In praktijk moet de balans gevonden tussen het beperken van de economische schade en het 
# In[42]:


BEPrep['confirmed_cases'].to_list()


# In[43]:


# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
 
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(ITPrep['confirmed_cases'].to_list(), n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=300, verbose=2)


# In[73]:


BEPredLSTM = BEPrep

while BEPredLSTM.shape[0] < ITPrep.shape[0]:
    counter = BEPredLSTM.shape[0]
    
    x_input = array(BEPredLSTM['confirmed_cases'].tail(3).to_list())
    x_input = x_input.reshape((1, n_steps, n_features))
    
    predicted_value = model.predict(x_input, verbose=2).item(0)
    print(predicted_value)
    BEPredLSTM = BEPredLSTM.append({'confirmed_cases': predicted_value,'day_nr':counter +1 }, ignore_index=True)


# In[74]:


BEPredLSTM.tail(100)


# In[75]:


plt.plot(BEPredLSTM.index, BEPredLSTM['confirmed_cases'], label = "Belgium")
plt.xlabel('Dates')
plt.ylabel('Amount')
plt.title('Combined cumulative chart')

plt.legend()
plt.show()

fig=plt.figure(figsize=(20, 20))


# In[76]:


BEPrep.shape[0]


# In[77]:


ITPrep.shape[0]


# In[78]:


BEPredLSTM['confirmed_cases']= BEPredLSTM['confirmed_cases'].multiply(11,589623)


# In[79]:


BEPredLSTM.head(100).tail(50)


# In[81]:


plt.plot(BEPredLSTM.index, BEPredLSTM['confirmed_cases'])
plt.xlabel('Dates')
plt.ylabel('Amount')
plt.title('Confirmed new cases Belgium')

plt.legend()
plt.show()

fig=plt.figure(figsize=(20, 20))


# In[ ]:




