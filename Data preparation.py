#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pylab as plt
import numpy as np


# # Dataset exploration

# ## Dataset #1: seaice

# source: https://www.kaggle.com/nsidcorg/daily-sea-ice-extent-data

# In[3]:


ice = pd.read_csv('./data/seaice.csv')
ice.columns = ['Year', 'Month', 'Day', 'Extent', 'Missing', 'Source Data',
       'hemisphere']
ice


# In[5]:


plt.plot(ice[['Extent']])


# In[6]:


ice.groupby('hemisphere').count()


# In[7]:


ice.groupby('Year').count()[['Extent']]


# In[8]:


ice.groupby('Year').mean()[['Extent']]


# In[9]:


plt.scatter(ice.groupby('Year').mean()[['Extent']][:-1].index, ice.groupby('Year').mean()[['Extent']][:-1])


# In[10]:


ice.groupby(['Year','hemisphere']).count().tail(60)


# In[11]:


plt.xlabel('Years')
plt.ylabel('Extent')
plt.plot(ice[ice['hemisphere'] == 'north'].groupby('Year').mean()[['Extent']][:-1].index, ice[ice['hemisphere'] == 'north'].groupby('Year').mean()[['Extent']][:-1],label='Northern hemisphere')
plt.plot(ice[ice['hemisphere'] == 'south'].groupby('Year').mean()[['Extent']][:-1].index, ice[ice['hemisphere'] == 'south'].groupby('Year').mean()[['Extent']][:-1],label='Southern hemisphere')
plt.legend()


# In[12]:


plt.xlabel('Years')
plt.ylabel('Extent')
plt.scatter(ice.groupby('Year').mean()[['Extent']][:-1].index, ice.groupby('Year').mean()[['Extent']][:-1])


# In[ ]:





# In[13]:


print('start : ' + str(ice['Year'][0]))
print('end : ' + str(ice['Year'].tail(1).iloc[0]))


# In[14]:


2019-1978


# ## Dataset #1: Toronto_temp

# In[8]:


tt


# In[4]:


# Source: https://www.kaggle.com/rainbowgirl/climate-data-toronto-19372018
tt = pd.read_csv('./data/Toronto_temp.csv')
tt = tt[tt['Day'] == 1]
tt['Year'] = tt['Year'].replace({'2,013':'2013',
              '2,014':'2014',
              '2,015':'2015',
              '2,016':'2016',
              '2,017':'2017',
              '2,018':'2018'})
# tt.groupby('Year').count()
tt = tt[(tt['Year'] != '1937')]
ttt = tt.groupby('Year').count()
#ttt.head(50)
#tt.groupby('Year').count().tail(50)
meantt = tt.groupby('Year').mean()['Mean Temp (C)']
meantt
#meantt.index
#meantt
meantt.sort_index(inplace=True)

plt.xlabel('Years')
plt.ylabel('Temperature (C)')
plt.xticks(np.array(range(0,meantt.size,10)))
plt.scatter(meantt.index, meantt)

print('start : ' + meantt.index[0])
print('end : ' + meantt.index[-1])

new_row = pd.Series({'Mean Temp (C)' : 0.555556, 'Year': '2018', 'Month':12})
tt = tt.append(new_row, ignore_index=True)
tt['Year'] = tt['Year'].astype(int)
mean_temp_monthly = tt[['Year','Month','Mean Temp (C)']].set_index(['Year','Month']).sort_index()
# mean_temp_monthly
mean_temp_monthly = mean_temp_monthly[mean_temp_monthly.index.get_level_values(0).astype(int) >= 1979 ]
mean_temp_monthly


# In[6]:


tt


# ## Dataset #3: seaice2

# Completer version of dataset #1
# 
# source: https://nsidc.org/arcticseaicenews/sea-ice-tools/

# In[13]:


ice2.mean()[1:-2]


# In[17]:


ice2_mean


# In[19]:


ice2.mean()[1:-2]


# In[24]:


ice2.mean()


# In[26]:


ice2


# In[27]:


ice2 = pd.read_csv('./data/seaice2.csv')
# ice2
ice2_mean = ice2.mean()[1:-2]
ice2_mean
ice2_mean.index = ice2_mean.index.values.astype(int)

plt.title('Yearly ice extent')
plt.scatter(ice2_mean.index,ice2_mean)
plt.xlabel('Years')
plt.ylabel('Extent')
plt.show()

# ice2['2018']
# pd.concat([ice2['2016'],ice2['2017'],ice2['2018'],ice2['2019']]).reset_index()[0]
# ice2[['2018']].append(ice2[['2019']])
ice2.rename(columns={'Unnamed: 0' : 'Month', 'Unnamed: 1' : 'Day'}, inplace = True)
ice2.drop([' ','1981-2010','Day','1978','2020'],axis=1,inplace=True)
values = ice2.values
i = 0
for row in values :
    if type(row[0]) != str :
        values[i][0] = month
    else:
        month = row[0]
    i = i +1
# ice2.columns.values
ice2_clean = pd.DataFrame(values)
ice2_clean.columns = ice2.columns.values
# ice2_clean.head(5)
ice2_monthly_mean = ice2_clean.set_index('Month').astype(float).groupby('Month',sort=False).mean()
# ice2_monthly_mean
# ice2_monthly_mean.T.stack().index.get_level_values(0)
# ice2_monthly_mean.T.stack().reset_index(level=['Month']).drop(columns=['Month'])
ice2_monthly_mean_chron = ice2_monthly_mean.T.stack().reset_index(level=['Month']).drop(columns=['Month'])
# ice2.columns.size
plt.title('Monthly ice extent')
plt.plot(ice2_monthly_mean_chron.values)
plt.xticks(np.array(range(0,500,75)))
plt.xlabel('Cumulative month')
plt.ylabel('Extent')
plt.show()

# np.unique(ice2_monthly_mean_chron.index.values).size*12
print('from ' + ice2_monthly_mean_chron.index.values[0] + ' until ' + ice2_monthly_mean_chron.index.values[-1])
ice2_monthly_mean_chron = ice2_monthly_mean.T.stack().reset_index(level=['Month']).drop(columns=['Month'])
ice2_monthly_mean_chron.columns = ['ice_extent']
ice2_monthly_mean_chron


# # Dataset Combination

# In[54]:


ice2_monthly_mean_chron_cut = ice2_monthly_mean_chron[:-12]
# ice2_monthly_mean_chron
# ice2_monthly_mean_chron_cut
# mean_temp_monthly
# ice2_monthly_mean_chron_cut
combined = mean_temp_monthly[mean_temp_monthly.index.get_level_values(0) >= 1979]
combined['ice_extent'] = ice2_monthly_mean_chron_cut.values
# combined
combined.rename(columns={'Mean Temp (C)': 'mean_temp'}, inplace=True)
dataframe_monthly = combined
# dataframe_monthly
# dataframe_monthly[['mean_temp']]
plt.plot(dataframe_monthly[['mean_temp']].values[-24:],label='temperature')
plt.plot(dataframe_monthly[['ice_extent']].values[-24:],label='ice extent')
plt.legend()
plt.show()
dataframe_yearly = combined.groupby('Year').mean()
# dataframe_yearly
# dataframe_monthly[['mean_temp']].values
plt.plot(dataframe_monthly[['mean_temp']].values,label='temperature')
plt.plot(dataframe_monthly[['ice_extent']].values,label='ice extent')
plt.legend()
dataframe_monthly.to_csv('./data/dataframe_monthly.csv')
dataframe_yearly.to_csv('./data/dataframe_yearly.csv')


# In[ ]:




