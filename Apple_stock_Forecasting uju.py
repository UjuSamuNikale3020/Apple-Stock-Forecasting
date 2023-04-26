#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[2]:


data=pd.read_csv(r"C:\Users\tejas\Downloads\AAPL.csv")


# In[3]:


data.head()


# In[4]:


data.tail()


# *Data Describtion :-
# 
# **Date : Date of trading
# 
# **Open : Price at which security first trades
# 
# **High : Highest Price of the trading day
# 
# **Low : Lowest Price of the trading day
# 
# **Close : Last Price the stock traded during the trading day
# 
# **Adj Close : Price that is adjusts Coroporate Actions on Closing Price
# 
# **Volume : Number of Shares that changed hands during the trading day

# # Data Preprocessing

# #### Transform Date to datetime object and Set as Index :

# In[5]:


data.dtypes


# In[6]:


data.Date = pd.to_datetime(data.Date)
data.dtypes


# In[7]:


data.head()


# In[8]:


df = data.copy()
df.head()


# In[9]:


df.set_index('Date',inplace=True)


# In[10]:


df.index


# In[11]:


df.head()


# ### Check for Holidays :

# In[12]:


from datetime import date


# In[13]:


data.head(1).index, data.tail(1).index


# ## Create CustomBusinessDay :
# 

# In[14]:


from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


# In[15]:


us_cal = CustomBusinessDay(calendar=USFederalHolidayCalendar())


# In[16]:


my_range = pd.date_range(start='2012-01-03',end='2019-12-30',freq=us_cal)
print(my_range.difference(data.index))


# In[17]:


Cu = CustomBusinessDay(calendar=USFederalHolidayCalendar(),holidays=['2012-04-06', '2012-10-29', '2012-10-30', 
                                                                    '2013-03-29','2014-04-18', '2015-04-03', 
                                                                    '2016-03-25', '2017-04-14', '2018-03-30', 
                                                                    '2018-12-05', '2019-04-19'])
my_range = pd.date_range(start='2012-01-03',end='2019-12-30',freq=Cu)
print(my_range.difference(data.index))


# ## C is the custom frequency here that includes the holidays other than weekends.
# 
# 

# In[18]:


df


# In[19]:


df = df.asfreq(Cu)


# In[20]:


df.index


# ### Now the datetime index has the frequncy C.

# # Time Series Analysis :
# 

# In[21]:


data.head()


# ### Considering only the closing price for the analysis :
# 
# 

# In[22]:


df.drop(['Open','High','Low','Adj Close','Volume'],axis=1,inplace=True)
df.head()


# ### Downsampling the data to Monthly :
# 

# In[23]:


df_m = df.resample('M').mean()
df_m.head(),df_m.tail()


# ### Line plot :
# 

# In[24]:


df_m.plot(figsize=(16,8), linewidth=2,marker='o', fontsize=12);


# ### Seasonal Decomposition plot :
# 

# In[25]:


from statsmodels.tsa.seasonal import seasonal_decompose

seas_add = seasonal_decompose(df_m, model='additive')
seas_add.plot()
plt.gcf().set_size_inches(12, 8);


# #### Repeating patterns are seen. It shows yearly seasonality.
# 
# 

# ### Augmented Dickey-Fuller Test :
# 

# #### Null Hypothesis - Time series has unit root, hence not stationary
# 
# #### Alternate Hypothesis - Time series has no unit root, hence stationary
# 
# #### If p-value is less than 0.05, reject Null Hypothesis
# 
# #### If p-value is greater than 0.05, fail to reject Null Hypothesis

# In[26]:


from statsmodels.tsa.stattools import adfuller


# In[27]:


def test_stationarity(series):
    
    stat_test = adfuller(series,autolag='AIC')

    print('Test Statistic : ',stat_test[0])
    print('p value : ',stat_test[1])
    print('Number of Lags : ',stat_test[2])
    print('Critical values : ')
    for key, val in stat_test[4].items():
        print('\t',key, ': ',val)
    print()    
    if stat_test[1] > 0.05:
        print('Series is non-stationary')
    else:
        print('Series is stationary')


# In[28]:


test_stationarity(df)


# #### Check the order of non-seasonal differencing needed :
# 

# In[29]:


import pmdarima as pm


# In[30]:


diff = pm.arima.ndiffs(df_m['Close'],max_d=4)
print(f'Order of non-seasonal Differencing = {diff}')


# In[31]:


dfm_lag = df_m['Close'].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0]).dropna()
dfm_lag.plot();


# In[32]:


diff1 = pm.arima.ndiffs(dfm_lag,max_d=4)
print(f'Order of non-seasonal Differencing = {diff1}')


# In[33]:


test_stationarity(dfm_lag)


# #### The series is trend stationary now.
# 
# 

# In[34]:


dfm_lag_se = dfm_lag.rolling(window=8).apply(lambda x: x.iloc[1] - x.iloc[0]).dropna()
dfm_lag_se.plot();


# In[35]:


test_stationarity(dfm_lag_se)


# 
# 
# #### The p value is very low and the test statistic is less than 1% critical value. This suggests that we can reject the null hypothesis with a significance level of less than 1% (i.e. a low probability that the result is a statistical fluke).
# 
# #### Rejecting the null hypothesis means that the process has no unit root, and in turn that the time series is stationary or does not have time-dependent structure.

# # Exploratory Data Analysis :
# 

# In[36]:


d = df_m.copy()

# Separating month and date into separate column

d["month"] = d.index.strftime("%B") # month extraction
d["year"] = d.index.strftime("%Y") # year extraction
d['Q'] = d.index.quarter    # quarter extraction

# categorizing the quarters
d.loc[d['Q']==1,'quarter']='Q1'
d.loc[d['Q']==2,'quarter']='Q2'
d.loc[d['Q']==3,'quarter']='Q3'
d.loc[d['Q']==4,'quarter']='Q4'

d.head(12)


# In[37]:


plt.figure(figsize=(16,8))
plt.xticks(fontsize=14,fontweight='bold')
sns.boxplot(x="year",y="Close",data=d)
plt.title('Box plot of Closing Price',fontweight='bold',fontsize=18,color='brown');


# ##### As seen in the plot the mean is not constant over the years, which proves the time series is not stationary. There is an outlier below lower whisker in the year 2012 which indicates the price was lower than minimum at some instance and there is an outlier on the higher side in the year 2019 which indicates the value of stock had gone up more than the maximum at an instance.

# In[38]:


plt.figure(figsize=(16,10))
ax = sns.barplot(data=d,x='year',y='Close',palette='husl',ci=None)
plt.xticks(fontsize=14,fontweight='bold')
plt.title('Bar plot of Closing Price Year wise',fontweight='bold',fontsize=18,color='brown');
for i in ax.containers:
    ax.bar_label(i,fontsize=15)


# #### The bars represent the frequencies of distinct values of stock price. In the above plot, we can see after a few ups and downs from 2012 to 2016 the apple stock price has increased largely since 2017 and overall there is a upward trend.

# In[39]:


plt.figure(figsize=(12,8))
ax = sns.barplot(data=d,x='quarter',y='Close',palette='husl',ci=None)
plt.xticks(fontsize=14,fontweight='bold')
plt.title('Bar plot of Closing Price Quarter wise',fontweight='bold',fontsize=18,color='brown');
for i in ax.containers:
    ax.bar_label(i,fontsize=15)


# #### The quarter plot shows a clear upward trend .
# 
# 

# In[40]:


plt.figure(figsize=(16,10))
ax = sns.barplot(data=d,x='year',y='Close',palette='husl',hue='quarter',ci=None)
plt.xticks(fontsize=14,fontweight='bold')
plt.title('Multiple bar plot of Year and Quarter wise Close price',fontweight='bold',fontsize=18,color='brown');
for i in ax.containers:
    ax.bar_label(i,fontsize=11,rotation=90)


# #### There is a steep increase in the adj closing price of stocks in the last quarter of year 2019 whcih will possibly have greater impact in the prices of the coming year.
# 
# 

# In[41]:


plt.figure(figsize=(16,10))
ax = sns.barplot(data=d,x='month',y='Close',palette='husl',ci=None)
plt.xticks(rotation=60,fontsize=14,fontweight='bold')
plt.title('Bar plot of Closing Price Month wise',fontweight='bold',fontsize=18,color='brown');
for i in ax.containers:
    ax.bar_label(i,fontsize=15)


# #### The adj close price is higher in the month of November followed by December and October. This must be due to soaring of stock price in the last quarter of the year 2019 which we saw above.

# ### Pivot table for year and quarter :

# In[42]:


d_pivot = d.pivot_table(values='Close', index='year', columns='quarter')
d_pivot


# In[43]:


d_pivot.plot(kind='bar',stacked=True,figsize=(15,6))
plt.title('Stacked bar plot of Close price with year and quarter',fontweight='bold',fontsize=18,color='brown')
plt.xticks(rotation=0,fontsize=14,fontweight='bold');


# #### The plot shows the yearly raising trend of stock price along with the quarters. The year 2013 has the lowest stock price in all quarters.
# 
# 

# In[44]:


d_pivot.plot(kind='area',stacked=False,figsize=(15,6))
plt.title('Area plot of Close price with year and quarter without stacking',fontweight='bold',fontsize=18,color='brown')
plt.xticks(rotation=0,fontsize=14,fontweight='bold');


# In[45]:


d_pivot.plot(kind='area',stacked=True,figsize=(15,6))
plt.title('Area plot of Close price with year and quarter with stacking',fontweight='bold',fontsize=18,color='brown')
plt.xticks(rotation=0,fontsize=14,fontweight='bold');


# #### Both the plots show how the last quarter of 2019 has larger area indicating soaring stock price.
# 
# 

# ### Pivot table for year and month :
# 

# In[46]:


d_pivot2 = d.pivot_table(values='Close', index='year', columns='month')
d_pivot2


# In[47]:


##HeatMap to Verify Multicollinearity between Features
fig = plt.figure(figsize=(16,12))
matrix = np.triu(d_pivot2.corr())
ax = sns.heatmap(d_pivot2.corr(),annot=True,annot_kws={"size":14},mask=matrix,cmap='coolwarm')
ax.tick_params(labelsize=14)
sns.set(font_scale=3)
ax.set_title('HeatMap')
plt.style.use('fivethirtyeight')
plt.show()


# # ACF and PACF plots :
# 

# In[48]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[49]:


fig, ax = plt.subplots(figsize=(12,6))
acf = plot_acf(dfm_lag_se,lags=30,ax=ax)


# In[50]:


fig, ax = plt.subplots(figsize=(12,6))
pacf = plot_pacf(dfm_lag_se,lags=30,ax=ax)


# #### In both ACF and PACF plots, it's not clear whether they are tailing off or cutting off and since the time series needed first order differencing to make it stationary and seasonality id present it is SARIMA model that has to used for forcasting. Here since in acf and pacf plot only one point in each is above the confidence band, p = 1, q = 1, d = 1 for ARIMA and P=1, Q=1, D=1 for SARIMA with seasonality s=252 as the series has holidays.

# ## Split into train and test data :

# In[51]:


train = df.iloc[:len(df)-249]

# Taking last one year for testing
test = df.iloc[len(df)-249:]
test.head(),test.tail()


# ## ARIMA model :

# In[52]:


from statsmodels.tsa.arima.model import ARIMA


# In[53]:


model_1 = ARIMA(train,order=(1,1,1))
model_1 = model_1.fit()
model_1.summary()


# In[54]:


start = len(train)
end = len(train)+len(test)-1
pred1 = model_1.predict(start=start,end=end,type='levels')


# In[55]:


plt.figure(figsize=(14,6))
plt.plot(pred1, label='Predicted')
plt.plot(test, label='Test')
plt.legend(loc='best')
plt.show()


# In[56]:


plt.figure(figsize=(14,6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(pred1, label='Predicted')
plt.legend(loc='best')
plt.show()


# In[57]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(pred1,test))

print('Root Mean Squared Error of ARIMA model =',rmse)


# In[58]:


test.mean()


# #### ARIMA with Seasonal order using Maximum likelihood estimation :

# ##### - As the seasonality is large.

# In[59]:


import statsmodels.api as sm


# In[60]:


model_se = sm.tsa.arima.ARIMA(train,order=(1,1,1),seasonal_order=(1,1,1,251))
result = model_se.fit(method='innovations_mle',low_memory=True,cov_type='none')


# In[61]:


pred2 = result.predict(start=start,end=end,type='levels')


# In[62]:



plt.figure(figsize=(14,6))
plt.plot(pred2, label='Predicted')
plt.plot(test, label='Test')
plt.legend(loc='best')
plt.show()


# In[63]:


plt.figure(figsize=(14,6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(pred2, label='Predicted')
plt.legend(loc='best')
plt.show()


# In[64]:


rmse2 = sqrt(mean_squared_error(pred2,test))

print('Root Mean Squared Error of ARIMA with seasonal order using innovations_mle method =',rmse2)


# #### This model has captured seasonality and the rmse is low compared to arima model.
# 
# 

# # SARIMA Model

# In[ ]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Taking half year Seasonality
model_sr = SARIMAX(train,order = (1,1,1),seasonal_order=(1,1,1,126))
model_sr = model_sr.fit()
model_sr.summary()


# In[ ]:


pred3 = model_sr.predict(start = start,end=end,type='levles')
plt.figure(figsize=(14,6))
plt.plot(pred3, label='Predicted')
plt.plot(test, label ='Test')
plt.legend(loc = 'best')
plt.show()


# In[ ]:


plt.figure(figsize=(14,6))
plt.plot(pred3, label='Train')
plt.plot(test, label ='Test')
plt.legend(loc = 'Predicted')
plt.legend(loc = 'best')
plt.show()


# In[ ]:


rmse3=sqrt(mean_squared_error(pred3,test))
print("Root Mean Squared Error of SARIMA Model=",rmse3)


# # Holt-Winters Triple Exponential Smoothing

# In[ ]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[ ]:


model_h = ExponentialSmoothing(train,trend='mul',seasonal='mul',seasonal_periods=252)
model_h = model_h.fit()
model_h.summary()


# In[ ]:


pred3=model_h.predict(start=start,end=end)


# In[ ]:


plt.figure(figsize=(14,6))
plt.plot(pred3, label='Predicted')
plt.plot(test, label ='Test')
plt.legend(loc = 'best')
plt.show()


# In[ ]:


plt.figure(figsize=(14,6))
plt.plot(pred3, label='Train')
plt.plot(test, label ='Test')
plt.legend(loc = 'Predicted')
plt.legend(loc = 'best')
plt.show()


# In[ ]:


rmse3=sqrt(mean_squared_error(pred3,test))
print("Root Mean Squared Error of ARIMA with seasonal order using innovations Model')


# In[ ]:




