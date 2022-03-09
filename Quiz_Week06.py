# Azarm Piran | A01195657

# Auto-ARIMA

from   pandas import read_csv
import pmdarima as pm

PATH   = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
series = read_csv(PATH + 'daily-min-temperatures.csv', header=0, index_col=0)

model  = pm.auto_arima(series, start_p=1, start_q=1,
                        test='adf',
                        max_p=3, max_q=3, m=12,
                        start_P=0, seasonal=False,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True)
print(model.summary())
predictions = model.predict(n_periods=9, dynamic=True)
print(predictions)






# Exercise 7
from   pandas import read_csv
import pmdarima as pm
from   pandas import read_csv
import pmdarima as pm
from    statsmodels.tsa.arima.model import ARIMA
import  matplotlib.pyplot as plt

PATH   = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
df = read_csv(PATH + 'daily-total-female-births.csv', header=0, index_col=0)

print(df)

NUM_TEST_SAMPLES = 5
lenData        = len(df)
dfTrain        = df.iloc[0:lenData - NUM_TEST_SAMPLES, :]
dfTest         = df.iloc[lenData - NUM_TEST_SAMPLES:,:]

print(lenData)
print(dfTrain)
print(dfTest)

# We are using ARIMA model with raining data which is dfTrain
model  = pm.auto_arima(dfTrain, start_p=1, start_q=1,
                        test='adf',
                        max_p=3, max_q=3, m=12,
                        start_P=0, seasonal=False,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True)
print(model.summary())
predictions = model.predict(n_periods=5, dynamic=True)
print("Here is the prediction:")
print(predictions)

import  numpy as np
from    sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(dfTest['Births'].values,np.array(predictions)))
print('Test RMSE: %.3f' % rmse)





# Quiz 6
# Azarm Piran | A01195657

import pandas as pd
import pmdarima as pm
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE = 'sales_over_time.csv'

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
df = pd.read_csv(PATH + FILE)

# Convert string column to date column.
df['dt'] = pd.to_datetime(df['date'])

# Group by date string column and get total revenue for each date.
df = df.groupby('dt')['revenue'].sum().reset_index(name='revenue')
# Set date as index.
df = df.set_index('dt')
print(df)

NUM_TEST_SAMPLES = 9
lenData = len(df)
dfTrain = df.iloc[0:lenData - NUM_TEST_SAMPLES, :]
dfTest = df.iloc[lenData - NUM_TEST_SAMPLES:, :]


# We are using ARIMA model with raining data which is dfTrain
model  = pm.auto_arima(dfTrain, start_p=1, start_q=1,
                        test='adf',
                        max_p=3, max_q=3, m=12,
                        start_P=0, seasonal=False,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True)
print(model.summary())
predictions = model.predict(n_periods=9, dynamic=True)
print("Here is the prediction:")
print(predictions)

import  numpy as np
from    sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(dfTest['revenue'].values,np.array(predictions)))
print('Test RMSE: %.3f' % rmse)

print('the auto-regressive (AR) component is 2. There is no differencing. The moving average (MA) component is 3.')
print('AR,p = 2 and I,d = 0 and MA,q = 3')