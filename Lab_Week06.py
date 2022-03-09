# Week 6 Lab
# Azarm Piran | A01195657

# PART A: ARIMA Models
# Autocorrelation and Partial Correlation

# Autocorrelation Function Plot (ACF)

# Partial Auto-Correlation Function (PACF)

# Figure 1: Daily Minimum Temperatures, ACF, PACF
from pandas import read_csv
from matplotlib import pyplot as plt
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE = 'daily-min-temperatures.csv'
series = read_csv(PATH + FILE, header=0, index_col=0)

print(series)
series.plot()
plt.xticks(rotation=45)
plt.title("Daily Minimum Temperatures")
plt.show()

# Plot ACF for stock.
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(series, lags=10)
plt.title("ACF Daily Minimum Temperatures")
plt.show()

plot_pacf(series, lags=10)
plt.title("PACF Daily Minimum Temperatures")
plt.show()


# Exercise 1

# Exercise 2

# Exercise 3



# Exercise 4
import  pandas as pd
import  statsmodels.api as sm
df       = sm.datasets.sunspots.load_pandas().data
df.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
print(df)

del df['YEAR']
print(df)

df.plot()
plt.xticks(rotation=45)
plt.title("Daily Minimum Temperatures")
plt.show()

# Plot ACF for stock.
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df, lags=50)
plt.title("ACF Daily Minimum Temperatures")
plt.show()

plot_pacf(df, lags=20)
plt.title("PACF Daily Minimum Temperatures")
plt.show()



# Autoregressive (AR) Models


# Example 1: Building Autoregression Models
import  pandas as pd
import  matplotlib.pyplot as plt
import  statsmodels.api as sm
from    statsmodels.tsa.arima.model import ARIMA
from    sklearn.metrics import mean_squared_error
import  numpy as np
df       = sm.datasets.sunspots.load_pandas().data
df.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'),freq='A')  # Annual frequency.
print(df)
# Show autocorrelation function.
# General correlation of lags with past lags.
from statsmodels.graphics.tsaplots import plot_acf
#plot_acf(df['SUNACTIVITY'], lags=50)
#plt.show()

# Split the data.
NUM_TEST_YEARS = 10
lenData        = len(df)
print(lenData)
dfTrain        = df.iloc[0:lenData - NUM_TEST_YEARS, :]
#print(dfTrain)
dfTest         = df.iloc[lenData-NUM_TEST_YEARS:,:]
print(dfTest)

def buildModelAndMakePredictions(AR, MA, dfTrain, dfTest):
    # This week we will use the ARIMA model.

    model  = ARIMA(dfTrain['SUNACTIVITY'], order=(AR, 0, MA)).fit()
    print("\n*** Evaluating ARMA(" + str(AR) + ",0," + str(MA) + ")")
    print('Coefficients: %s' %model.params)

    # Strings which can be converted to time stamps are passed in.
    # For this case the entire time range for the test set is represented.
    predictions = model.predict('1999-12-31', '2008-12-31', dynamic=True)
    rmse = np.sqrt(mean_squared_error(dfTest['SUNACTIVITY'].values,
                                      np.array(predictions)))
    print('Test RMSE: %.3f' % rmse)
    print('Model AIC %.3f' % model.aic)
    print('Model BIC %.3f' % model.bic)
    return model, predictions

print(dfTest)
arma_mod20, predictionsARMA_20 = buildModelAndMakePredictions(2, 0, dfTrain, dfTest)

plt.plot(dfTest.index, dfTest['SUNACTIVITY'],
         label='Actual Values', color='blue')
plt.plot(dfTest.index, predictionsARMA_20,
         label='Predicted Values AR(2)', color='orange')
plt.legend(loc='best')
plt.show()





# Example 2: Comparing Predictions
import  pandas as pd
import  matplotlib.pyplot as plt
import  statsmodels.api as sm
from    statsmodels.tsa.arima.model import ARIMA
from    sklearn.metrics import mean_squared_error
import  numpy as np
df       = sm.datasets.sunspots.load_pandas().data
df.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'),freq='A')  # Annual frequency.

# Show autocorrelation function.
# General correlation of lags with past lags.
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df['SUNACTIVITY'], lags=50)
plt.show()

# Split the data.
NUM_TEST_YEARS = 10
lenData        = len(df)
dfTrain        = df.iloc[0:lenData - NUM_TEST_YEARS, :]
dfTest         = df.iloc[lenData-NUM_TEST_YEARS:,:]

def buildModelAndMakePredictions(AR, MA, dfTrain, dfTest):
    # This week we will use the ARIMA model.

    model  = ARIMA(dfTrain['SUNACTIVITY'], order=(AR, 0, MA)).fit()
    print("\n*** Evaluating ARMA(" + str(AR) + ",0," + str(MA) + ")")
    print('Coefficients: %s' %model.params)

    # Strings which can be converted to time stamps are passed in.
    # For this case the entire time range for the test set is represented.
    predictions = model.predict('1999-12-31', '2008-12-31', dynamic=True)
    rmse = np.sqrt(mean_squared_error(dfTest['SUNACTIVITY'].values,
                                      np.array(predictions)))
    print('Test RMSE: %.3f' % rmse)
    print('Model AIC %.3f' % model.aic)
    print('Model BIC %.3f' % model.bic)
    return model, predictions

print(dfTest)
arma_mod20, predictionsARMA_20 = buildModelAndMakePredictions(2, 0, dfTrain, dfTest)
arma_mod30, predictionsARMA_30 = buildModelAndMakePredictions(3, 0, dfTrain, dfTest)

plt.plot(dfTest.index, dfTest['SUNACTIVITY'],
         label='Actual Values', color='blue')
plt.plot(dfTest.index, predictionsARMA_20,
         label='Predicted Values AR(2)', color='orange')
plt.plot(dfTest.index, predictionsARMA_30,
         label='Predicted Values AR(3)', color='brown')
plt.legend(loc='best')
plt.show()



# Moving Average Model (MA Model)

# Example 3: Comparing ARMA Plots
import  pandas as pd
import  matplotlib.pyplot as plt
import  statsmodels.api as sm
from    statsmodels.tsa.arima.model import ARIMA
from    sklearn.metrics import mean_squared_error
import  numpy as np
df       = sm.datasets.sunspots.load_pandas().data
df.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'),freq='A')  # Annual frequency.

# Show autocorrelation function.
# General correlation of lags with past lags.
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df['SUNACTIVITY'], lags=50)
plt.show()

# Split the data.
NUM_TEST_YEARS = 10
lenData        = len(df)
dfTrain        = df.iloc[0:lenData - NUM_TEST_YEARS, :]
dfTest         = df.iloc[lenData-NUM_TEST_YEARS:,:]

def buildModelAndMakePredictions(AR, MA, dfTrain, dfTest):
    # This week we will use the ARIMA model.

    model  = ARIMA(dfTrain['SUNACTIVITY'], order=(AR, 0, MA)).fit()
    print("\n*** Evaluating ARMA(" + str(AR) + ",0," + str(MA) + ")")
    print('Coefficients: %s' %model.params)

    # Strings which can be converted to time stamps are passed in.
    # For this case the entire time range for the test set is represented.
    predictions = model.predict('1999-12-31', '2008-12-31', dynamic=True)
    rmse = np.sqrt(mean_squared_error(dfTest['SUNACTIVITY'].values,
                                      np.array(predictions)))
    print('Test RMSE: %.3f' % rmse)
    print('Model AIC %.3f' % model.aic)
    print('Model BIC %.3f' % model.bic)
    return model, predictions

print(dfTest)
arma_mod20, predictionsARMA_20 = buildModelAndMakePredictions(2, 0, dfTrain, dfTest)
arma_mod30, predictionsARMA_30 = buildModelAndMakePredictions(3, 0, dfTrain, dfTest)
arma_mod305, predictionsARMA_305 = buildModelAndMakePredictions(3,5,dfTrain, dfTest)

plt.plot(dfTest.index, dfTest['SUNACTIVITY'],
         label='Actual Values', color='blue')
plt.plot(dfTest.index, predictionsARMA_20,
         label='Predicted Values AR(2)', color='orange')
plt.plot(dfTest.index, predictionsARMA_30,
         label='Predicted Values AR(3)', color='brown')
plt.plot(dfTest.index, predictionsARMA_305,
         label='Predicted Values ARMA(35)', color='black')
plt.legend(loc='best')
plt.show()




# Handling Random Noise

# Stationarity

# Differencing

# Example 4: Differencing
import pandas as pd
import datetime

import matplotlib.pyplot as plt
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE = 'shampoo.csv'
df   = pd.read_csv(PATH + FILE, index_col=0)
df.info()

# Plot data before differencing.
df.plot()
plt.xticks(rotation=45)
plt.show()

# Perform differencing.
df = df.diff()

# Plot data after differencing.
plt.plot(df)
plt.xticks(rotation=75)
plt.show()


# Avoiding Too Much Differencing

# Augmented Dickey-Fuller Test (ADF)

# Example 6: ADF Test
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Import data
df = pd.read_csv(
"https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv", \
                  names=['value'], header=0)
print(df)
df.value.plot()
plt.title("www usage")
plt.show()

from statsmodels.tsa.stattools import adfuller
result = adfuller(df.value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])



# Exercise 6
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Import data
df = pd.read_csv(
"https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv",names=['value'], header=0)
print(df)
df.value.plot()
plt.title("www usage")
plt.show()

from statsmodels.tsa.stattools import adfuller
result = adfuller(df.value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
# After this point, because p-value is greater than 0.05, we need to do more differencing because data is not stationary

# Perform differencing.
df = df.diff()

# Plot data after differencing.
plt.plot(df)
plt.xticks(rotation=75)
plt.show()

# Now, we want to perform ADF test on our data

from statsmodels.tsa.stattools import adfuller
result = adfuller(df.value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
# Now, at this point the p-value is 0.070268 which is still greater than 0.05




# ARIMA for Differencing

# Example 7: Optimizing p,d,q parameters with auto_arima()

from   pandas import read_csv
import pmdarima as pm

PATH   = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
series = read_csv(PATH + 'daily-min-temperatures.csv', header=0, index_col=0)
print(series)
model  = pm.auto_arima(series, start_p=1, start_q=1,
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












# Exercise 7 - version 1
from   pandas import read_csv
import pmdarima as pm
from    statsmodels.tsa.arima.model import ARIMA
import  matplotlib.pyplot as plt

PATH   = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
df = read_csv(PATH + 'daily-total-female-births.csv', header=0, index_col=0)

print(df)

NUM_TEST_SAMPLES = 5
lenData        = len(df)
#print(lenData)
dfTrain        = df.iloc[0:lenData - NUM_TEST_SAMPLES, :]
#print(dfTrain)
dfTest         = df.iloc[lenData - NUM_TEST_SAMPLES:,:]
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
rmse = np.sqrt(mean_squared_error(dfTest['Births'].values,
                                      np.array(predictions)))
print('Test RMSE: %.3f' % rmse)



plt.plot(dfTest.index, dfTest['Births'],
         label='Actual Values', color='blue')
plt.plot(dfTest.index, predictions,
         label='Predicted Values', color='orange')
plt.legend(loc='best')
plt.show()






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
rmse = np.sqrt(mean_squared_error(dfTest['Births'].values,
                                      np.array(predictions)))
print('Test RMSE: %.3f' % rmse)


plt.plot(dfTest.index, dfTest['Births'],
         label='Actual Values', color='blue')
plt.plot(dfTest.index, predictions,
         label='Predicted Values', color='orange')
plt.legend(loc='best')
plt.show()







# Example 8: Weighted Moving Average
import datetime
import pandas_datareader  as pdr
import matplotlib.pyplot as plt

def getStock(stk, ttlDays):
    numDays = int(ttlDays)
    dt = datetime.date.today()
    dtPast = dt + datetime.timedelta(days=-numDays)
    df = pdr.get_data_yahoo(stk,
                            start=datetime.datetime(dtPast.year, dtPast.month,
                                                    dtPast.day),
                            end=datetime.datetime(dt.year, dt.month, dt.day))
    return df

df = getStock('AMD', 1100)
print(df)

rolling_mean  = df['Close'].rolling(window=20).mean()
rolling_mean2 = df['Close'].rolling(window=50).mean()

#plt.figure(figsize=(10,30))
df['Close'].plot(label='AMD Close ', color='gray', alpha=0.3)
rolling_mean.plot(label='AMD 20 Day SMA', style='--', color='orange')
rolling_mean2.plot(label='AMD 50 Day SMA', style='--',color='magenta')

plt.legend()
plt.show()




# Exercise 11
import matplotlib.pyplot as plt
import pandas as pd
PATH   = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
originalDf = pd.read_csv(PATH + 'daily-min-temperatures.csv', header=0, index_col=0)

#print(originalDf)
NUM_TEST_DAYS = 80
lenData        = len(originalDf)
dfTrain        = originalDf.iloc[0:lenData - NUM_TEST_DAYS, :]
df             = originalDf.iloc[lenData-NUM_TEST_DAYS:,:]


rolling_mean  = df['Temp'].rolling(window=5).mean()
rolling_mean2 = df['Temp'].rolling(window=10).mean()

df['Temp'].plot(label='AMD Temp ', color='gray', alpha=0.3)
rolling_mean.plot(label='AMD 5 Day SMA', style='--', color='orange')
rolling_mean2.plot(label='AMD 10 Day SMA', style='--',color='magenta')
plt.xticks(rotation=90)
plt.legend()
plt.show()