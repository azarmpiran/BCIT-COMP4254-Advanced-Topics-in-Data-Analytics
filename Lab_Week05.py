# Lab Week 5
# Azarm Piran


import pandas as pd
# Using Pandas to_Datetime function
from datetime import datetime
dt = datetime(year=2015, month=7, day=4)
#print(dt)

import pandas as pd
print(pd.to_datetime('7/8/1952'))

import pandas as pd
print(pd.to_datetime('7/8/1952', dayfirst=True))





# Azarm
# I need to ask a question about this, and make sure it is correct
# Exercise 1
import pandas as pd
import datetime
data = { 'string_dates':['7/8/1952', '7/9/1952'] }
df   = pd.DataFrame(data=data)

print(df)
print("\n")

# adding a new column
df['Date'] = pd.to_datetime(df['string_dates'])
print(df)

df.dtypes








# DateTime Indicies
# Example 1: Setting a DateTime Index
df = df.set_index('Date')



# Exercise 2
import pandas as pd
import datetime
data = { 'string_dates':['7/8/1952', '7/9/1952'] }
df   = pd.DataFrame(data=data)

# adding a new column
df['Date'] = pd.to_datetime(df['string_dates'])
df = df.set_index('Date')
print(df)

df.dtypes




# Example 2: Setting Frequency and Range

import pandas as pd

co2 = [
342.76, 343.96, 344.82, 345.82, 347.24, 348.09, 348.66, 347.90, 346.27, 344.21,
342.88, 342.58, 343.99, 345.31, 345.98, 346.72, 347.63, 349.24, 349.83, 349.10,
347.52, 345.43, 344.48, 343.89, 345.29, 346.54, 347.66, 348.07, 349.12, 350.55,
351.34, 350.80, 349.10, 347.54, 346.20, 346.20, 347.44, 348.67]

df = pd.DataFrame({'CO2':co2}, index=pd.date_range('01-01-2002',periods=len(co2), freq='MS'))
print(df)



# Exercise 3
import pandas as pd

co2 = [
342.76, 343.96, 344.82, 345.82, 347.24, 348.09, 348.66, 347.90, 346.27, 344.21,
342.88, 342.58, 343.99, 345.31, 345.98, 346.72, 347.63, 349.24, 349.83, 349.10,
347.52, 345.43, 344.48, 343.89, 345.29, 346.54, 347.66, 348.07, 349.12, 350.55,
351.34, 350.80, 349.10, 347.54, 346.20, 346.20, 347.44, 348.67]

print(co2)

df = pd.DataFrame({'CO2':co2}, index=pd.date_range('09-01-2021',periods=len(co2), freq='W'))
print(df) # Now, we have a data frame with a Index date column which increments weekly starting from September 21



# Exercise 3 - Version 2

import pandas as pd
df = []
print(df)

j = 0
for i in range(0,100):
    df.append(j)
    j = j + 3

print(df)

df2 = pd.DataFrame({'df':df}, index=pd.date_range('09-01-2021',periods=len(df), freq='W'))
print(df2)







# Example 3: Plotting a Single Time Series
import pandas as pd
import matplotlib.pyplot as plt
data = { 'string_dates':['7/5/1972', '7/1/1973',  '7/1/1974', '7/1/1975'],
         'sales':  [15,18,26,42],
         'cost':   [3,12,20,37],
         'exports':[8,12,16,18],
         'imports':[3,16,20,32]}

df         = pd.DataFrame(data=data)
df['date'] = pd.to_datetime(df['string_dates'])
df         = df.set_index('date')

# Adjust font-size.
plt.rcParams['font.size'] = 16

# Draw multiple plots. Specifies 1 row, 2 columns.
plt.subplots(nrows=1, ncols=2,  figsize=(14,7))

plt.subplot(1,2,1) # Drawn top left.
df['sales'].plot(marker='o', color='black')

plt.subplot(1,2,1) # Draw top left.
df['cost'].plot(marker='o',   color='red')

# Set attributes for top left plot.
plt.xticks(rotation=70)
plt.title("Sales in Millions $ July 1972 to July 1975")
plt.legend()

plt.subplot(1,2,2) # Draw top right.
df['exports'].plot(marker='o',   color='green')

# Set attributes for top left plot.
plt.xticks(rotation=70)
plt.title("Exports in Millions $ July 1972 to July 1975")
plt.legend()
plt.show()




# Exercise 4
import pandas as pd
import matplotlib.pyplot as plt
data = { 'string_dates':['7/5/1972', '7/1/1973',  '7/1/1974', '7/1/1975'],
         'sales':  [15,18,26,42],
         'cost':   [3,12,20,37],
         'exports':[8,12,16,18],
         'imports':[3,16,20,32]}

df         = pd.DataFrame(data=data)
df['date'] = pd.to_datetime(df['string_dates'])
df         = df.set_index('date')

# Adjust font-size.
plt.rcParams['font.size'] = 16

# Draw multiple plots. Specifies 1 row, 2 columns.
plt.subplots(nrows=1, ncols=2,  figsize=(14,7))

plt.subplot(1,2,1) # Drawn top left.
df['sales'].plot(marker='o', color='black')

plt.subplot(1,2,1) # Draw top left.
df['cost'].plot(marker='o',   color='red')

# Set attributes for top left plot.
plt.xticks(rotation=70)
plt.title("Sales in Millions $ July 1972 to July 1975")
plt.legend()

plt.subplot(1,2,2) # Draw top right.
df['exports'].plot(marker='o',   color='green')

plt.subplot(1,2,2) # Draw top right.
df['imports'].plot(marker='o',   color='blue')

# Set attributes for top left plot.
plt.xticks(rotation=70)
plt.title("Imports/Exports $M July 1972 to July 1975")
plt.legend()
plt.show()




# Comparing Financial Data Over Time


# Example 4: Comparing Percentage Change
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader  as pdr
import datetime
import pandas             as pd
import matplotlib.pyplot  as plt
import matplotlib.dates   as mdates

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def getStock(stk, ttlDays):
    numDays = int(ttlDays)
    dt      = datetime.date.today()
    dtPast  = dt + datetime.timedelta(days=-numDays)
    df      = pdr.get_data_yahoo(stk,
    start   = datetime.datetime(dtPast.year, dtPast.month, dtPast.day),
    end     = datetime.datetime(dt.year, dt.month, dt.day))
    return df

NUM_DAYS = 30
dfGoogle = getStock('GOOGL', NUM_DAYS)
dfApple  = getStock('AAPL', NUM_DAYS)

# Adjust font-size.
plt.rcParams['font.size'] = 16

# Draw multiple plots. Specifies 1 row, 2 columns.
plt.subplots(nrows=1, ncols=2,  figsize=(14,7))

plt.subplot(1,2,1) # Drawn top left.
dfGoogle['Close'].plot(marker='o',  color='blue', label="Google")
dfApple['Close'].plot(marker='o',   color='red', label="Apple")
plt.xticks(rotation=70)
plt.title("Apple vs. Google  Change")
plt.legend()

plt.subplot(1,2,2) # Drawn top right.
dfGoogle['Close'].pct_change().plot(marker='o',  color='blue', label="Google")
dfApple['Close'].pct_change().plot(marker='o',   color='red', label="Apple")
plt.xticks(rotation=70)
plt.title("Apple vs. Google Percent Change")
plt.legend()
plt.show()





# Exercise 5
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader  as pdr
import datetime
import pandas             as pd
import matplotlib.pyplot  as plt
import matplotlib.dates   as mdates

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def getStock(stk, ttlDays):
    numDays = int(ttlDays)
    dt      = datetime.date.today()
    dtPast  = dt + datetime.timedelta(days=-numDays)
    df      = pdr.get_data_yahoo(stk,
    start   = datetime.datetime(dtPast.year, dtPast.month, dtPast.day),
    end     = datetime.datetime(dt.year, dt.month, dt.day))
    return df

NUM_DAYS = 30
dfGoogle = getStock('GOOGL', NUM_DAYS)
dfApple  = getStock('AAPL', NUM_DAYS)
dfMSFT   = getStock('MSFT', NUM_DAYS)

# Adjust font-size.
plt.rcParams['font.size'] = 16

# Draw multiple plots. Specifies 1 row, 2 columns.
plt.subplots(nrows=1, ncols=2,  figsize=(14,7))

plt.subplot(1,2,1) # Drawn top left.
dfGoogle['Close'].plot(marker='o',  color='blue', label="Google")
dfApple['Close'].plot(marker='o',   color='red', label="Apple")
dfMSFT['Close'].plot(marker='o',   color='green', label="Microsoft")
plt.xticks(rotation=70)
plt.title("Apple vs. Google vs. Microsoft Change")
plt.legend()

plt.subplot(1,2,2) # Drawn top right.
dfGoogle['Close'].pct_change().plot(marker='o',  color='blue', label="Google")
dfApple['Close'].pct_change().plot(marker='o',   color='red', label="Apple")
dfMSFT['Close'].pct_change().plot(marker='o',   color='green', label="Microsoft")
plt.xticks(rotation=70)
plt.title("Apple vs. Google vs. Microsoft Percent Change")
plt.legend()
plt.show()



# Time Shifting

# Example 5: Time Shifting
import pandas as pd

co2 = [342.76, 343.96, 344.82, 345.82, 347.24, 348.09, 348.66, 347.90, 346.27]

df  = pd.DataFrame({'CO2':co2}, index=pd.date_range('09-01-2020',periods=len(co2), freq='B'))

df['CO2_t-1'] = df['CO2'].shift(periods=1)
print(df)



# Exercise 6
import pandas as pd

co2 = [342.76, 343.96, 344.82, 345.82, 347.24, 348.09, 348.66, 347.90, 346.27]

df  = pd.DataFrame({'CO2':co2}, index=pd.date_range('09-01-2020',periods=len(co2), freq='B'))

df['CO2_t-1'] = df['CO2'].shift(periods=1)
df['CO2_t-2'] = df['CO2'].shift(periods=2)
#print(df)
df  = df.dropna()
print(df)




# Least Squares Regression with Time Series

# Example 6: Least Squares Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##################################################################
# CONFIGURATION SECTION
NUM_DAYS        = 1200
STOCK_SYMBOL    = 'AAPL'
NUM_TIME_STEPS  = 2
TEST_DAYS       = 7
##################################################################

# Get stock data.
import datetime
import pandas_datareader  as pdr

def getStock(stk, ttlDays):
    numDays = int(ttlDays)
    dt = datetime.date.today()
    dtPast = dt + datetime.timedelta(days=-numDays)
    df = pdr.get_data_yahoo(stk,
                            start=datetime.datetime(dtPast.year, dtPast.month,
                                  dtPast.day),
                            end=datetime.datetime(dt.year, dt.month, dt.day))
    return df

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Disable warnings.
pd.options.mode.chained_assignment = None

# Get stocks for past 1200 days.
df = getStock(STOCK_SYMBOL, NUM_DAYS)
print(df)

def backShiftColumns(df, colName):
    # Create time step columns for 'Open' price.
    for i in range(1, NUM_TIME_STEPS + 1):
        newColumnName     = colName + 't-' + str(i)
        df[newColumnName] = df[colName].shift(periods=i)
    return df

dfBack = backShiftColumns(df, 'Open')
#print(dfBack)
dfBack = backShiftColumns(dfBack, 'Close')
#print(dfBack)
dfBack = dfBack.dropna() # Remove nulls after back-shifting.
#print(dfBack)
y  = dfBack[['Open']]
print(y)
X  = dfBack[['Closet-1']]
print(X)

# Add intercept for OLS regression.
import statsmodels.api       as sm
X = sm.add_constant(X)

# Split into test and train sets. The test data must be
# the latest data range.
lenData = len(X)
X_train = X[0:lenData-TEST_DAYS]
y_train = y[0:lenData-TEST_DAYS]
X_test  = X[lenData-TEST_DAYS:]
y_test  = y[lenData-TEST_DAYS:]
print(X_test)

################ SECTION B
# Model and make predictions.
model       = sm.OLS(y_train, X_train).fit()
print(model.summary())
predictions = model.predict(X_test)

################ SECTION C
# Show RMSE and plot the data.
from sklearn  import metrics
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))

plt.plot(y_test.index, predictions, label='Predicted', marker='o', color='orange')
plt.plot(y_test.index, y_test,      label='Actual',    marker='o', color='blue')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.show()






















# Exercise 7

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##################################################################
# CONFIGURATION SECTION
NUM_DAYS        = 1200
STOCK_SYMBOL    = 'AAPL'
NUM_TIME_STEPS  = 2
TEST_DAYS       = 7
##################################################################

# Get stock data.
import datetime
import pandas_datareader  as pdr

def getStock(stk, ttlDays):
    numDays = int(ttlDays)
    dt = datetime.date.today()
    dtPast = dt + datetime.timedelta(days=-numDays)
    df = pdr.get_data_yahoo(stk,
                            start=datetime.datetime(dtPast.year, dtPast.month,
                                  dtPast.day),
                            end=datetime.datetime(dt.year, dt.month, dt.day))
    return df

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Disable warnings.
pd.options.mode.chained_assignment = None

# Get stocks for past 1200 days.
df = getStock(STOCK_SYMBOL, NUM_DAYS)

def backShiftColumns(df, colName):
    # Create time step columns for 'Open' price.
    for i in range(1, NUM_TIME_STEPS + 1):
        newColumnName     = colName + 't-' + str(i)
        df[newColumnName] = df[colName].shift(periods=i)
    return df

dfBack = backShiftColumns(df, 'Open')
dfBack = backShiftColumns(dfBack, 'Close')
dfBack = dfBack.dropna() # Remove nulls after back-shifting.
y  = dfBack[['Open']]
X  = dfBack[['Closet-1']]

# Add intercept for OLS regression.
import statsmodels.api       as sm
X = sm.add_constant(X)

# Split into test and train sets. The test data must be
# the latest data range.
lenData = len(X)
X_train = X[0:lenData-TEST_DAYS]
y_train = y[0:lenData-TEST_DAYS]
X_test  = X[lenData-TEST_DAYS:]
y_test  = y[lenData-TEST_DAYS:]
print(X_test)

################ SECTION B
# Model and make predictions.
model       = sm.OLS(y_train, X_train).fit()
print(model.summary())
predictions = model.predict(X_test)

################ SECTION C
# Show RMSE and plot the data.
from sklearn  import metrics
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))

plt.plot(y_test.index, predictions, label='Predicted', marker='o', color='orange')
plt.plot(y_test.index, y_test,      label='Actual',    marker='o', color='blue')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.show()










# Exercise 8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##################################################################
# CONFIGURATION SECTION
NUM_DAYS        = 1200
STOCK_SYMBOL    = 'AAPL'
NUM_TIME_STEPS  = 3
TEST_DAYS       = 7
##################################################################

# Get stock data.
import datetime
import pandas_datareader  as pdr

def getStock(stk, ttlDays):
    numDays = int(ttlDays)
    dt = datetime.date.today()
    dtPast = dt + datetime.timedelta(days=-numDays)
    df = pdr.get_data_yahoo(stk,
                            start=datetime.datetime(dtPast.year, dtPast.month,
                                  dtPast.day),
                            end=datetime.datetime(dt.year, dt.month, dt.day))
    return df

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Disable warnings.
pd.options.mode.chained_assignment = None

# Get stocks for past 1200 days.
df = getStock(STOCK_SYMBOL, NUM_DAYS)

def backShiftColumns(df, colName):
    # Create time step columns for 'Open' price.
    for i in range(1, NUM_TIME_STEPS + 1):
        newColumnName     = colName + 't-' + str(i)
        df[newColumnName] = df[colName].shift(periods=i)
    return df

dfBack = backShiftColumns(df, 'Open')
dfBack = backShiftColumns(dfBack, 'Close')
dfBack = dfBack.dropna() # Remove nulls after back-shifting.
y  = dfBack[['Open']]
X  = dfBack
del X['Open']
print(X)

# Add intercept for OLS regression.
import statsmodels.api       as sm
X = sm.add_constant(X)

# Split into test and train sets. The test data must be
# the latest data range.
lenData = len(X)
X_train = X[0:lenData-TEST_DAYS]
y_train = y[0:lenData-TEST_DAYS]
X_test  = X[lenData-TEST_DAYS:]
y_test  = y[lenData-TEST_DAYS:]
print(X_test)

################ SECTION B
# Model and make predictions.
model       = sm.OLS(y_train, X_train).fit()
print(model.summary())
predictions = model.predict(X_test)

################ SECTION C
# Show RMSE and plot the data.
from sklearn  import metrics
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))

plt.plot(y_test.index, predictions, label='Predicted', marker='o', color='orange')
plt.plot(y_test.index, y_test,      label='Actual',    marker='o', color='blue')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.show()

# Based on the first  model, I am going to re assign the X and remove all the variables that are not statistically significant
del X['Adj Close']
del X['Opent-2']
del X['Opent-3']
del X['Closet-2']
print(X)

# Add intercept for OLS regression.
import statsmodels.api       as sm
X = sm.add_constant(X)

# Split into test and train sets. The test data must be
# the latest data range.
lenData = len(X)
X_train = X[0:lenData-TEST_DAYS]
y_train = y[0:lenData-TEST_DAYS]
X_test  = X[lenData-TEST_DAYS:]
y_test  = y[lenData-TEST_DAYS:]
print(X_test)

################ SECTION B
# Model and make predictions.
model       = sm.OLS(y_train, X_train).fit()
print(model.summary())
predictions = model.predict(X_test)

################ SECTION C
# Show RMSE and plot the data.
from sklearn  import metrics
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))

plt.plot(y_test.index, predictions, label='Predicted', marker='o', color='orange')
plt.plot(y_test.index, y_test,      label='Actual',    marker='o', color='blue')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.show()







# Example 7: Implementing StandardScaler with Time Series Data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##################################################################
# CONFIGURATION SECTION
NUM_DAYS        = 1200
STOCK_SYMBOL    = 'AAPL'
NUM_TIME_STEPS  = 2
TEST_DAYS       = 7
##################################################################

# Get stock data.
import datetime
import pandas_datareader  as pdr

def getStock(stk, ttlDays):
    numDays = int(ttlDays)
    dt = datetime.date.today()
    dtPast = dt + datetime.timedelta(days=-numDays)
    df = pdr.get_data_yahoo(stk,
                            start=datetime.datetime(dtPast.year, dtPast.month,
                                  dtPast.day),
                            end=datetime.datetime(dt.year, dt.month, dt.day))
    return df

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Disable warnings.
pd.options.mode.chained_assignment = None

# Get stocks for past 1200 days.
df = getStock(STOCK_SYMBOL, NUM_DAYS)

def backShiftColumns(df, colName):
    # Create time step columns for 'Open' price.
    for i in range(1, NUM_TIME_STEPS + 1):
        newColumnName     = colName + 't-' + str(i)
        df[newColumnName] = df[colName].shift(periods=i)
    return df

dfBack = backShiftColumns(df, 'Open')
dfBack = backShiftColumns(dfBack, 'Close')
dfBack = dfBack.dropna() # Remove nulls after back-shifting.
y  = dfBack[['Open']]
X  = dfBack[['Closet-1']]

# Add intercept for OLS regression.
import statsmodels.api       as sm
X = sm.add_constant(X)

# Split into test and train sets. The test data must be
# the latest data range.
lenData = len(X)
X_train = X[0:lenData-TEST_DAYS]
y_train = y[0:lenData-TEST_DAYS]
X_test  = X[lenData-TEST_DAYS:]
y_test  = y[lenData-TEST_DAYS:]
print(X_test)

################ SECTION B
# Model and make predictions.
from sklearn.preprocessing import StandardScaler
# Always fit scaler with training data.
scalerX = StandardScaler()
X_trainScaled = scalerX.fit_transform(X_train)
X_testScaled  = scalerX.transform(X_test)

# Scale y data separately when performing linear regression.
# You only need  to scale the taining data.
scalery = StandardScaler()
y_trainScaled = scalery.fit_transform(y_train)

# Model and make predictions.
model       = sm.OLS(y_trainScaled, X_trainScaled).fit()
print(model.summary())
predictionsScaled = model.predict(X_testScaled)

# Notice we return the predictions back to the actual size.
predictions = scalery.inverse_transform(predictionsScaled)


################ SECTION C
# Show RMSE and plot the data.
from sklearn  import metrics
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))

plt.plot(y_test.index, predictions, label='Predicted', marker='o', color='orange')
plt.plot(y_test.index, y_test,      label='Actual',    marker='o', color='blue')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.show()






# Example 9:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##################################################################
# CONFIGURATION SECTION
NUM_DAYS        = 1200
STOCK_SYMBOL    = 'AAPL'
NUM_TIME_STEPS  = 2
TEST_DAYS       = 7
##################################################################

# Get stock data.
import datetime
import pandas_datareader  as pdr

def getStock(stk, ttlDays):
    numDays = int(ttlDays)
    dt = datetime.date.today()
    dtPast = dt + datetime.timedelta(days=-numDays)
    df = pdr.get_data_yahoo(stk,
                            start=datetime.datetime(dtPast.year, dtPast.month,
                                  dtPast.day),
                            end=datetime.datetime(dt.year, dt.month, dt.day))
    return df

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Disable warnings.
pd.options.mode.chained_assignment = None

# Get stocks for past 1200 days.
df = getStock(STOCK_SYMBOL, NUM_DAYS)

def backShiftColumns(df, colName):
    # Create time step columns for 'Open' price.
    for i in range(1, NUM_TIME_STEPS + 1):
        newColumnName     = colName + 't-' + str(i)
        df[newColumnName] = df[colName].shift(periods=i)
    return df

dfBack = backShiftColumns(df, 'Open')
dfBack = backShiftColumns(dfBack, 'Close')
dfBack = dfBack.dropna() # Remove nulls after back-shifting.
y  = dfBack[['Open']]
X  = dfBack[['Closet-1']]

# Add intercept for OLS regression.
import statsmodels.api       as sm
X = sm.add_constant(X)

# Split into test and train sets. The test data must be
# the latest data range.
lenData = len(X)
X_train = X[0:lenData-TEST_DAYS]
y_train = y[0:lenData-TEST_DAYS]
X_test  = X[lenData-TEST_DAYS:]
y_test  = y[lenData-TEST_DAYS:]
print(X_test)

################ SECTION B
# Model and make predictions.
from sklearn.preprocessing import RobustScaler
# Always fit scaler with training data.
scalerX = RobustScaler()
X_trainScaled = scalerX.fit_transform(X_train)
X_testScaled  = scalerX.transform(X_test)

# Scale y data separately when performing linear regression.
# You only need  to scale the taining data.
scalery = RobustScaler()
y_trainScaled = scalery.fit_transform(y_train)

# Model and make predictions.
model       = sm.OLS(y_trainScaled, X_trainScaled).fit()
print(model.summary())
predictionsScaled = model.predict(X_testScaled)

# Notice we return the predictions back to the actual size.
predictions = scalery.inverse_transform(predictionsScaled)









# Example 10: Changing to MinMaxScaler for Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##################################################################
# CONFIGURATION SECTION
NUM_DAYS        = 1200
STOCK_SYMBOL    = 'AAPL'
NUM_TIME_STEPS  = 2
TEST_DAYS       = 7
##################################################################

# Get stock data.
import datetime
def getStock(stk, ttlDays):
    numDays = int(ttlDays)
    dt = datetime.date.today()
    dtPast = dt + datetime.timedelta(days=-numDays)
    df = pdr.get_data_yahoo(stk,
                            start=datetime.datetime(dtPast.year, dtPast.month,
                                  dtPast.day),
                            end=datetime.datetime(dt.year, dt.month, dt.day))
    return df

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Disable warnings.
pd.options.mode.chained_assignment = None

# Get stocks for past 1200 days.
df = getStock(STOCK_SYMBOL, NUM_DAYS)

def backShiftColumns(df, colName):
    # Create time step columns for 'Open' price.
    for i in range(1, NUM_TIME_STEPS + 1):
        newColumnName     = colName + 't-' + str(i)
        df[newColumnName] = df[colName].shift(periods=i)
    return df

dfBack = backShiftColumns(df, 'Open')
dfBack = backShiftColumns(dfBack, 'Close')
dfBack = dfBack.dropna() # Remove nulls after back-shifting.
y  = dfBack[['Open']]
X  = dfBack[['Closet-1']]

# Add intercept for OLS regression.
import statsmodels.api       as sm
X = sm.add_constant(X)

# Split into test and train sets. The test data must be
# the latest data range.
lenData = len(X)
X_train = X[0:lenData-TEST_DAYS]
y_train = y[0:lenData-TEST_DAYS]
X_test  = X[lenData-TEST_DAYS:]
y_test  = y[lenData-TEST_DAYS:]
print(X_test)

################ SECTION B
# Model and make predictions.
from sklearn.preprocessing import MinMaxScaler
# Always fit scaler with training data.
scalerX = MinMaxScaler()
X_trainScaled = scalerX.fit_transform(X_train)
X_testScaled  = scalerX.transform(X_test)

# Scale y data separately when performing linear regression.
# You only need  to scale the taining data.
scalery = MinMaxScaler()
y_trainScaled = scalery.fit_transform(y_train)

# Model and make predictions.
model       = sm.OLS(y_trainScaled, X_trainScaled).fit()
print(model.summary())
predictionsScaled = model.predict(X_testScaled)

# Notice we return the predictions back to the actual size.
predictions = scalery.inverse_transform(predictionsScaled.reshape(-1,1))









# Exercise 10

# Example 10: Changing to MinMaxScaler for Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader  as pdr

##################################################################
# CONFIGURATION SECTION
NUM_DAYS = 1200
STOCK_SYMBOL = 'AAPL'
NUM_TIME_STEPS = 2
TEST_DAYS = 7
##################################################################

# Get stock data.
import datetime


def getStock(stk, ttlDays):
    numDays = int(ttlDays)
    dt = datetime.date.today()
    dtPast = dt + datetime.timedelta(days=-numDays)
    df = pdr.get_data_yahoo(stk,
                            start=datetime.datetime(dtPast.year, dtPast.month,
                                                    dtPast.day),
                            end=datetime.datetime(dt.year, dt.month, dt.day))
    return df


# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Disable warnings.
pd.options.mode.chained_assignment = None

# Get stocks for past 1200 days.
df = getStock(STOCK_SYMBOL, NUM_DAYS)


def backShiftColumns(df, colName):
    # Create time step columns for 'Open' price.
    for i in range(1, NUM_TIME_STEPS + 1):
        newColumnName = colName + 't-' + str(i)
        df[newColumnName] = df[colName].shift(periods=i)
    return df


dfBack = backShiftColumns(df, 'Open')
dfBack = backShiftColumns(dfBack, 'Close')
dfBack = dfBack.dropna()  # Remove nulls after back-shifting.
y = dfBack[['Open']]
X = dfBack[['Closet-1']]

# Add intercept for OLS regression.
import statsmodels.api       as sm

X = sm.add_constant(X)

# Split into test and train sets. The test data must be
# the latest data range.
lenData = len(X)
X_train = X[0:lenData - TEST_DAYS]
y_train = y[0:lenData - TEST_DAYS]
X_test = X[lenData - TEST_DAYS:]
y_test = y[lenData - TEST_DAYS:]
print(X_test)

################ SECTION B
# Model and make predictions.
from sklearn.preprocessing import MinMaxScaler

# Always fit scaler with training data.
scalerX = MinMaxScaler()
X_trainScaled = scalerX.fit_transform(X_train)
X_testScaled = scalerX.transform(X_test)

# Scale y data separately when performing linear regression.
# You only need  to scale the taining data.
scalery = MinMaxScaler()
y_trainScaled = scalery.fit_transform(y_train)

# Model and make predictions.
model = sm.OLS(y_trainScaled, X_trainScaled).fit()
print(model.summary())
predictionsScaled = model.predict(X_testScaled)

# Notice we return the predictions back to the actual size.
predictions = scalery.inverse_transform(predictionsScaled.reshape(-1, 1))

################ SECTION C
# Show RMSE and plot the data.
from sklearn  import metrics
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))

plt.plot(y_test.index, predictions, label='Predicted', marker='o', color='orange')
plt.plot(y_test.index, y_test,      label='Actual',    marker='o', color='blue')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.show()


















# Next Day Prediction

# Example 11: Predicting the Future
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##################################################################
# CONFIGURATION SECTION
NUM_DAYS        = 1200
STOCK_SYMBOL    = 'AAPL'
NUM_TIME_STEPS  = 2
TEST_DAYS       = 7
##################################################################

# Get stock data.
import datetime
def getStock(stk, ttlDays):
    numDays = int(ttlDays)
    dt = datetime.date.today()
    dtPast = dt + datetime.timedelta(days=-numDays)
    df = pdr.get_data_yahoo(stk,
                            start=datetime.datetime(dtPast.year, dtPast.month,
                                  dtPast.day),
                            end=datetime.datetime(dt.year, dt.month, dt.day))
    return df

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Disable warnings.
pd.options.mode.chained_assignment = None

# Get stocks for past 1200 days.
df = getStock(STOCK_SYMBOL, NUM_DAYS)

def backShiftColumns(df, colName):
    # Create time step columns for 'Open' price.
    for i in range(1, NUM_TIME_STEPS + 1):
        newColumnName     = colName + 't-' + str(i)
        df[newColumnName] = df[colName].shift(periods=i)
    return df

dfBack = backShiftColumns(df, 'Open')
dfBack = backShiftColumns(dfBack, 'Close')
dfBack = dfBack.dropna() # Remove nulls after back-shifting.
y  = dfBack[['Open']]
X  = dfBack[['Closet-1']]

# Add intercept for OLS regression.
import statsmodels.api       as sm
X = sm.add_constant(X)

# Split into test and train sets. The test data must be
# the latest data range.
lenData = len(X)
X_train = X[0:lenData-TEST_DAYS]
y_train = y[0:lenData-TEST_DAYS]
X_test  = X[lenData-TEST_DAYS:]
y_test  = y[lenData-TEST_DAYS:]
print(X_test)

################ SECTION B
# Model and make predictions.
from sklearn.preprocessing import MinMaxScaler
# Always fit scaler with training data.
scalerX = MinMaxScaler()
X_trainScaled = scalerX.fit_transform(X_train)
X_testScaled  = scalerX.transform(X_test)

# Scale y data separately when performing linear regression.
# You only need  to scale the taining data.
scalery = MinMaxScaler()
y_trainScaled = scalery.fit_transform(y_train)

# Model and make predictions.
model       = sm.OLS(y_trainScaled, X_trainScaled).fit()
print(model.summary())
predictionsScaled = model.predict(X_testScaled)

# Notice we return the predictions back to the actual size.
predictions = scalery.inverse_transform(predictionsScaled.reshape(-1,1))

def dayAheadPrediction(df, scalerX, scalerY, model):
    lastRowIndex = len(df) - 1
    close_t_1 = df.iloc[lastRowIndex]['Close']
    X = pd.DataFrame(data={
        "Closet-1":[close_t_1]
    })

    # add_constant() doesn't work if there is
    # already a column with variance=0.
    # To fix this use 'has_constant' parameter.
    XC = sm.add_constant(X, has_constant='add')
    X_scaled = scalerX.transform(XC)
    scaledPrediction = model.predict(X_scaled)
    prediction = scalerY.inverse_transform(scaledPrediction.reshape(-1, 1))
    print("Predicted price: " + str(prediction))
    return prediction

dayAheadPrediction(df, scalerX, scalery, model)








# Time Series Decomposition

# Example 12: Multiplicative Decomposition Visualization
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas                 as pd
import matplotlib.pyplot      as plt

# Import data.
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE = "drugSales.csv"
df   = pd.read_csv(PATH + FILE,  parse_dates=['date'], index_col='date')
type(df.index)

# Perform decomposition using multiplicative decomposition.
tseries = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend="freq")

tseries.plot()
plt.show()





# Example 13: Numeric Multiplicative Decomposition
# Extract the Components ----
# Actual Values = Product of (Seasonal * Trend * Resid)
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas                 as pd
import matplotlib.pyplot      as plt

# Import data.
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE = "drugSales.csv"
df   = pd.read_csv(PATH + FILE,  parse_dates=['date'], index_col='date')
type(df.index)

# Perform decomposition using multiplicative decomposition.
tseries = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend="freq")

tseries.plot()
plt.show()

dfComponents = pd.concat([tseries.seasonal, tseries.trend,
                          tseries.resid, tseries.observed], axis=1)
dfComponents.columns = ['seas', 'trend', 'resid', 'actual_values']
print(dfComponents.head())













# Exercise 11
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas                 as pd
import matplotlib.pyplot      as plt

# Import data.
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE = "drugSales.csv"
df   = pd.read_csv(PATH + FILE,  parse_dates=['date'], index_col='date')
type(df.index)

# Perform decomposition using multiplicative decomposition.
tseries = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend="freq")

#tseries.plot()
#plt.show()

dfComponents = pd.concat([tseries.seasonal, tseries.trend,tseries.resid, tseries.observed], axis=1)
dfComponents.columns = ['seas', 'trend', 'resid', 'actual_values']
#print(dfComponents.head())

print(dfComponents)

y1 = dfComponents.iloc[0]['seas'] * dfComponents.iloc[0]['trend'] * dfComponents.iloc[0]['resid']
print(y1)


y2 = dfComponents.iloc[1]['seas'] * dfComponents.iloc[1]['trend'] * dfComponents.iloc[1]['resid']
print(y2)





# Additive Decomposition

# Example 14: Additive Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas                 as pd
import matplotlib.pyplot      as plt

# Import data.
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE = "drugSales.csv"
df   = pd.read_csv(PATH + FILE,  parse_dates=['date'], index_col='date')
type(df.index)

# Perform decomposition using multiplicative decomposition.
tseries = seasonal_decompose(df['value'], model='additive', extrapolate_trend="freq")

tseries.plot()
plt.show()







# Example 15: Plotting Seasonal Detail

from statsmodels.tsa.seasonal import seasonal_decompose
import pandas                 as pd
import matplotlib.pyplot      as plt
import matplotlib.dates       as mdates
# Import data.
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE = "drugSales.csv"
df   = pd.read_csv(PATH + FILE,  parse_dates=['date'], index_col='date')
type(df.index)

fig, ax = plt.subplots()

# Perform decomposition using multiplicative decomposition.
tseries  = seasonal_decompose(df['value'], model='multiplicative',
                              extrapolate_trend='freq')
trend    = tseries.trend
seasonal = tseries.seasonal

# Set vertical major grid.
ax.xaxis.set_major_locator(mdates.YearLocator(day=1))
ax.xaxis.grid(True, which = 'major', linewidth = 1, color = 'black')

# Set vertical minor grid.
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1,4,7,10),bymonthday=1))
ax.xaxis.grid(True, which = 'minor', linewidth = 1, color = 'red')

start, end = '2005-01', '2009-12'
ax.plot(seasonal.loc[start:end], color='green')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
plt.setp(ax.xaxis.get_minorticklabels(), rotation=70)

ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
plt.title("Seasonal Drug Sales")
plt.show()





# Exercise 12
# Multiplicative Decomposition Visualization
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas                 as pd
import matplotlib.pyplot      as plt
# Import data.
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE = "AirPassengers.csv"
df   = pd.read_csv(PATH + FILE,  parse_dates=['date'], index_col='date')
type(df.index)

# Perform decomposition using multiplicative decomposition.
tseries = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend="freq")

tseries.plot()
plt.show()



# Additive Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas                 as pd
import matplotlib.pyplot      as plt
# Import data.
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE = "AirPassengers.csv"
df   = pd.read_csv(PATH + FILE,  parse_dates=['date'], index_col='date')
type(df.index)

# Perform decomposition using multiplicative decomposition.
tseries = seasonal_decompose(df['value'], model='additive', extrapolate_trend="freq")

tseries.plot()
plt.show()
