

# Time Shifting
import pandas as pd

co2 = [342.76, 343.96, 344.82, 345.82, 347.24, 348.09, 348.66, 347.90, 346.27]
df  = pd.DataFrame({'CO2':co2}, index=pd.date_range('09-01-2020',
                                periods=len(co2), freq='B'))
df['CO2_t-1'] = df['CO2'].shift(periods=1)
print(df)








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

#print(df)
dfBack = backShiftColumns(df, 'Open')
print(dfBack)
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


###################################################################################


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
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test, predictions)))

plt.plot(y_test.index, predictions, label='Predicted', marker='o', color='orange')
plt.plot(y_test.index, y_test,      label='Actual',    marker='o', color='blue')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.show()




####################################
# Quiz 5


import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

FILE = "AirPassengers.csv"

PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"

df = pd.read_csv(PATH + FILE)

df = df.set_index('date')

print(df)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


NUM_TIME_STEPS  = 3
TEST_DAYS       = 7
##################################################################

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Disable warnings.
pd.options.mode.chained_assignment = None


def backShiftColumns(df, colName):
    # Create time step columns for 'Open' price.
    for i in range(1, NUM_TIME_STEPS + 1):
        newColumnName     = colName + 't-' + str(i)
        df[newColumnName] = df[colName].shift(periods=i)
    return df

#print(df)
dfBack = backShiftColumns(df, 'value')
#print(dfBack)

dfBack = dfBack.dropna() # Remove nulls after back-shifting.
y  = dfBack[['valuet-3']]
X  = dfBack[['value','valuet-1','valuet-2']]

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


################ RMSE without Scaling is 66.85307435200201 (when NUM_TIME_STEPS = 2)
# Model and make predictions.
model       = sm.OLS(y_train, X_train).fit()
print(model.summary())
predictions = model.predict(X_test)

from sklearn  import metrics
print('Root Mean Squared Error without scaling:',np.sqrt(metrics.mean_squared_error(y_test, predictions)))


################ RMSE with Scaling is 66.85307435200198 (when NUM_TIME_STEPS = 2)
from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler()
X_trainScaled = scalerX.fit_transform(X_train)
X_testScaled  = scalerX.transform(X_test)

scalery = StandardScaler()
y_trainScaled = scalery.fit_transform(y_train)

model       = sm.OLS(y_trainScaled, X_trainScaled).fit()
print(model.summary())
predictionsScaled = model.predict(X_testScaled)


predictions = scalery.inverse_transform(predictionsScaled)

# Show RMSE and plot the data.
from sklearn  import metrics
print('Root Mean Squared Error with Scaling:',np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# RMSe with scaling is a little bit smaller so it is better.  Lower values of RMSE indicate better fit.
# when I assign 3 to NUM_TIME_STEPS  , the p>|t| for value is 0.023 which is smaller than 0.05 so I try one more
# when I assign 4 to NUM_TIME_STEPS, p-value for value is 0.141 and for valuet-1 is 0.858 so I think 3 will be better

# after deciding to have NUM_TIME_STEPS = 3 now we have RMSE without scaling 49.53457231725868 and with scaling 49.534572317258714. as we can see with scaling is smaller so it is better