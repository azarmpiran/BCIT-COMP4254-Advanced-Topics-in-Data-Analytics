# Lab Week 2
# Azarm Piran - A01195657


# Example 1: Plotting a Scatter Plot
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# Create DataFrame.
dataSet = {'rain':   [0.2, 0.32, 0.38, 0.41, 0.43, 0.5, 0.49, 0.7, 0.3, 0.52],
           'growth': [0.1, 0.15, 0.4, 0.6, 0.44, 0.55, 0.56, 0.6, 0.22, 0.48] }

df = pd.DataFrame(dataSet, columns= ['rain', 'growth'])
print(df)
# Make the font bigger.
font = {'size' : 22}
plt.rc('font', **font)

# Store x and y values.
X = df['rain']
y = df['growth']

# Adjust relative width and height of plot.
figure(figsize=(14, 6))

plt.scatter(X, y)
plt.ylabel("Growth")
plt.xlabel("Rain")
plt.title("Rain versus Growth")
plt.show()






# Example 2: Generating Test and Training Data
import statsmodels.api as sm
# Adding an intercept *** This is requried ***. Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X = sm.add_constant(X)

# Create training set with 70% of data and test set with 30% of data.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)
print("X_train")
print(X_train)
print("\ny_train")
print(y_train)
print("\nX_test")
print(X_test)
print("\ny_test")
print(y_test)



# Exercise 1
import statsmodels.api as sm
X = sm.add_constant(X)

# Create training set with 70% of data and test set with 30% of data.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
print("X_train")
print(X_train)
print("\ny_train")
print(y_train)
print("\nX_test")
print(X_test)
print("\ny_test")
print(y_test)









# Example 3: Linear Regression in Python
import statsmodels.api as sm
# Adding an intercept *** This is requried ***. Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X = sm.add_constant(X)

# Create training set with 60% of data and test set with 40% of data.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)

from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import numpy as np

model = sm.OLS(y_train, X_train).fit()
print(model.summary())

predictions = model.predict(X_test) # make the predictions by the model
mse = mean_squared_error(predictions, y_test)
rmse = np.sqrt(mse)
print("Root mean square error: " + str(rmse))





# Exercise 2











# Example 4: Evaluating the Best Fit Line
# Create DataFrame.
import pandas as pd
dataSet = {'rain':   [0.2, 0.32, 0.38, 0.41, 0.43, 0.5, 0.49, 0.7, 0.3, 0.52],
           'growth': [0.1, 0.15, 0.4, 0.6, 0.44, 0.55, 0.56, 0.6, 0.22, 0.48] }

df = pd.DataFrame(dataSet, columns= ['rain', 'growth'])
print(df)
# Make the font bigger.
font = {'size' : 22}
plt.rc('font', **font)

# Store x and y values.
X = df['rain']
y = df['growth']

# Adjust relative width and height of plot.
figure(figsize=(14, 6))
import statsmodels.api as sm

# Adding an intercept *** This is requried ***. Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X = sm.add_constant(X)

# Create training set with 60% of data and test set with 40% of data.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)

from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import numpy as np

model = sm.OLS(y_train, X_train).fit()
print(model.summary())

predictions = model.predict(X_test) # make the predictions by the model
mse = mean_squared_error(predictions, y_test)
rmse = np.sqrt(mse)
print("Root mean square error: " + str(rmse))

plt.plot(X_test['rain'], predictions, '-o', label="predictions", color='orange')
plt.scatter(X_test['rain'], y_test, label="actual")
plt.legend(loc="best")
plt.title("Actual vs. Predicted")
plt.show()




# Exercise 3
# True


# Exercise 4
# False





# Example 5: Simple Exploratory Data Analysis

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

FOLDER = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE = 'gmat_vs_lsat.csv'
df = pd.read_csv(FOLDER + FILE)
#print(df)
print(df.head())
print(df.info())
print(df.describe())

# Make the plot fonts bigger.
font = {'size': 22}
plt.rc('font', **font)

def drawHistogram(attributeList, attributeName):
    plt.hist(attributeList)
    plt.title(attributeName + " Distribution")
    plt.xlabel = attributeName
    plt.ylabel = "Frequency"
    plt.show()

drawHistogram(df['GMAT'], 'GMAT')



# Exercise 5
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

FOLDER = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE = 'gmat_vs_lsat.csv'
df = pd.read_csv(FOLDER + FILE)
#print(df)
print(df.head())
print("Info:")
print(df.info())
print("\n")
print("Describe:")
print(df.describe())

# Make the plot fonts bigger.
font = {'size': 22}
plt.rc('font', **font)

def drawHistogram(attributeList, attributeName):
    plt.hist(attributeList)
    plt.title(attributeName + " Distribution")
    plt.xlabel = attributeName
    plt.ylabel = "Frequency"
    plt.show()

#drawHistogram(df['GMAT'], 'GMAT')
drawHistogram(df['LSAT'], 'LSAT')





# Exercise 6
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

FOLDER = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE = 'gmat_vs_lsat.csv'
df = pd.read_csv(FOLDER + FILE)

import statsmodels.api as sm
X = df['GMAT']
Y = df['LSAT']
X = sm.add_constant(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.75)

from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import numpy as np

model = sm.OLS(Y_train, X_train).fit()
print("\n")
print("Model Summary:")
print(model.summary())

predictions = model.predict(X_test)
mse = mean_squared_error(predictions, Y_test)
rmse = np.sqrt(mse)
print("Root mean square error: " + str(rmse))

plt.plot(X_test['GMAT'], predictions, '-o', label="predictions", color='orange')
plt.scatter(X_test['GMAT'], Y_test, label="actual")
plt.legend(loc="best")
plt.title("Actual vs. Predicted")
plt.show()

















# Multiple Linear Regression






# Example 6: Half-Baked Exploratory Data Analysis
import pandas as pd
import matplotlib.pyplot as plt

PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
CSV_DATA = "winequality.csv"

dataset = pd.read_csv(PATH + CSV_DATA)

print(dataset.head(3))
print(dataset.describe())

# Plot the heatmap.
import seaborn as sns
heatmap = sns.heatmap(dataset.corr() [['quality']].sort_values(by='quality',ascending=False), vmin=-1, vmax=1, annot=True)
print(heatmap)
heatmap.set_title('Wine Quality',fontdict={'fontsize':18}, pad=16)
plt.show()





# Example 7: Calculating the Wine Quality Frequencies
import pandas as pd
import matplotlib.pyplot as plt

PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
CSV_DATA = "winequality.csv"

dataset = pd.read_csv(PATH + CSV_DATA)

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(dataset.head(3))
print(dataset.describe())

# Plot the heatmap.
import seaborn as sns
heatmap = sns.heatmap(dataset.corr() [['quality']].sort_values(by='quality',ascending=False), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Wine Quality',fontdict={'fontsize':18}, pad=16)
plt.show()
# Show counts.
print(dataset['quality'].value_counts(ascending=True))





# Example 8: Multiple Linear Regression for Wine Quality: Model A
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
# Here we have all the data reading it from the CSV file in dataset Var - if you print dataset you will see it
print(dataset)

# Now we assign all columns except 'quality' to the X - These are predictor variables
X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
             'pH', 'sulphates','alcohol']]
print(X)
# Don't forget this step.
X = sm.add_constant(X)

# Now we save quality in Y variable which is the target variable - because we wan to predict the quality of wine based on all other predictor variables in the dataset
y = dataset['quality']

# Here we split the data to train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# We use OLS model - passing y_train and x_train to the function
model = sm.OLS(y_train, X_train).fit()
# We need to pass X_test to the predict function
predictions = model.predict(X_test) # make the predictions by the model

print(model.summary())
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, predictions)))




# Exercise 7








# Exercise 8









# Example 9: Multiple Linear Regression for Wine Quality: Model B
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error

# Same as example 8 but we eliminate some of the predictor variables because they were insignificant
X = dataset[['volatile acidity', 'chlorides', 'total sulfur dioxide','sulphates','alcohol']]

print(X)
# Don't forget this step.
X = sm.add_constant(X)


y = dataset['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# We use OLS model - passing y_train and x_train to the function
model = sm.OLS(y_train, X_train).fit()
# We need to pass X_test to the predict function
predictions = model.predict(X_test) # make the predictions by the model

print(model.summary())
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, predictions)))





# Example 10: Multiple Linear Regression for Wine Quality: Model C
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error

X = dataset[[ 'volatile acidity', 'chlorides',  'sulphates','alcohol']]

# Don't forget this step.
X = sm.add_constant(X)

y = dataset['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = sm.OLS(y_train, X_train).fit()

predictions = model.predict(X_test) # make the predictions by the model

print(model.summary())
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, predictions)))










# Example 11: Graphing Actual, Predicted and Residual Values
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error

X = dataset[[ 'volatile acidity', 'chlorides',  'sulphates','alcohol']]

# Don't forget this step.
X = sm.add_constant(X)

y = dataset['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = sm.OLS(y_train, X_train).fit()

predictions = model.predict(X_test) # make the predictions by the model

print(model.summary())
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, predictions)))
import matplotlib.pyplot as plt
def plotPredictionVsActual(title, y_test, predictions):
    plt.scatter(y_test, predictions)
    plt.legend()
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title('Predicted (Y) vs. Actual (X): ' + title)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.show()

plotPredictionVsActual("Wine Quality", y_test, predictions)








# Example 12: Manual Predictions
def getWineQuality(actualQuality, volatileAcidity, chlorides, sulphates, alcohol):
    wineQuality = 2.7939 - 1.2395 *volatileAcidity - 1.7142 *chlorides + \
                  + 0.8517*sulphates + 0.2974*alcohol;
    print("Wine Quality actual: " + str(actualQuality) + "   predicted: "
        + str(wineQuality))

getWineQuality(6, 0.47, 0.171, 0.76, 10.8)
getWineQuality(5, 0.82, 0.095, 0.53, 9.6)
getWineQuality(7, 0.29, 0.063, 0.84, 11.7)
















# Exercise 9
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api       as sm
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
CSV_DATA = "USA_Housing.csv"
dataset  = pd.read_csv(PATH + CSV_DATA, sep=',')
# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(dataset.head())

# First 3 rows of dataframe:
print(dataset.head(3))






# Exercise 10
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
CSV_DATA = "USA_Housing.csv"
dataset = pd.read_csv(PATH + CSV_DATA, sep=',')

# Statistical summary for the numerical columns
print(dataset.describe())





# Exercise 11
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
CSV_DATA = "USA_Housing.csv"
dataset = pd.read_csv(PATH + CSV_DATA, sep=',')
import seaborn as sns
heatmap = sns.heatmap(dataset.corr() [['Price']].sort_values(by='Price',ascending=False), vmin=-1, vmax=1, annot=True)
heatmap.set_title('House Price',
                  fontdict={'fontsize':10}, pad=10)
plt.show()






# Exercise 12












# Exercise 13
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
CSV_DATA = "USA_Housing.csv"
dataset = pd.read_csv(PATH + CSV_DATA, sep=',')
print(dataset)


# model A:
# Predictor variables:
X = dataset[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms','Area Population']]
print(X)
# Don't forget this step.
X = sm.add_constant(X)
# Target variables:
y = dataset['Price']
# Here we split the data to train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# We use OLS model - passing y_train and x_train to the function
model = sm.OLS(y_train, X_train).fit()
# We need to pass X_test to the predict function
predictions = model.predict(X_test)
print(model.summary())
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, predictions)))
# We need to eliminate the "Avg. Area Number of Bedrooms" variable because it is statistically insignificant: P>|t| = 0.134


# model B:
# We adjust our model so it does not have "Avg. Area Number of Bedrooms" anymore
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
CSV_DATA = "USA_Housing.csv"
dataset = pd.read_csv(PATH + CSV_DATA, sep=',')
print(dataset)
X = dataset[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Area Population']]
print(X)
X = sm.add_constant(X)
y = dataset['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test)
print(model.summary())
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, predictions)))
# Now, we can see that all the predictor variables in the model B are statistically significant.
# Now, we need to pay attention to coef column
# We have these values for coef column: -2647000.00, 21.67, 165800.00, 121600.00, 15.28
# 21.67 and 15.28 are the smallest compare to other three numbers so they do not have a big effect on the target. That is why we eliminate them in the next model


# model C:
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
CSV_DATA = "USA_Housing.csv"
dataset = pd.read_csv(PATH + CSV_DATA, sep=',')
print(dataset)
X = dataset[['Avg. Area House Age', 'Avg. Area Number of Rooms']]
print(X)
X = sm.add_constant(X)
y = dataset['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test)
print("OLS model summary:")
print("\n")
print(model.summary())
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, predictions)))







# Exercise 14





# Exercise 15




# Exercise 16
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
CSV_DATA = "USA_Housing.csv"
dataset = pd.read_csv(PATH + CSV_DATA, sep=',')
print(dataset)
X = dataset[['Avg. Area House Age', 'Avg. Area Number of Rooms']]
print(X)
X = sm.add_constant(X)
y = dataset['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test)
print(model.summary())
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, predictions)))
import matplotlib.pyplot as plt
def plotPredictionVsActual(title, y_test, predictions):
    plt.scatter(y_test, predictions)
    plt.legend()
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title('Predicted (Y) vs. Actual (X): ' + title)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.show()
plotPredictionVsActual("Housing Price", y_test, predictions)




# Exercise 17








# Exercise 18
def getHousingPrice(actualPrice, houseAge, numberOfRooms):
    housePrice = -555700 + 162700 * houseAge + 116800 * numberOfRooms;
    print("House Price actual: " + str(actualPrice) + "   predicted: "
        + str(housePrice))


getHousingPrice(1260616.807,7.18823609451864,5.58672866482765)
getHousingPrice(1545154.813,4.42367179,8.167688003)
getHousingPrice(1306674.66,5.443156467,8.517512711)
