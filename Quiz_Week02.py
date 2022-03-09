
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
heatmap.set_title('House Price',fontdict={'fontsize':10}, pad=10)
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




# Exercise 18
def getHousingPrice(actualPrice, houseAge, numberOfRooms):
    housePrice = -555700 + 162700 * houseAge + 116800 * numberOfRooms;
    print("House Price actual: " + str(actualPrice) + "   predicted: "
        + str(housePrice))


getHousingPrice(1260616.807,7.18823609451864,5.58672866482765)
getHousingPrice(1545154.813,4.42367179,8.167688003)
getHousingPrice(1306674.66,5.443156467,8.517512711)



# Quiz 2


import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api       as sm
from sklearn.metrics import mean_squared_error
import numpy as np
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
CSV_DATA = "medicalInsuranceCharges.csv"
df = pd.read_csv(PATH + CSV_DATA, sep=',')
# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df)


# model A:
# Predictor variables:
X = df[['age', 'bmi', 'children', 'gender','us_state','smokes']]
print(X)
X = sm.add_constant(X)
y = df['insuranceClaims']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test)
print(model.summary())
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, predictions)))
# We need to eliminate "children" and "gender" and "us_state" in our next model because they are not statistically significant. p>|t| is not zero

# model B:
# Predictor variables:
X = df[['age', 'bmi','smokes']]
print(X)
X = sm.add_constant(X)
y = df['insuranceClaims']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test)
print(model.summary())
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, predictions)))
# now that all the p>|t| are zero, we need to pay attention to coef column
# as we can see all the 3 numbers in coef column for this column are large numbers which means all 3 variables has a considerable impact on target so we keep them all

# Part B
import matplotlib.pyplot as plt
def plotPredictionVsActual(title, y_test, predictions):
    plt.scatter(y_test, predictions)
    plt.legend()
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title('Predicted (Y) vs. Actual (X): ' + title)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.show()
plotPredictionVsActual("Medical Insurance Changes over 5 years", y_test, predictions)







