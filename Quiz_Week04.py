


# Impute
import pandas as pd
import numpy  as np
from sklearn.model_selection import train_test_split
import statsmodels.api       as     sm
from   sklearn               import metrics

# Import data into a DataFrame.
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE = "babysamp-98.txt"

df = pd.read_table(PATH + FILE)

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head())        # View a snapshot of the data.
print(df.describe())    # View stats including counts which highlight missing values.

def imputeNullValues(colName, df):
    # Create two new column names based on original column name.
    indicatorColName = 'm_'   + colName # Tracks whether imputed.
    imputedColName   = 'imp_' + colName # Stores original & imputed data.

    # Get mean or median depending on preference.
    imputedValue = df[colName].mean()

    # Populate new columns with data.
    imputedColumn  = []
    indictorColumn = []
    for i in range(len(df)):
        isImputed = False

        # mi_OriginalName column stores imputed & original data.
        if(np.isnan(df.loc[i][colName])):
            isImputed = True
            imputedColumn.append(imputedValue)
        else:
            imputedColumn.append(df.loc[i][colName])

        # mi_OriginalName column tracks if is imputed (1) or not (0).
        if(isImputed):
            indictorColumn.append(1)
        else:
            indictorColumn.append(0)

    # Append new columns to dataframe but always keep original column.
    df[indicatorColName] = indictorColumn
    df[imputedColName]   = imputedColumn
    del df[colName]     # Drop column with null values.
    return df

df = imputeNullValues('DadAge', df)
df = imputeNullValues('MomEduc', df)
df = imputeNullValues('prenatalstart', df)
print(df.head(10))

# You can include both the indicator 'm' and imputed 'imp' columns in your model.
# Sometimes both columns boost regression performance and sometimes they do not.
X = df.copy()       # Create separate copy to prevent unwanted tampering of data.
del X['weight']     # Delete target variable.
del X['orig.id']    # Delete unique identifier which is completely random.
del X['preemie']    # Delete non-numeric column.
del X['sex']        # Delete non-numeric column.

print("\n Here are all potential X features - no more nulls exist.")
print(X)
print(X.describe())

# Adding an intercept *** This is required ***. Don't forget this step.
X = sm.add_constant(X)
y = df['weight']
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2)

# Build and evaluate model.
model       = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test) # make the predictions by the model
print(model.summary())
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))











# Here is the code that generates a numeric column called pounds for the price.
import  pandas as pd
import  numpy  as np
from    sklearn.model_selection import train_test_split
import  statsmodels.api       as     sm
from    sklearn               import metrics

# Import data into a DataFrame.
PATH = "/Users/pm/Desktop/DayDocs/data/"
FILE = "carPrice.csv"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
df  = pd.read_csv(PATH + FILE)
del df['reference'] # Remove ID column.

df['pounds'] = 0

def getColumnPosition(df, columnName):
    keys = list(df.keys())
    for i in range(0, len(keys)):
        if(keys[i]==columnName):
            return i

# Create pounds column so it stores an integer.
df['pounds']   = 0
poundsPosition = getColumnPosition(df, 'pounds')

# Generate and store numeric price for algorithm.
for i in range(0, len(df)):
    strPounds = df.iloc[i]['price']
    strPounds = strPounds.replace("£", "")
    strPounds = strPounds.replace(",", "")
    pounds    = int(strPounds)
    df.iat[i, poundsPosition] = pounds

print(df.head())
print(df.describe())




# Here is the code which loads the data and generates the dummy variable columns.

import pandas as pd
import numpy  as np
from sklearn.model_selection import train_test_split
import statsmodels.api       as     sm
from   sklearn               import metrics

# Import data into a DataFrame.
PATH = "/Users/pm/Desktop/DayDocs/data/"
FILE = "babysamp-98.txt"

df = pd.read_table(PATH + FILE)

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head())        # View a snapshot of the data.

# You can include both the indicator 'm' and imputed 'imp' columns in your model.
# Sometimes both columns boost regression performance and sometimes they do not.
X = df.copy()       # Create separate copy to prevent unwanted tampering of data.
del X['weight']     # Delete target variable.
del X['orig.id']    # Delete unique identifier which is completely random.

X = pd.get_dummies(X, columns=['preemie', 'sex'])

print("\n Here are all potential X features - no more nulls exist.")
print(X.head())






# Cross Fold Validation Introduction
# scikit-learn k-fold cross-validation
from numpy import array
from sklearn.model_selection import KFold

# data sample
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

# splits data into 3 randomized folds
kfold = KFold(3, True)

# enumerate splits
for train, test in kfold.split(data):
    print('train: %s, test: %s' % (data[train], data[test]))








# Cross Fold Validation for Logistic Regression
import pandas  as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

PATH = "/Users/pm/Desktop/DayDocs/data/"
CSV_DATA = "computerPurchase.csv"
df = pd.read_csv(PATH + CSV_DATA,
                 skiprows=1,  # Don't include header row as part of data.
                 encoding="ISO-8859-1", sep=',',
                 names=("User ID", "Gender", "Age", "EstimatedSalary",
                        "Purchased"))
count = 0

# Normalize estimated salary
from sklearn.preprocessing import MinMaxScaler
sc_x = MinMaxScaler()

# Fit and transform the data.
dfX = df[['EstimatedSalary', 'Age']]
dfXScaled = sc_x.fit_transform(dfX)

# The 'Purchased' column does not need to be scaled because it is the target
# and it ranges between 0 and 1.
y = df['Purchased']

import numpy as np

# enumerate splits - returns train and test arrays of indexes.
# scikit-learn k-fold cross-validation
from sklearn.model_selection import KFold

# data sample

# prepare cross validation with three folds.
kfold = KFold(3, True)
accuracyList = []
precisionList = []
f1List = []
for train_index, test_index in kfold.split(dfXScaled):
    X_train, X_test = dfXScaled[train_index], dfXScaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Perform logistic regression.
    logisticModel = LogisticRegression(fit_intercept=True,
                                       solver='liblinear')
    # Fit the model.
    logisticModel.fit(X_train, np.ravel(y_train))

    y_pred = logisticModel.predict(X_test)
    y_prob = logisticModel.predict_proba(X_test)

    # Show confusion matrix and accuracy scores.
    cm = pd.crosstab(y_test, y_pred,
                     rownames=['Actual'],
                     colnames=['Predicted'])
    count += 1
    print("\n***K-fold: " + str(count))

    # Calculate accuracy and precision scores and add to the list.
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)

    accuracyList.append(accuracy)
    precisionList.append(precision)

    print('\nAccuracy: ', accuracy)
    print("\nPrecision: ", precision)
    print("\nConfusion Matrix")
    print(cm)

# Show averages of scores over multiple runs.
print("\nAccuracy and Standard Deviation For All Folds:")
print("*********************************************")
print("Average accuracy:  " + str(np.mean(accuracyList)))
print("Accuracy std:      " + str(np.std(accuracyList)))
print("Average precision: " + str(np.mean(precisionList)))
print("Precision std:     " + str(np.std(precisionList)))


recall    	= metrics.recall_score(y_test, y_pred)
f1    	 	= metrics.f1_score(y_test, y_pred)














# Quiz 4
# Azarm Piran

import pandas as pd
import numpy  as np
from sklearn.model_selection import train_test_split
import statsmodels.api       as     sm
from   sklearn               import metrics

# Import data into a DataFrame.
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE = "diet_program.csv"

#df = pd.read_table(PATH + FILE)

df = pd.read_csv(PATH + FILE, delimiter=',')

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head())        # View a snapshot of the data.
print(df.describe())    # View stats including counts which highlight missing values.
print(df)

def imputeNullValues(colName, df):
    # Create two new column names based on original column name.
    indicatorColName = 'm_'   + colName # Tracks whether imputed.
    imputedColName   = 'imp_' + colName # Stores original & imputed data.

    # Get mean or median depending on preference.
    imputedValue = df[colName].mean()

    # Populate new columns with data.
    imputedColumn  = []
    indictorColumn = []
    for i in range(len(df)):
        isImputed = False

        # mi_OriginalName column stores imputed & original data.
        if(np.isnan(df.loc[i][colName])):
            isImputed = True
            imputedColumn.append(imputedValue)
        else:
            imputedColumn.append(df.loc[i][colName])

        # mi_OriginalName column tracks if is imputed (1) or not (0).
        if(isImputed):
            indictorColumn.append(1)
        else:
            indictorColumn.append(0)

    # Append new columns to dataframe but always keep original column.
    df[indicatorColName] = indictorColumn
    df[imputedColName]   = imputedColumn
    del df[colName]     # Drop column with null values.
    return df

df = imputeNullValues('Before', df)
print(df.head(10))


X = df.copy()       # Create separate copy to prevent unwanted tampering of data.
del X['After8weeks']     # Delete target variable.


print("\n Here are all potential X features - no more nulls exist.")
print(X)
print(X.describe())

# Adding an intercept *** This is required ***. Don't forget this step.
X = sm.add_constant(X)
y = df['After8weeks']
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2)

# Build and evaluate model.
model       = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test) # make the predictions by the model
print(model.summary())
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))


import numpy as np

# enumerate splits - returns train and test arrays of indexes.
# scikit-learn k-fold cross-validation
from sklearn.model_selection import KFold

# prepare cross validation with three folds.
kfold    = KFold(3, True)
rmseList = []
bicList  = []
rsquareLst = []
count    = 1

for train_index, test_index in kfold.split(X):
    X_train = X.loc[X.index.isin(train_index)]
    X_test  = X.loc[X.index.isin(test_index)]
    y_train = y.loc[y.index.isin(train_index)]
    y_test  = y.loc[y.index.isin(test_index)]

    # Perform linear regression.
    model = sm.OLS(y_train, X_train).fit()
    print(model.summary())

    y_pred = model.predict(X_test)  # make the predictions by the model
    mse    = metrics.mean_squared_error(y_test, y_pred)
    rmse   = np.sqrt(mse)
    rmseList.append(rmse)
    bic    = model.bic
    bicList.append(bic)
    rsqr   = model.rsquared
    rsquareLst.append(rsqr)

    print("\n***K-fold: " + str(count))
    print("RMSE:     " + str(rmse))
    print("BIC:      " + str(bic))
    print("R^2:      " + str(rsqr))

    count += 1

# Show averages of scores over multiple runs.
print("*********************************************")
print("\nScores for all folds:")
print("*********************************************")
print("RMSE Average :   " + str(np.mean(rmseList)))
print("RMSE SD:         " + str(np.std(rmseList)))
print("BIC Average :    " + str(np.mean(bicList)))
print("BIC SD:          " + str(np.std(bicList)))
print("RSQ Average :    " + str(np.mean(rsquareLst)))
print("RSQ SD:          " + str(np.std(rsquareLst)))
