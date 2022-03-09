# Azarm Piran
# A01195657
# Lab 7



# Decision Trees for Classification

# Example 1: Manually Building a Decision Tree

# Calculating Gini Impurity

# Example 2: Decision Tree for Classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
CSV     = "bill_authentication.csv"
dataset = pd.read_csv(PATH + CSV)
X       = dataset.drop('Class', axis=1)
y       = dataset['Class']
print(dataset)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

def showAccuracyScores(y_test, y_pred):
    print("\nModel Evaluation")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("")
    tn = cm[0][0]
    fp = cm[0][1]
    tp = cm[1][1]
    fn = cm[1][0]
    accuracy  = (tp + tn)/(tn + fp + tp + fn)
    precision = tp/(tp + fp)
    recall    = tp/(tp + fn)
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))

showAccuracyScores(y_test, y_pred)



# Visualizing the Decision Tree

# Example 3: Visualizing the Decision Tree Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
CSV     = "bill_authentication.csv"
dataset = pd.read_csv(PATH + CSV)
print(dataset)
# Class is the target variable
X       = dataset.drop('Class', axis=1)
y       = dataset['Class']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

def showAccuracyScores(y_test, y_pred):
    print("\nModel Evaluation")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("")
    tn = cm[0][0]
    fp = cm[0][1]
    tp = cm[1][1]
    fn = cm[1][0]
    accuracy  = (tp + tn)/(tn + fp + tp + fn)
    precision = tp/(tp + fp)
    recall    = tp/(tp + fn)
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))

showAccuracyScores(y_test, y_pred)

from sklearn.tree import plot_tree
fig, ax = plt.subplots(figsize=(20, 10))

plot_tree(classifier.fit(X_train, y_train), max_depth=4, fontsize=4)
a = plot_tree(classifier,
              feature_names=['Variance', 'Skewness', 'Kurtosis', 'Entropy'],
              class_names='Class',
              filled=True,
              rounded=True,
              fontsize=14)
plt.show()








# Exercise 4 - Version 2
import sklearn.datasets as datasets
from   sklearn.tree import DecisionTreeClassifier
from   sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd

# check for the sklearn version, it has to be 0.21
import sklearn
print(sklearn.__version__)
breast_cancer = datasets.load_breast_cancer()

data = pd.DataFrame(breast_cancer.data, columns=[breast_cancer.feature_names])
data['Target'] = pd.Series(data=breast_cancer.target, index=data.index)

#print data.keys()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.20,random_state=0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y_pred)

from sklearn.metrics import classification_report, confusion_matrix

def showAccuracyScores(y_test, y_pred):
    print("\nModel Evaluation")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("")
    tn = cm[0][0]
    fp = cm[0][1]
    tp = cm[1][1]
    fn = cm[1][0]
    accuracy  = (tp + tn)/(tn + fp + tp + fn)
    precision = tp/(tp + fp)
    recall    = tp/(tp + fn)
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))

showAccuracyScores(y_test, y_pred)

from sklearn.tree import plot_tree
fig, ax = plt.subplots(figsize=(20, 10))

plot_tree(classifier.fit(X_train, y_train), max_depth=4, fontsize=4)
a = plot_tree(classifier,
              feature_names=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension'],
              class_names= 'Target',
              filled=True,
              rounded=True,
              fontsize=14)
plt.show()




















# Exercise 4

import sklearn.datasets as datasets
from   sklearn.tree import DecisionTreeClassifier
from   sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd

# check for the sklearn version, it has to be 0.21
import sklearn
print(sklearn.__version__)
breast_cancer = datasets.load_breast_cancer()

data = breast_cancer
df = pd.DataFrame(data.data, columns=data.feature_names)
df.head()
print(df)
print(df.head)
print(df. columns)

# Write the dataset into excel to see all tge column
# create excel writer object
writer = pd.ExcelWriter('Cancer.xlsx')
# write dataframe to excel
df.to_excel(writer)
# save the excel
writer.save()

df['Target'] = pd.Series(data=breast_cancer.target, index=data.index)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.20,random_state=0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

def showAccuracyScores(y_test, y_pred):
    print("\nModel Evaluation")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("")
    tn = cm[0][0]
    fp = cm[0][1]
    tp = cm[1][1]
    fn = cm[1][0]
    accuracy  = (tp + tn)/(tn + fp + tp + fn)
    precision = tp/(tp + fp)
    recall    = tp/(tp + fn)
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))

showAccuracyScores(y_test, y_pred)

from sklearn.tree import plot_tree
fig, ax = plt.subplots(figsize=(20, 10))

plot_tree(classifier.fit(X_train, y_train), max_depth=4, fontsize=4)
a = plot_tree(classifier,
              feature_names=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension'],
              class_names='Class',
              filled=True,
              rounded=True,
              fontsize=14)
plt.show()




























# Performing Linear Regression with a DecisionTreeRegressor

# Example 5: Decision Tree for Regression (Predicting Gas Consumption)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
CSV  = "petrol_consumption.csv"
df   = pd.read_csv(PATH + CSV)

# Show all columns.
pd.set_option('display.max_columns', None)
# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

X = df.drop('Petrol_Consumption', axis=1)
y = df['Petrol_Consumption']
print(df)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print("\nActual versus predicted values")
print(df)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



# Random Forests

# Example 6: Random Forests

# Pandas is used for data manipulation
import pandas as pd
from   sklearn.metrics import mean_squared_error

# Read in data and display first 5 rows
features = pd.read_csv('C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\temperatures.csv')

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(features)

# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)

# Display the first 5 rows of the last 12 columns.
print(features.head(5))

# Use numpy to convert to arrays
import numpy as np

# Labels are the values we want to predict
labels = np.array(features['actual'])

# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('actual', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels =\
    train_test_split(features, labels, test_size = 0.25, random_state = 42)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')

# Print out the mean square error.
mse = mean_squared_error(test_labels, predictions)
print('RMSE:', np.sqrt(mse))





# Example 7: Show the Importance’s

# Pandas is used for data manipulation
import pandas as pd
from   sklearn.metrics import mean_squared_error

# Read in data and display first 5 rows
features = pd.read_csv('C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\temperatures.csv')

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(features)

# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)

# Display the first 5 rows of the last 12 columns.
print(features.head(5))

# Use numpy to convert to arrays
import numpy as np

# Labels are the values we want to predict
labels = np.array(features['actual'])

# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('actual', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels =\
    train_test_split(features, labels, test_size = 0.25, random_state = 42)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')

# Print out the mean square error.
mse = mean_squared_error(test_labels, predictions)
print('RMSE:', np.sqrt(mse))

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]





# Example 8: Building Forest with Important Features Only

import pandas as pd
from   sklearn.metrics import mean_squared_error

# Read in data and display first 5 rows
features = pd.read_csv('C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\temperatures.csv')

#print(features)
# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#print(features)

# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)

# Display the first 5 rows of the last 12 columns.
#print(features.head(5))

# Use numpy to convert to arrays
import numpy as np

# Labels are the values we want to predict
labels = np.array(features['actual'])
#print(labels)

# Remove the labels from the features
# axis 1 refers to the columns
features = features.drop('actual', axis = 1)
print(features)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels =\
    train_test_split(features, labels, test_size = 0.25, random_state = 42)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')

# Print out the mean square error.
mse = mean_squared_error(test_labels, predictions)
print('RMSE:', np.sqrt(mse))

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)

# Extract the two most important features
important_indices = [feature_list.index('temp_1'), feature_list.index('average')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]

# Train the random forest
rf_most_important.fit(train_important, train_labels)

# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)

# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / test_labels))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')













# Exercise 5
#Import scikit-learn dataset library
from sklearn import datasets

#Load dataset
iris = datasets.load_iris()

# Creating a DataFrame of given iris dataset.
import pandas as pd
data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})
# what is this???????
iris['target_names']
print(iris)
print(data.head())

# Import train_test_split function
from sklearn.model_selection import train_test_split
X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y=data['species']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
rf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=rf.predict(X_test)
rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Predict species for a single flower.
# sepal length = 3, sepal width = 5
# petal length = 4, petal width = 2
prediction = rf.predict([[3, 5, 4, 2]])
# 'setosa', 'versicolor', 'virginica'
print(prediction)

feature_list = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]







# Exercise 6
from sklearn import datasets

#Load dataset
iris = datasets.load_iris()

# Creating a DataFrame of given iris dataset.
import pandas as pd
data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})
# what is this???????
iris['target_names']
print(iris)
print(data.head())

# Import train_test_split function
from sklearn.model_selection import train_test_split
X=data[['petal length']]  # Features
y=data['species']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
rf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=rf.predict(X_test)
rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Predict species for a single flower.
# sepal length = 3, sepal width = 5
# petal length = 4, petal width = 2
prediction = rf.predict([[2]])
# 'setosa', 'versicolor', 'virginica'
print(prediction)





# Random Forest Hyperparameter Tuning

# Example 9: Searching for the Optimal RandomForest Hyperparmaters
# Pandas is used for data manipulation
import pandas as pd
from   sklearn.metrics import mean_squared_error

# Read in data and display first 5 rows
features = pd.read_csv('C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\temperatures.csv')

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(features)

# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)

# Display the first 5 rows of the last 12 columns.
print(features.head(5))

# Use numpy to convert to arrays
import numpy as np

# Labels are the values we want to predict
labels = np.array(features['actual'])
print(labels)
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('actual', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels =train_test_split(features, labels, test_size = 0.25, random_state = 42)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')

# Print out the mean square error.
mse = mean_squared_error(test_labels, predictions)
print('RMSE:', np.sqrt(mse))

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

{'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_features, train_labels)

print("Best parrameters")
print(rf_random.best_params_)








# Exercise 7
import pandas as pd
from   sklearn.metrics import mean_squared_error

# Read in data and display first 5 rows
features = pd.read_csv('C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\temperatures.csv')

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(features)

# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)

# Display the first 5 rows of the last 12 columns.
print(features.head(5))

# Use numpy to convert to arrays
import numpy as np

# Labels are the values we want to predict
labels = np.array(features['actual'])

# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('actual', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels =\
    train_test_split(features, labels, test_size = 0.25, random_state = 42)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1600, random_state = 42, min_samples_split = 10, min_samples_leaf = 2, max_features = 'auto', max_depth = None, bootstrap = True )


#{'n_estimators': 1600, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': None, 'bootstrap': True}

# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')

# Print out the mean square error.
mse = mean_squared_error(test_labels, predictions)
print('RMSE:', np.sqrt(mse))









