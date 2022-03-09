
# Lab Week 8
# Azarm Piran | A01195657


# Example 1: Data Preparation

from pydataset import data
import pandas as pd
import numpy as np

# Get the housing data
df = data('Housing')
# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

print(df)

for i, j in enumerate(np.unique(pd.qcut(df['price'], 3))):
    print("i: " + str(i) + "  j: " + str(j))

df['price'] = pd.qcut(df['price'], 3, labels=['0', '1', '2']).cat.codes

def convertToBinaryValues(df, columns):
    for i in range(0, len(columns)):
        df[columns[i]] = df[columns[i]].map({'yes': 1, 'no': 0})
    return df

df = convertToBinaryValues(df, ['driveway', 'recroom', 'fullbase', 'gashw', 'airco', 'prefarea'])

print(df)

# Split into two sets
y = df['price']
X = df.drop('price', 1)
print(y)



# Example 2: Bagging

from pydataset import data
import pandas as pd
import numpy as np

# Get the housing data
df = data('Housing')
# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

print(df)

for i, j in enumerate(np.unique(pd.qcut(df['price'], 3))):
    print("i: " + str(i) + "  j: " + str(j))

df['price'] = pd.qcut(df['price'], 3, labels=['0', '1', '2']).cat.codes

def convertToBinaryValues(df, columns):
    for i in range(0, len(columns)):
        df[columns[i]] = df[columns[i]].map({'yes': 1, 'no': 0})
    return df

df = convertToBinaryValues(df, ['driveway', 'recroom', 'fullbase', 'gashw', 'airco', 'prefarea'])

print(df)

# Split into two sets
y = df['price']
X = df.drop('price', 1)
print(y)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble        import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.linear_model    import RidgeClassifier
from sklearn.svm             import SVC

seed = 1075
np.random.seed(seed)

# Create classifiers
rf          = RandomForestClassifier()
et          = ExtraTreesClassifier()
knn         = KNeighborsClassifier()
svc         = SVC()
rg          = RidgeClassifier()

# Build array of classifiers.
classifierArray   = [rf, et, knn, svc, rg]

def showStats(classifier, scores):
    print(classifier + ":    ", end="")
    strMean = str(round(scores.mean(),2))

    strStd  = str(round(scores.std(),2))
    print("Mean: "  + strMean + "   ", end="")
    print("Std: " + strStd)

# Search for the best classifier.
for clf in classifierArray:
    # cross_val_score esitmates the model accuracy given the inputs.
    # cv indicates how many folds.
    individualModel_scores = cross_val_score(clf, X, y, cv=10)

    # max_features means the maximum number of features to draw from X.
    # max_samples
    bagging_clf            = BaggingClassifier(
        clf, max_samples=0.4, max_features=11)
    bagging_scores = cross_val_score(bagging_clf, X, y, cv=10)

    showStats(clf.__class__.__name__, individualModel_scores)
    showStats("Bagged " + clf.__class__.__name__, bagging_scores)
    print("")



# Exercise 1

from pydataset import data
import pandas as pd
import numpy as np

# Get the housing data
df = data('Housing')
# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

print(df)

for i, j in enumerate(np.unique(pd.qcut(df['price'], 3))):
    print("i: " + str(i) + "  j: " + str(j))

df['price'] = pd.qcut(df['price'], 3, labels=['0', '1', '2']).cat.codes

def convertToBinaryValues(df, columns):
    for i in range(0, len(columns)):
        df[columns[i]] = df[columns[i]].map({'yes': 1, 'no': 0})
    return df

df = convertToBinaryValues(df, ['driveway', 'recroom', 'fullbase', 'gashw', 'airco', 'prefarea'])

print(df)

# Split into two sets
y = df['price']
X = df.drop('price', 1)
print(y)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble        import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.linear_model    import RidgeClassifier
from sklearn.svm             import SVC

seed = 1075
np.random.seed(seed)

# Create classifiers
rf          = RandomForestClassifier()
et          = ExtraTreesClassifier()
knn         = KNeighborsClassifier()
svc         = SVC()
rg          = RidgeClassifier()

# Build array of classifiers.
classifierArray   = [rf, et, knn, svc, rg]

def showStats(classifier, scores):
    print(classifier + ":    ", end="")
    strMean = str(round(scores.mean(),2))

    strStd  = str(round(scores.std(),2))
    print("Mean: "  + strMean + "   ", end="")
    print("Std: " + strStd)

# Search for the best classifier.
for clf in classifierArray:
    # cross_val_score esitmates the model accuracy given the inputs.
    # cv indicates how many folds.
    individualModel_scores = cross_val_score(clf, X, y, cv=10)

    # max_features means the maximum number of features to draw from X.
    # max_samples
    bagging_clf            = BaggingClassifier(
        clf, max_samples=0.4, max_features=7)
    bagging_scores = cross_val_score(bagging_clf, X, y, cv=10)

    showStats(clf.__class__.__name__, individualModel_scores)
    showStats("Bagged " + clf.__class__.__name__, bagging_scores)
    print("")







# Exercise 2

from pydataset import data
import pandas as pd
import numpy as np


# Get the housing data
df = data('Housing')
# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

print(df)

for i, j in enumerate(np.unique(pd.qcut(df['price'], 3))):
    print("i: " + str(i) + "  j: " + str(j))

df['price'] = pd.qcut(df['price'], 3, labels=['0', '1', '2']).cat.codes

def convertToBinaryValues(df, columns):
    for i in range(0, len(columns)):
        df[columns[i]] = df[columns[i]].map({'yes': 1, 'no': 0})
    return df

df = convertToBinaryValues(df, ['driveway', 'recroom', 'fullbase', 'gashw', 'airco', 'prefarea'])

print(df)

# Split into two sets
y = df['price']
X = df.drop('price', 1)
print(y)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble        import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.linear_model    import RidgeClassifier
from sklearn.svm             import SVC
from   sklearn.linear_model    import LogisticRegression

print(df)

seed = 1075
np.random.seed(seed)

# Create classifiers
rf          = RandomForestClassifier()
et          = ExtraTreesClassifier()
knn         = KNeighborsClassifier()
svc         = SVC()
rg          = RidgeClassifier()
lr = LogisticRegression(fit_intercept=True, solver='liblinear')

# Build array of classifiers.
classifierArray   = [rf, et, knn, svc, rg, lr]

def showStats(classifier, scores):
    print(classifier + ":    ", end="")
    strMean = str(round(scores.mean(),2))

    strStd  = str(round(scores.std(),2))
    print("Mean: "  + strMean + "   ", end="")
    print("Std: " + strStd)

# Search for the best classifier.
for clf in classifierArray:
    # cross_val_score esitmates the model accuracy given the inputs.
    # cv indicates how many folds.
    individualModel_scores = cross_val_score(clf, X, y, cv=10)

    # max_features means the maximum number of features to draw from X.
    # max_samples
    bagging_clf            = BaggingClassifier(
        clf, max_samples=0.4, max_features=11)
    bagging_scores = cross_val_score(bagging_clf, X, y, cv=10)

    showStats(clf.__class__.__name__, individualModel_scores)
    showStats("Bagged " + clf.__class__.__name__, bagging_scores)
    print("")





# Exercise 3

from pydataset import data
import pandas as pd
import numpy as np
from sklearn.preprocessing   import LabelEncoder

PATH       = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
df         = pd.read_csv(PATH + 'loan_default.csv')
print(df.head())

# Prepare the data.
X = df.copy()
del X['default']
y = df['default']

print(df)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble        import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.linear_model    import RidgeClassifier
from sklearn.svm             import SVC
from   sklearn.linear_model    import LogisticRegression

seed = 1075
np.random.seed(seed)

# Create classifiers
rf          = RandomForestClassifier()
et          = ExtraTreesClassifier()
knn         = KNeighborsClassifier()
svc         = SVC()
rg          = RidgeClassifier()
lr = LogisticRegression(fit_intercept=True, solver='liblinear')

# Build array of classifiers.
classifierArray   = [rf, et, knn, svc, rg, lr]

def showStats(classifier, scores):
    print(classifier + ":    ", end="")
    strMean = str(round(scores.mean(),2))

    strStd  = str(round(scores.std(),2))
    print("Mean: "  + strMean + "   ", end="")
    print("Std: " + strStd)

# Search for the best classifier.
for clf in classifierArray:
    # cross_val_score esitmates the model accuracy given the inputs.
    # cv indicates how many folds.
    individualModel_scores = cross_val_score(clf, X, y, cv=10)

    # max_features means the maximum number of features to draw from X.
    # max_samples
    bagging_clf            = BaggingClassifier(
        clf, max_samples=0.4, max_features=7)
    bagging_scores = cross_val_score(bagging_clf, X, y, cv=10)

    showStats(clf.__class__.__name__, individualModel_scores)
    showStats("Bagged " + clf.__class__.__name__, bagging_scores)
    print("")



# Example 3: Evaluating Precision, Recall and F1 Scores

from pydataset import data
import pandas as pd
import numpy as np

# Get the housing data
df = data('Housing')
# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

print(df)

for i, j in enumerate(np.unique(pd.qcut(df['price'], 3))):
    print("i: " + str(i) + "  j: " + str(j))

df['price'] = pd.qcut(df['price'], 3, labels=['0', '1', '2']).cat.codes

def convertToBinaryValues(df, columns):
    for i in range(0, len(columns)):
        df[columns[i]] = df[columns[i]].map({'yes': 1, 'no': 0})
    return df

df = convertToBinaryValues(df, ['driveway', 'recroom', 'fullbase', 'gashw', 'airco', 'prefarea'])

print(df)

# Split into two sets
y = df['price']
X = df.drop('price', 1)
print(y)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble        import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.linear_model    import RidgeClassifier
from sklearn.svm             import SVC

seed = 1075
np.random.seed(seed)

# Create classifiers
rf          = RandomForestClassifier()
et          = ExtraTreesClassifier()
knn         = KNeighborsClassifier()
svc         = SVC()
rg          = RidgeClassifier()

# Build array of classifiers.
classifierArray   = [rf, et, knn, svc, rg]

def showStats(classifier, scores):
    print(classifier + ":    ", end="")
    strMean = str(round(scores.mean(),2))

    strStd  = str(round(scores.std(),2))
    print("Mean: "  + strMean + "   ", end="")
    print("Std: " + strStd)

from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

def evaluateModel(model, X_test, y_test, title):
    print("\n*** " + title + " ***")
    predictions = model.predict(X_test)
    accuracy    = metrics.accuracy_score(y_test, predictions)
    recall      = metrics.recall_score(y_test, predictions, average='weighted')
    precision   = metrics.precision_score(y_test, predictions, average='weighted')
    f1          = metrics.f1_score(y_test, predictions, average='weighted')

    print("Accuracy:  " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall:    " + str(recall))
    print("F1:        " + str(f1))

# Search for the best classifier.
for clf in classifierArray:
    modelType = clf.__class__.__name__

    # max_features means the maximum number of features to draw from X.
    # max_samples sets the percentage of available data used for fitting.
    bagging_clf = BaggingClassifier(clf, max_samples=0.4, max_features=11)
    baggedModel = bagging_clf.fit(X_train, y_train)
    evaluateModel(baggedModel, X_test, y_test, "Bagged: " + modelType)








# Exercise 4

from pydataset import data
import pandas as pd
import numpy as np

# Get the housing data
df = data('Housing')
# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

print(df)

for i, j in enumerate(np.unique(pd.qcut(df['price'], 3))):
    print("i: " + str(i) + "  j: " + str(j))

df['price'] = pd.qcut(df['price'], 3, labels=['0', '1', '2']).cat.codes

def convertToBinaryValues(df, columns):
    for i in range(0, len(columns)):
        df[columns[i]] = df[columns[i]].map({'yes': 1, 'no': 0})
    return df

df = convertToBinaryValues(df, ['driveway', 'recroom', 'fullbase', 'gashw', 'airco', 'prefarea'])

print(df)

# Split into two sets
y = df['price']
X = df.drop('price', 1)
print(y)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble        import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.linear_model    import RidgeClassifier
from sklearn.svm             import SVC

seed = 1075
np.random.seed(seed)

# Create classifiers
rf          = RandomForestClassifier()
et          = ExtraTreesClassifier()
knn         = KNeighborsClassifier()
svc         = SVC()
rg          = RidgeClassifier()

# Build array of classifiers.
classifierArray   = [rf, et, knn, svc, rg]

def showStats(classifier, scores):
    print(classifier + ":    ", end="")
    strMean = str(round(scores.mean(),2))

    strStd  = str(round(scores.std(),2))
    print("Mean: "  + strMean + "   ", end="")
    print("Std: " + strStd)

from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

def evaluateModel(model, X_test, y_test, title):
    print("\n*** " + title + " ***")
    predictions = model.predict(X_test)
    accuracy    = metrics.accuracy_score(y_test, predictions)
    recall      = metrics.recall_score(y_test, predictions, average='weighted')
    precision   = metrics.precision_score(y_test, predictions, average='weighted')
    f1          = metrics.f1_score(y_test, predictions, average='weighted')

    print("Accuracy:  " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall:    " + str(recall))
    print("F1:        " + str(f1))

# Search for the best classifier.
for clf in classifierArray:
    modelType = clf.__class__.__name__

    # max_features means the maximum number of features to draw from X.
    # max_samples sets the percentage of available data used for fitting.
    bagging_clf = BaggingClassifier(clf, max_samples=0.4, max_features=11)
    baggedModel = bagging_clf.fit(X_train, y_train)
    evaluateModel(baggedModel, X_test, y_test, "Bagged: " + modelType)

    individualModel_scores = cross_val_score(clf, X, y, cv=10)
    bagging_clf = BaggingClassifier(clf, max_samples=0.4, max_features=11)
    bagging_scores = cross_val_score(bagging_clf, X, y, cv=10)

    showStats(clf.__class__.__name__, individualModel_scores)
    showStats("Bagged " + clf.__class__.__name__, bagging_scores)
    print("")





# Exercise 5


from pydataset import data
import pandas as pd
import numpy as np
from sklearn.preprocessing   import LabelEncoder

PATH       = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
df         = pd.read_csv(PATH + 'loan_default.csv')
print(df.head())

# Prepare the data.
X = df.copy()
del X['default']
y = df['default']

print(df)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble        import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.linear_model    import RidgeClassifier
from sklearn.svm             import SVC
from   sklearn.linear_model    import LogisticRegression

import warnings
warnings.filterwarnings('ignore')
seed = 1075
np.random.seed(seed)

# Create classifiers
rf          = RandomForestClassifier()
et          = ExtraTreesClassifier()
knn         = KNeighborsClassifier()
svc         = SVC()
rg          = RidgeClassifier()
lr = LogisticRegression(fit_intercept=True, solver='liblinear')

# Build array of classifiers.
classifierArray   = [rf, et, knn, svc, rg, lr]

def showStats(classifier, scores):
    print(classifier + ":    ", end="")
    strMean = str(round(scores.mean(),2))

    strStd  = str(round(scores.std(),2))
    print("Mean: "  + strMean + "   ", end="")
    print("Std: " + strStd)

# Search for the best classifier.
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

def evaluateModel(model, X_test, y_test, title):
    print("\n*** " + title + " ***")
    predictions = model.predict(X_test)
    accuracy    = metrics.accuracy_score(y_test, predictions)
    recall      = metrics.recall_score(y_test, predictions, average='weighted')
    precision   = metrics.precision_score(y_test, predictions, average='weighted')
    f1          = metrics.f1_score(y_test, predictions, average='weighted')

    print("Accuracy:  " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall:    " + str(recall))
    print("F1:        " + str(f1))

# Search for the best classifier.
for clf in classifierArray:
    modelType = clf.__class__.__name__

    # max_features means the maximum number of features to draw from X.
    # max_samples sets the percentage of available data used for fitting.
    bagging_clf = BaggingClassifier(clf, max_samples=0.4, max_features=7)
    baggedModel = bagging_clf.fit(X_train, y_train)
    evaluateModel(baggedModel, X_test, y_test, "Bagged: " + modelType)














