# Assignment 1
# Azarm Piran | A01195657

import pandas as pd
from   sklearn               import metrics
from sklearn.linear_model import LogisticRegression
import numpy  as np


################################################################################## Step 1
# Import data into a DataFrame.
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE = "insurance_claims.csv"
df   = pd.read_table(PATH + FILE, sep=',')

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

X = df.copy()       # Create separate copy to prevent unwanted tampering of data.
del X['CLAIM_FLAG']     # Delete target variable.


################################################################################## Step 2
# Cleaning data

# KIDSDRIV column

# BIRTH column
from datetime import datetime
X["BIRTH"] = pd.to_datetime(X['BIRTH'])

# AGE column

# HOMEKIDS column

# YOJ column

# INCOME column
X['INCOME'] = X['INCOME'].str.replace(r'$', '', regex=False)
X['INCOME'] = X['INCOME'].str.replace(r',', '', regex=False)
X["INCOME"] = pd.to_numeric(X["INCOME"], downcast="float")

# PARENT1 column
X = pd.get_dummies(X, columns=['PARENT1'])

# HOME_VAL column
X['HOME_VAL'] = X['HOME_VAL'].str.replace(r'$', '', regex=False)
X['HOME_VAL'] = X['HOME_VAL'].str.replace(r',', '', regex=False)
X["HOME_VAL"] = pd.to_numeric(X["HOME_VAL"], downcast="float")

# MSTATUS column
X['MSTATUS'] = X['MSTATUS'].str.replace(r'z_No', 'No', regex=False)
X = pd.get_dummies(X, columns=['MSTATUS'])

# GENDER column
X['GENDER'] = X['GENDER'].str.replace(r'z_F', 'F', regex=False)
X = pd.get_dummies(X, columns=['GENDER'])

# EDUCATION column
X['EDUCATION'] = X['EDUCATION'].str.replace(r'<High School', '1', regex=False)
X['EDUCATION'] = X['EDUCATION'].str.replace(r'Bachelors', '2', regex=False)
X['EDUCATION'] = X['EDUCATION'].str.replace(r'PhD', '3', regex=False)
X['EDUCATION'] = X['EDUCATION'].str.replace(r'Masters', '4', regex=False)
X['EDUCATION'] = X['EDUCATION'].str.replace(r'z_High School', '5', regex=False)
X["EDUCATION"] = pd.to_numeric(X["EDUCATION"], downcast="float")

# OCCUPATION column
X['OCCUPATION'] = X['OCCUPATION'].str.replace(r'Clerical', '1', regex=False)
X['OCCUPATION'] = X['OCCUPATION'].str.replace(r'Doctor', '2', regex=False)
X['OCCUPATION'] = X['OCCUPATION'].str.replace(r'Home Maker', '3', regex=False)
X['OCCUPATION'] = X['OCCUPATION'].str.replace(r'Lawyer', '4', regex=False)
X['OCCUPATION'] = X['OCCUPATION'].str.replace(r'Manager', '5', regex=False)
X['OCCUPATION'] = X['OCCUPATION'].str.replace(r'Professional', '6', regex=False)
X['OCCUPATION'] = X['OCCUPATION'].str.replace(r'Student', '7', regex=False)
X['OCCUPATION'] = X['OCCUPATION'].str.replace(r'Student', '8', regex=False)
X['OCCUPATION'] = X['OCCUPATION'].str.replace(r'z_Blue Collar', '9', regex=False)
X['OCCUPATION'] = pd.to_numeric(X["OCCUPATION"], downcast="float")

# TRAVTIME column

# CAR_USE column
X = pd.get_dummies(X, columns=['CAR_USE'])

# BLUEBOOK column
X['BLUEBOOK'] = X['BLUEBOOK'].str.replace(r'$', '', regex=False)
X['BLUEBOOK'] = X['BLUEBOOK'].str.replace(r',', '', regex=False)
X['BLUEBOOK'] = pd.to_numeric(X["BLUEBOOK"], downcast="float")

# TIF column

# CAR_TYPE column
X['CAR_TYPE'] = X['CAR_TYPE'].str.replace(r'Minivan', '1', regex=False)
X['CAR_TYPE'] = X['CAR_TYPE'].str.replace(r'Panel Truck', '2', regex=False)
X['CAR_TYPE'] = X['CAR_TYPE'].str.replace(r'Pickup', '3', regex=False)
X['CAR_TYPE'] = X['CAR_TYPE'].str.replace(r'Sports Car', '4', regex=False)
X['CAR_TYPE'] = X['CAR_TYPE'].str.replace(r'Van', '5', regex=False)
X['CAR_TYPE'] = X['CAR_TYPE'].str.replace(r'z_SUV', '6', regex=False)
X['CAR_TYPE'] = pd.to_numeric(X['CAR_TYPE'], downcast="float")

# RED_CAR column
X = pd.get_dummies(X, columns=['RED_CAR'])

# REVOKED column
X = pd.get_dummies(X, columns=['REVOKED'])

# MVR_PTS column

# CAR_AGE column

# URBANICITY column
X['URBANICITY'] = X['URBANICITY'].str.replace(r'Highly Urban/ Urban', '1', regex=False)
X['URBANICITY'] = X['URBANICITY'].str.replace(r'z_Highly Rural/ Rural', '2', regex=False)
X['URBANICITY'] = pd.to_numeric(X['URBANICITY'], downcast="float")


################################################################################## Step 3
# imputing data

# Impute Function
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

# Using describe(), we can find columns with null values
#print(X.describe())

# Calling impute function
X = imputeNullValues('AGE', X)
X = imputeNullValues('YOJ', X)
X = imputeNullValues('INCOME', X)
X = imputeNullValues('HOME_VAL', X)
X = imputeNullValues('OCCUPATION', X)
X = imputeNullValues('CAR_AGE', X)




################################################################################## Step 4
# Chi square test

# Re assign X
X = X[['KIDSDRIV', 'HOMEKIDS', 'TRAVTIME','TIF','MVR_PTS','imp_INCOME','m_INCOME','PARENT1_No','PARENT1_Yes'
       ,'m_HOME_VAL','imp_HOME_VAL','MSTATUS_No','MSTATUS_Yes','GENDER_F','GENDER_M','EDUCATION','m_OCCUPATION'
       ,'imp_OCCUPATION','CAR_USE_Commercial','CAR_USE_Private','BLUEBOOK','CAR_TYPE','RED_CAR_no','RED_CAR_yes'
       ,'REVOKED_No','REVOKED_Yes','URBANICITY','m_AGE','imp_AGE','m_YOJ','imp_YOJ','m_CAR_AGE','imp_CAR_AGE']]
y = df[['CLAIM_FLAG']]

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

test = SelectKBest(score_func=chi2, k=32)
chiScores = test.fit(X, y) # Summarize scores
np.set_printoptions(precision=32)

# # Generally, >=3.8 is good)
print('---------------------------------------------------------------------------')
print("Predictor variables: ")
print(list(X.keys()))
print('---------------------------------------------------------------------------')
print("Predictor Chi-Square Scores: \n " + str(chiScores.scores_))
print('---------------------------------------------------------------------------')



################################################################################## Step 5
# Removing variables that is smaller than 3.8
# KIDSDRIV              = 1.1077587786895791e+02 = 110.775877868957 --> We have this
# HOMEKIDS              = 1.6148652595101191e+02 = 161.486525951011 --> We have this
# TRAVTIME              = 9.8393064883399404e+01 = 98.3930648833994 --> We have this
# TIF                   = 1.1315235963013083e+02 = 113.15235963013  --> We have this
# MVR_PTS               = 7.6712526569146723e+02 = 767.125265691467  --> We have this
# imp_INCOME            = 4.0377962866252549e+06 = 4037796.28662525 --> Removed
# m_INCOME              = 8.0860064947290704e-05 = 0.00008086006494 --> Removed -- smaller than 3.8
# PARENT1_No            = 1.9230820390286020e+01 = 19.230820390286 --> Added
# PARENT1_Yes           = 1.2407337932380236e+02 = 124.073379323802 --> Added
# m_HOME_VAL            = 1.5027350882784798e-01 = 0.150273508827847 --> Removed -- smaller than 3.8
# imp_HOME_VAL          = 1.7176906443834662e+07 = 17176906.4438346 --> Removed
# MSTATUS_No            = 4.7552901053926966e+01 = 47.5529010539269
# MSTATUS_Yes           = 3.2355104967581070e+01 = 32.355104967581
# GENDER_F              = 4.7380051914551430e-01 = 0.473800519145514 --> Removed -- smaller than 3.8
# GENDER_M              = 5.5851453708276599e-01 = 0.558514537082765 --> Removed -- smaller than 3.8
# EDUCATION             = 4.6225553800361681e+00 = 4.62255538003616
# m_OCCUPATION          = 1.0041264424711143e+00 = 1.00412644247111 --> Removed -- smaller than 3.8
# imp_OCCUPATION        = 3.9796490398495159e+01 = 39.7964903984951 --> Added
# CAR_USE_Commercial    = 6.2786208103193637e+01 = 62.7862081031936 --> Added
# CAR_USE_Private       = 3.7636822090271380e+01 = 37.6368220902713 --> Added
# BLUEBOOK              = 2.1526391453765595e+05 = 215263.914537655 --> Removed - Just a little bit improvement
# CAR_TYPE              = 3.0611238682249489e+01 = 30.6112386822494 --> Added
# RED_CAR_no            = 1.5498701626366735e-03 = 0.00154987016263667 --> Removed -- smaller than 3.8
# RED_CAR_yes           = 3.9291588969499920e-03 = 0.00392915889694999 --> Removed -- smaller than 3.8
# REVOKED_No            = 1.4180351944477657e+01 = 14.1803519444776
# REVOKED_Yes           = 1.0389045628775673e+02 = 103.890456287756
# URBANICITY            = 3.7157450824343414e+01 = 37.1574508243434 --> Added
# m_AGE                 = 9.7969526470053729e+00 = 9.79695264700537 --> Does not do anything - Removed
# imp_AGE               = 9.5756798888641242e+01 = 95.7567988886412 --> Decreased a little bit -- Removed
# m_YOJ                 = 5.7395893520919464e-02 = 0.0573958935209194 --> Removed -- smaller than 3.8
# imp_YOJ               = 4.0178170648817975e+01 = 40.1781706488179 --> Decreased a little bit -- Removed
# m_CAR_AGE             = 2.6708622410034161e-02 = 0.0267086224100341 --> Removed -- smaller than 3.8
# imp_CAR_AGE           = 2.7298913654364424e+02 = 272.989136543644 --> Added





################################################################################## Step 6
# Implementing Logistic Regression

from   sklearn.model_selection import train_test_split
from   sklearn.linear_model    import LogisticRegression

# Re-assign X with significant columns only after chi-square test.
X = X[['KIDSDRIV', 'HOMEKIDS', 'TRAVTIME','TIF','MVR_PTS','imp_OCCUPATION','CAR_TYPE','URBANICITY','CAR_USE_Commercial','CAR_USE_Private','imp_CAR_AGE']]

y = df['CLAIM_FLAG']

# Split data.
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25,random_state=0)

# Perform logistic regression.
logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear',random_state=0)

# Fit the model.
logisticModel.fit(X_train,y_train)
y_pred=logisticModel.predict(X_test)
print(y_pred)

# Show accuracy scores.
print('---------------------------------------------------------------------------')
print('Results without scaling:')

# Show confusion matrix
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print("\nConfusion Matrix")
print(cm)

TN = cm[0][0] # True Negative  (Col 0, Row 0)
FN = cm[0][1] # False Negative (Col 0, Row 1)
FP = cm[1][0] # False Positive (Col 1, Row 0)
TP = cm[1][1] # True Positive  (Col 1, Row 1)


precision = (TP/(FP + TP))
print("\nPrecision:  " + str(round(precision, 3)))

recall = (TP/(TP + FN))
print("Recall:     " + str(round(recall,3)))

F1 = 2*((precision*recall)/(precision+recall))
print("F1:         " + str(round(F1,3)))

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))




print('After chi aquare test we removed all variables smaller than 3.8 because they were not statistically significant.')
print('After removing insignificant variables, I ended up having too many variables which makes our model more complicated.')
print('I removed variables step by step and checked all the important measure such as accuracy,recall,F1 and precision after removing each variable to make sure it does not affect my model in a bad way. Just a tiny change is okay if it makes my model more simple.')



################################################################################## Step 8
# Scaling

# MinMaxScaler
print('---------------------------------------------------------------------------')
print('Results with MinMaxScaler:')

from sklearn.preprocessing import MinMaxScaler
sc_x    = MinMaxScaler()
X_Scale = sc_x.fit_transform(X)

# Split data.
X_train, X_test, y_train, y_test = train_test_split(X_Scale, y, test_size=0.25, random_state=0)

# Perform logistic regression.
logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear',random_state=0)

# Fit the model.
logisticModel.fit(X_train,y_train)
y_pred=logisticModel.predict(X_test)


# Show confusion matrix
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print("\nConfusion Matrix")
print(cm)

TN = cm[0][0] # True Negative  (Col 0, Row 0)
FN = cm[0][1] # False Negative (Col 0, Row 1)
FP = cm[1][0] # False Positive (Col 1, Row 0)
TP = cm[1][1] # True Positive  (Col 1, Row 1)


precision = (TP/(FP + TP))
print("\nPrecision:  " + str(round(precision, 3)))

recall = (TP/(TP + FN))
print("Recall:     " + str(round(recall,3)))

F1 = 2*((precision*recall)/(precision+recall))
print("F1:         " + str(round(F1,3)))

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))



# StandardScaler
print('---------------------------------------------------------------------------')
print('Results with StandardScaler:')

from sklearn.preprocessing import StandardScaler
sc_x    = StandardScaler()
X_Scale = sc_x.fit_transform(X)

# Split data.
X_train, X_test, y_train, y_test = train_test_split(X_Scale, y, test_size=0.25, random_state=0)

# Perform logistic regression.
logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear',random_state=0)

# Fit the model.
logisticModel.fit(X_train,y_train)
y_pred=logisticModel.predict(X_test)

# Show confusion matrix
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print("\nConfusion Matrix")
print(cm)

TN = cm[0][0] # True Negative  (Col 0, Row 0)
FN = cm[0][1] # False Negative (Col 0, Row 1)
FP = cm[1][0] # False Positive (Col 1, Row 0)
TP = cm[1][1] # True Positive  (Col 1, Row 1)


precision = (TP/(FP + TP))
print("\nPrecision:  " + str(round(precision, 3)))

recall = (TP/(TP + FN))
print("Recall:     " + str(round(recall,3)))

F1 = 2*((precision*recall)/(precision+recall))
print("F1:         " + str(round(F1,3)))

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))




# RobustScaler
print('---------------------------------------------------------------------------')
print('Results with RobustScaler:')

from sklearn.preprocessing import RobustScaler
sc_x    = RobustScaler()
X_Scale = sc_x.fit_transform(X)

# Split data.
X_train, X_test, y_train, y_test = train_test_split(X_Scale, y, test_size=0.25, random_state=0)

# Perform logistic regression.
logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear',random_state=0)

# Fit the model.
logisticModel.fit(X_train,y_train)
y_pred=logisticModel.predict(X_test)

# Show confusion matrix
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print("\nConfusion Matrix")
print(cm)

TN = cm[0][0] # True Negative  (Col 0, Row 0)
FN = cm[0][1] # False Negative (Col 0, Row 1)
FP = cm[1][0] # False Positive (Col 1, Row 0)
TP = cm[1][1] # True Positive  (Col 1, Row 1)


precision = (TP/(FP + TP))
print("\nPrecision:  " + str(round(precision, 3)))

recall = (TP/(TP + FN))
print("Recall:     " + str(round(recall,3)))

F1 = 2*((precision*recall)/(precision+recall))
print("F1:         " + str(round(F1,3)))

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))


print('---------------------------------------------------------------------------')
print('The results and accuracy is better with StandardScaler so I will use this scaling')



################################################################################## Step 9
# Normalize estimated salary with StandardScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

# Fit and transform the data.
dfX = X[['KIDSDRIV', 'HOMEKIDS', 'TRAVTIME','TIF','MVR_PTS','imp_OCCUPATION','CAR_TYPE','URBANICITY','CAR_USE_Commercial','CAR_USE_Private','imp_CAR_AGE']]

dfXScaled = sc_x.fit_transform(dfX)

import numpy as np

# enumerate splits - returns train and test arrays of indexes.
# scikit-learn k-fold cross-validation
from sklearn.model_selection import KFold

# data sample

# prepare cross validation with eight folds.
kfold         = KFold(8, True)
accuracyList  = []
precisionList = []
f1List        = []
count         = 0

for train_index, test_index in kfold.split(X):
    X_train = X.loc[X.index.isin(train_index)]
    X_test  = X.loc[X.index.isin(test_index)]
    y_train = y.loc[y.index.isin(train_index)]
    y_test  = y.loc[y.index.isin(test_index)]

    # Perform logistic regression.
    logisticModel = LogisticRegression(fit_intercept=True,
                                       solver='liblinear')
    # Fit the model.
    logisticModel.fit(X_train, y_train)

    y_pred = logisticModel.predict(X_test)

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



################################################################################## Step 11
# Drawing a ROC Curve

# Create one-vs-rest logistic regression object
clf = LogisticRegression(
    random_state=0,
    multi_class='multinomial', solver='newton-cg')


# Train model
model  = clf.fit(X_train, y_train)

# Predict class
y_pred = model.predict(X_test)
print(y_pred)

# View predicted probabilities
y_prob = model.predict_proba(X_test)
print(y_prob)



import matplotlib.pyplot       as plt
from sklearn.metrics           import roc_curve
from sklearn.metrics           import roc_auc_score

auc = roc_auc_score(y_test, y_prob[:, 1],)
print('Logistic: ROC AUC=%.3f' % (auc))

# calculate roc curves
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
plt.plot(fpr, tpr, marker='.', label='Logistic')
plt.plot([0,1], [0,1], '--', label='')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()





