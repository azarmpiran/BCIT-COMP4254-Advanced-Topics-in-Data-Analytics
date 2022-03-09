
import numpy   as np
import pandas as pd


PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE = 'loan_default.csv'
df = pd.read_csv(PATH + FILE)
print(df)



# Separate into x and y values.
# First we assign all the variables except target into the X to see which on is statistically significant through the chi square test
predictorVariables = ['age', 'ed','employ', 'income','debtinc','creddebt','othdebt']
X = df[predictorVariables]
y = df['default']

print(X)
print(y)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# Finding Significant Predictor Variables with chi-square - We know that greatar than 3.8 is good
test = SelectKBest(score_func=chi2, k=7)
chiScores = test.fit(X, y)
np.set_printoptions(precision=7)

# Printing the result to see which variable is statistically significant
print("\nPredictor variables: " + str(predictorVariables))
print("Predictor Chi-Square Scores: " + str(chiScores.scores_))

# Based on result all are greater than 3.8 so we dont need to re assign the X
# Now we need to re assign the X




from   sklearn.model_selection import train_test_split
from   sklearn.linear_model    import LogisticRegression

# Spilit the data and creating the model
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.30,random_state=0)

logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear',random_state=0)

logisticModel.fit(X_train,y_train)
y_pred=logisticModel.predict(X_test)
print(y_pred)

# Creating the matrics
from sklearn import metrics
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print("\nConfusion Matrix")
print(cm)

print('\nAccuracy: ',metrics.accuracy_score(y_test, y_pred))


# I dont know why but it does not create one column of the confusion matrix
# To calculate the F1 and recall and precision we need to have these variables:
TN = cm[0][0]
FN = cm[0][1]
FP = cm[1][0]
TP = cm[1][1]

print("")
print("True Negative:  " + str(TN))
print("False Negative: " + str(FN))
print("False Positive: " + str(FP))
print("True Positive:  " + str(TP))

precision = (TP/(FP + TP))
print("\nPrecision:  " + str(round(precision, 3)))

recall = (TP/(TP + FN))
print("Recall:     " + str(round(recall,3)))

F1 = 2*((precision*recall)/(precision+recall))
print("F1:         " + str(round(F1,3)))