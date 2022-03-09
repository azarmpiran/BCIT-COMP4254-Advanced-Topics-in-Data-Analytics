# Example 1: Simple Logistic Regression for GMAT vs. GPA
import pandas  as pd
import numpy   as np

# Setup data.
candidates = {'gmat': [780,750,690,710,680,730,690,720,
 740,690,610,690,710,680,770,610,580,650,540,590,620,
 600,550,550,570,670,660,580,650,660,640,620,660,660,
 680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,
 3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,
 3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,
 3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,
 1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,
 5,1,2,1,4,5],
              'admitted': [1,1,1,1,1,1,0,1,1,0,0,1,
 1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,
 0,0,1]}

df = pd.DataFrame(candidates,columns= ['gmat', 'gpa','work_experience','admitted'])

# Separate into x and y values.
predictorVariables = ['gmat', 'gpa','work_experience']
X = df[predictorVariables]
y = df['admitted']

# Re-assign X with significant columns only after chi-square test.
X = df[['gmat', 'work_experience']]

############## SECTION B start #################################################
# Split data.
from   sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25,random_state=0)
############## SECTION B end   #################################################

# Perform logistic regression.
from   sklearn.linear_model    import LogisticRegression

logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear',random_state=0)
logisticModel.fit(X_train,y_train)
y_pred=logisticModel.predict(X_test)
print(y_pred)

# Show precision, recall and F1 scores for all classes.
from   sklearn                 import metrics
accuracy  = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average=None)
recall    = metrics.recall_score(   y_test, y_pred, average=None)
f1        = metrics.f1_score(       y_test, y_pred, average=None)


print("Accuracy:  " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: "    + str(recall))
print("F1: "        + str(f1))





# Exercise 1
import pandas  as pd
import numpy   as np

# Setup data.
candidates = {'gmat': [780,750,690,710,680,730,690,720,
 740,690,610,690,710,680,770,610,580,650,540,590,620,
 600,550,550,570,670,660,580,650,660,640,620,660,660,
 680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,
 3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,
 3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,
 3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,
 1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,
 5,1,2,1,4,5],
              'admitted': [1,1,1,1,1,1,0,1,1,0,0,1,
 1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,
 0,0,1]}

df = pd.DataFrame(candidates,columns= ['gmat', 'gpa','work_experience','admitted'])

# Separate into x and y values.
predictorVariables = ['gmat', 'gpa','work_experience']
X = df[predictorVariables]
y = df['admitted']

# Re-assign X with significant columns only after chi-square test.
X = df[['gmat', 'work_experience']]

############## SECTION B start #################################################
from sklearn.preprocessing import MinMaxScaler
sc_x     = MinMaxScaler()
X_Scaled = sc_x.fit_transform(X)

# Split data.
from   sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_Scaled, y, test_size=0.25,random_state=0)
############## SECTION B end   #################################################


# Perform logistic regression.
from   sklearn.linear_model    import LogisticRegression

logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear',random_state=0)
logisticModel.fit(X_train,y_train)
y_pred=logisticModel.predict(X_test)
print(y_pred)

# Show precision, recall and F1 scores for all classes.
from   sklearn                 import metrics
accuracy  = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average=None)
recall    = metrics.recall_score(   y_test, y_pred, average=None)
f1        = metrics.f1_score(       y_test, y_pred, average=None)

print("Accuracy:  " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: "    + str(recall))
print("F1: "        + str(f1))





# Example 2: Linear Regression without Scaling
import pandas as pd
import numpy as np
from sklearn import datasets
from   sklearn.model_selection import train_test_split
import statsmodels.api         as sm
import numpy                   as np
from   sklearn                 import metrics

wine = datasets.load_wine()
dataset = pd.DataFrame(
    data=np.c_[wine['data'], wine['target']],
    columns=wine['feature_names'] + ['target']
)

# Create copy to prevent overwrite.
X = dataset.copy()
del X['target']         # Remove target variable
del X['hue']            # Remove unwanted features
del X['ash']
del X['magnesium']
del X['malic_acid']
del X['alcohol']

y = dataset['target']

# Adding an intercept *** This is requried ***. Don't forget this step.
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model       = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test) # make the predictions by the model

print(model.summary())
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))



# Example 3: Linear Regression with Scaling
import pandas as pd
import numpy as np
from sklearn import datasets
from   sklearn.model_selection import train_test_split
import statsmodels.api         as sm
import numpy                   as np
from   sklearn                 import metrics

wine = datasets.load_wine()
dataset = pd.DataFrame(
    data=np.c_[wine['data'], wine['target']],
    columns=wine['feature_names'] + ['target']
)

# Create copy to prevent overwrite.
X = dataset.copy()
del X['target']         # Remove target variable
del X['hue']            # Remove unwanted features
del X['ash']
del X['magnesium']
del X['malic_acid']
del X['alcohol']

y = dataset['target']

# Adding an intercept *** This is requried ***. Don't forget this step.
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import RobustScaler
sc_x     = RobustScaler()
X_train_scaled = sc_x.fit_transform(X_train)
X_test_scaled  = sc_x.transform(X_test)

sc_y           = RobustScaler()
y_train_scaled = sc_y.fit_transform(np.array(y_train).reshape(-1,1))

model       = sm.OLS(y_train_scaled, X_train_scaled).fit()
unscaledPredictions = model.predict(X_test_scaled) # make predictions
predictions = sc_y.inverse_transform(np.array(unscaledPredictions).reshape(-1,1))

print(model.summary())
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))




# Example 4: Drawing a ROC Curve
import pandas  as pd
import numpy   as np

# Setup data.
candidates = {'gmat': [780, 750, 690, 710, 680, 730, 690, 720,
                       740, 690, 610, 690, 710, 680, 770, 610, 580, 650, 540, 590, 620,
                       600, 550, 550, 570, 670, 660, 580, 650, 660, 640, 620, 660, 660,
                       680, 650, 670, 580, 590, 690],
              'gpa': [4, 3.9, 3.3, 3.7, 3.9, 3.7, 2.3, 3.3,
                      3.3, 1.7, 2.7, 3.7, 3.7, 3.3, 3.3, 3, 2.7, 3.7, 2.7, 2.3,
                      3.3, 2, 2.3, 2.7, 3, 3.3, 3.7, 2.3, 3.7, 3.3, 3, 2.7, 4,
                      3.3, 3.3, 2.3, 2.7, 3.3, 1.7, 3.7],
              'work_experience': [3, 4, 3, 5, 4, 6, 1, 4, 5,
                                  1, 3, 5, 6, 4, 3, 1, 4, 6, 2, 3, 2, 1, 4, 1, 2, 6, 4, 2, 6, 5, 1, 2, 4, 6,
                                  5, 1, 2, 1, 4, 5],
              'admitted': [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1,
                           1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0,
                           0, 0, 1]}

df = pd.DataFrame(candidates, columns=['gmat', 'gpa',
                                       'work_experience', 'admitted'])
# Separate into x and y values.
predictorVariables = ['gmat', 'gpa', 'work_experience']
X = df[predictorVariables]
y = df['admitted']

# Re-assign X with significant columns only after chi-square test.
X = df[['gmat', 'work_experience']]

############## SECTION B start #################################################
# Split data.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=0)
############## SECTION B end   #################################################

# Perform logistic regression.
from sklearn.linear_model import LogisticRegression

logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear',
                                   random_state=0)
logisticModel.fit(X_train, y_train)
y_pred = logisticModel.predict(X_test)
print(y_pred)

# Show precision, recall and F1 scores for all classes.
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average=None)
recall = metrics.recall_score(y_test, y_pred, average=None)
f1 = metrics.f1_score(y_test, y_pred, average=None)

print("Accuracy:  " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1: " + str(f1))

import matplotlib.pyplot       as plt
from sklearn.metrics           import roc_curve
from sklearn.metrics           import roc_auc_score

auc = roc_auc_score(y_test, y_prob[:, 1],)
print('Logistic: ROC AUC=%.3f' % (auc))

# calculate roc curves
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_prob[:, 1])
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.plot([0,1], [0,1], '--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

import pandas as pd
data = {"Probability Admitted": y_prob[:, 1],
        "Actual Outcome": y_test}
dfEvaluate = pd.DataFrame(data=data)
print(dfEvaluate)
