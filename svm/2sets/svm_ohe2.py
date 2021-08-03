import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn import metrics

# remember to run the cleanup functions to get the dataframes and use them here
# trainme2 = cleanupmydata(training2)
# testme = cleanupmydata(testing)
# trainpro = prowindows(trainme2,9)
# testpro = prowindows(testme,9)

ohe = OneHotEncoder() # creating the onehotencoder for the data       
X_train = trainpro.drop(['posmiddlePRO', 'omegaPRO', 'cistrans'], axis=1)
X_traincols = [c for c in X_train.columns]
X_train_enc = pd.DataFrame(ohe.fit_transform(X_train[X_traincols]).toarray())
y_train = pd.get_dummies(trainpro['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'}) # creating dummies for cis trans conformation using the rename() method

df = testpro
X_test = df.drop(['posmiddlePRO', 'omegaPRO', 'cistrans'], axis=1)
X_testcols = [c for c in X_test.columns]
X_test_enc = pd.DataFrame(ohe.fit_transform(X_test[X_testcols]).toarray())
y_test = pd.get_dummies(df['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'}) # creating dummies for cis trans conformation using the rename() method

##################

paramgrid = {'C': [0.1,1,10,100], 'gamma':[10,1,0.1,0.01]} # performing a grid search to find the best parameters
grid = GridSearchCV(SVC(), paramgrid, cv=5, verbose=3)

grid.fit(X_train_enc, y_train.values.ravel())
svm_predictions = grid.predict(X_test_enc)

print(metrics.plot_confusion_matrix(grid,X_test_enc,y_test, display_labels=['cis', 'trans'], cmap=plt.cm.Blues, normalize='true'))
print(metrics.plot_roc_curve(grid, X_test_enc, y_test))
print(metrics.matthews_corrcoef(y_test, svm_predictions).round(3))
print(grid.best_params_)


