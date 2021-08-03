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

# remember to locate the correct blosum file
# BLOSUM62 = https://github.com/MaraTat/my_msc_project/blob/master/matrices/BLOSUM62
# BLOSUM45 = https://github.com/MaraTat/my_msc_project/blob/master/matrices/BLOSUM45
# BLOSUM80 = https://github.com/MaraTat/my_msc_project/blob/master/matrices/BLOSUM80

blosum = pd.read_csv(BLOSUM62, skiprows=6, sep='\s+', index_col=0)
blosum = blosum.reset_index(drop=True)    
def blosum_encode(seq):
    x = pd.DataFrame([blosum[i] for i in seq]).reset_index(drop=True)
    m = x.values.flatten()
    return m

X_train = trainpro.drop(['posmiddlePRO', 'omegaPRO', 'cistrans'], axis=1)
X_traincols = [c for c in X_train.columns]
X_train_enc = X_train[X_traincols].apply(lambda k: pd.Series(blosum_encode(k)),axis=1) # encoding the X using the blosum matrix and function below
y_train = pd.get_dummies(trainpro['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'}) # creating dummies for cis trans conformation using the rename() method

df = testpro
X_test = df.drop(['posmiddlePRO', 'omegaPRO', 'cistrans'], axis=1)
X_testcols = [c for c in X_test.columns]
X_test_enc = X_test[X_testcols].apply(lambda k: pd.Series(blosum_encode(k)),axis=1) # encoding the X using the blosum matrix and function below
y_test = pd.get_dummies(df['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'}) # creating dummies for cis trans conformation using the rename() method


paramgrid = {'C': [0.1,1,10,100], 'gamma':[10,1,0.1,0.01]} # performing a grid search to find the best parameters
grid = GridSearchCV(SVC(), paramgrid, cv=5, verbose=3)

grid.fit(X_train_enc, y_train.values.ravel())
svm_predictions = grid.predict(X_test_enc)

print(metrics.plot_confusion_matrix(grid,X_test_enc,y_test, display_labels=['cis', 'trans'], cmap=plt.cm.Blues, normalize='true'))
print(metrics.plot_roc_curve(grid, X_test_enc, y_test))
print(metrics.matthews_corrcoef(y_test, svm_predictions).round(3))
print(grid.best_params_)





