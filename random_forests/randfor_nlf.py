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

NLF_matrix = 'https://github.com/MaraTat/my_msc_project/blob/master/matrices/NLF_matrix.txt'

nlf = pd.read_csv(NLF_matrix, sep=',', index_col=0)
nlf = nlf.reset_index(drop=True)
def nlf_encode(seq):   
    x = pd.DataFrame([nlf[i] for i in seq]).reset_index(drop=True)  
    m = x.values.flatten()
    return m

X_train = trainpro.drop(['posmiddlePRO', 'omegaPRO', 'cistrans'], axis=1)
X_traincols = [c for c in X_train.columns]
X_train_enc = X_train[X_traincols].apply(lambda k: pd.Series(nlf_encode(k)),axis=1) # encoding the X using the blosum matrix and function below
y_train = pd.get_dummies(trainpro['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'}) # creating dummies for cis trans conformation using the rename() method
   

X_test = df.drop(['posmiddlePRO', 'omegaPRO', 'cistrans'], axis=1)
X_testcols = [c for c in X_test.columns]
X_test_enc = X_test[X_testcols].apply(lambda k: pd.Series(nlf_encode(k)),axis=1) # encoding the X using the blosum matrix and function below
y_test = pd.get_dummies(df['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'}) # creating dummies for cis trans conformation using the rename() method

########################################

rfc = RandomForestClassifier(n_estimators=100, oob_score=True, verbose=3)
rfc.fit(X_train_enc, y_train.values.ravel())
rfc_predictions = rfc.predict(X_test_enc)

print(metrics.matthews_corrcoef(y_test, rfc_predictions).round(3))
print(metrics.plot_confusion_matrix(rfc,X_test_enc,y_test, display_labels=['cis', 'trans'], cmap=plt.cm.Blues, normalize='true'))
print(metrics.plot_roc_curve(rfc, X_test_enc, y_test))
print(rfc.oob_score_.round(3))


