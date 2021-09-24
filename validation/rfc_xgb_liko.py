import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

allmixdfs21 = pd.read_pickle('D:/MSc_project/datapickles/allmixdfs21.pkl')
allmixdfs19 = pd.read_pickle('D:/MSc_project/datapickles/allmixdfs19.pkl')
fullset21 = pd.read_pickle('D:/MSc_project/datapickles/fullset_21.pkl')
fullset19 = pd.read_pickle('D:/MSc_project/datapickles/fullset19.pkl')
fullset17 = pd.read_pickle('D:/MSc_project/datapickles/fullset17.pkl')

validation_21 = pd.read_pickle('D:/MSc_project/datapickles/validation_21.pkl')
validation_19 = pd.read_pickle('D:/MSc_project/datapickles/validation_19.pkl')
validation_17 = pd.read_pickle('D:/MSc_project/datapickles/validation_17.pkl')
# imbal_val_21 = pd.read_pickle('D:/MSc_project/datapickles/imbal_val_21.pkl')
# imbal_val_19 = pd.read_pickle('D:/MSc_project/datapickles/imbal_val_19.pkl')
# imbal_val_17 = pd.read_pickle('D:/MSc_project/datapickles/imbal_val_17.pkl')

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

#### validation for LiKoehl matrix #####################
likoehl = pd.read_csv('D:/MSc_project/LiKoehl.csv', sep=',', index_col=0)
likoehl = likoehl.reset_index(drop=True)
def liko_encode(seq):   
    x = pd.DataFrame([likoehl[i] for i in seq]).reset_index(drop=True)  
    m = x.values.flatten()
    return m

###############################################################
dfv = validation_19

X_test = dfv.drop(['posmiddlePRO', 'omegaPRO', 'cistrans'], axis=1)
X_testcols = [c for c in X_test.columns]
X_test_enc = X_test[X_testcols].apply(lambda k: pd.Series(liko_encode(k)),axis=1) # encoding the X using the blosum matrix and function below
y_test_enc = pd.get_dummies(dfv['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'}) # creating dummies for cis trans conformation using the rename() method

rfc_liko_mcc = []
for v in  allmixdfs19:
    df = v
     
    X_train = df.drop(['posmiddlePRO', 'omegaPRO', 'cistrans'], axis=1)
    X_traincols = [c for c in X_train.columns]
    X_train_enc = X_train[X_traincols].apply(lambda k: pd.Series(liko_encode(k)),axis=1) # encoding the X using the blosum matrix and function below
    y_train = pd.get_dummies(df['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'}) # creating dummies for cis trans conformation using the rename() method
    
    # dfv = validation_21
    # X_test = dfv.drop(['posmiddlePRO', 'omegaPRO', 'cistrans'], axis=1)
    # X_testcols = [c for c in X_test.columns]
    # X_test_enc = X_test[X_testcols].apply(lambda k: pd.Series(blosum_encode(k)),axis=1) # encoding the X using the blosum matrix and function below
    # y_test_enc = pd.get_dummies(dfv['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'}) # creating dummies for cis trans conformation using the rename() method

    rfc = RandomForestClassifier(n_estimators=500, oob_score=True, verbose=3)
    # n_scores = cross_val_score(rfc, X_enc, y, scoring='accuracy', cv=5, n_jobs=1)
    rfc.fit(X_train_enc, y_train.values.ravel())
    rfc_predictions = rfc.predict(X_test_enc)
    
    rfc_liko_mcc.append(metrics.matthews_corrcoef(y_test_enc, rfc_predictions).round(3))
    print(metrics.plot_confusion_matrix(rfc,X_test_enc,y_test_enc, display_labels=['cis', 'trans'], cmap=plt.cm.Blues, normalize='true'))
    print(metrics.plot_roc_curve(rfc, X_test_enc, y_test_enc))
    print(metrics.classification_report(y_test_enc, rfc_predictions.round(3), target_names=['cis', 'trans'] ))

############# XGBoost below ######################

xgb_liko_mcc = []

df = fullset17

X_train = df.drop(['posmiddlePRO', 'omegaPRO', 'cistrans'], axis=1)
X_traincols = [c for c in X_train.columns]
X_train_enc = X_train[X_traincols].apply(lambda k: pd.Series(liko_encode(k)),axis=1) # encoding the X using the blosum matrix and function below
y_train = pd.get_dummies(df['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'}) # creating dummies for cis trans conformation using the rename() method

dfv = validation_17

X_test = dfv.drop(['posmiddlePRO', 'omegaPRO', 'cistrans'], axis=1)
X_testcols = [c for c in X_test.columns]
X_test_enc = X_test[X_testcols].apply(lambda k: pd.Series(liko_encode(k)),axis=1) # encoding the X using the blosum matrix and function below
y_test_enc = pd.get_dummies(dfv['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'}) # creating dummies for cis trans conformation using the rename() method

xgb = XGBClassifier(n_estimators=1000, tree_method= 'exact', use_label_encoder=False, verbosity=2,objective='binary:logistic')
xgb.fit(X_train_enc, y_train.values.ravel())
xgb_predictions = xgb.predict(X_test_enc)
xgb_accur = metrics.accuracy_score(y_test_enc, xgb_predictions)

xgb_liko_mcc.append(metrics.matthews_corrcoef(y_test_enc, xgb_predictions).round(3))
print(metrics.plot_confusion_matrix(xgb,X_test_enc,y_test_enc, display_labels=['cis', 'trans'], cmap=plt.cm.Blues, normalize='true'))
print(metrics.plot_roc_curve(xgb, X_test_enc, y_test_enc))
print(metrics.classification_report(y_test_enc, xgb_predictions.round(3),target_names=['cis', 'trans']))





