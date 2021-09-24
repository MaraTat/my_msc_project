import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

allmixdfs21 = pd.read_pickle('D:/MSc_project/datapickles/allmixdfs21.pkl')

grid_mcc_rfc = []
grid_params_rfc =[]

for v in  allmixdfs21:
    df = v
    
    X = df.drop(['posmiddlePRO', 'omegaPRO', 'cistrans'], axis=1)
    Xcols = [c for c in X.columns]
    ohe = OneHotEncoder() # creating the onehotencoder for the data
    X_enc = pd.DataFrame(ohe.fit_transform(X[Xcols]).toarray())
    y = pd.get_dummies(df['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'}) # creating dummies for cis trans conformation using the rename() method
    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size= 0.1)

    rfc = RandomForestClassifier(oob_score=True, class_weight='balanced', verbose=3)
    paramgrid = {'n_estimators': [100,200,500,1000]} # performing a grid search to find the best number of trees
    grid = GridSearchCV(rfc, paramgrid , cv=5, verbose=3) # grid is the model, could be KNN or random forests
    grid.fit(X_train, y_train.values.ravel()) # fitting the model
    rfc_predictions = grid.predict(X_test) # getting the preds
    
    filename_rfc = 'D:/MSc_project/func_testing/new_sets/models/ohe_rfc_df21_cv5.sav' # saving the model
    joblib.dump(grid, filename_rfc)
    
    grid_mcc_rfc.append(metrics.matthews_corrcoef(y_test, rfc_predictions).round(3))
    grid_params_rfc.append(grid.best_params_)
    
    
    print(metrics.plot_confusion_matrix(grid,X_test,y_test, display_labels=['cis', 'trans'], cmap=plt.cm.Blues, normalize='true'))
    print(metrics.plot_roc_curve(grid, X_test, y_test))









