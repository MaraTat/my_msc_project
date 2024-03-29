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

##### preparing the encoding matrix #######
blosum = pd.read_csv('D:/MSc_project/func_testing/BLOSUM62', skiprows=6, sep='\s+', index_col=0)
blosum = blosum.reset_index(drop=True)
# blosum
def blosum_encode(seq):
    #encode a peptide into blosum features
    # s=list(seq)
    x = pd.DataFrame([blosum[i] for i in seq]).reset_index(drop=True)
    # show_matrix(x)
    m = x.values.flatten()
    return m

############# developing XGBoost below ######################

fullset21 = pd.read_pickle('D:/MSc_project/datapickles/fullset_21.pkl')

grid_mcc_xgb_B62 = []
grid_params_xgb_B62 =[]

df = fullset21

X = df.drop(['posmiddlePRO', 'omegaPRO', 'cistrans'], axis=1)
Xcols = [c for c in X.columns]
X_enc = X[Xcols].apply(lambda k: pd.Series(blosum_encode(k)),axis=1) # encoding the X using the blosum matrix and function above
y = pd.get_dummies(df['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'}) # creating dummies for cis trans conformation using the rename() method
X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size= 0.1)

xgb = XGBClassifier(tree_method= 'exact', use_label_encoder=False, verbosity=2,objective='binary:logistic')
paramgrid = {'n_estimators': [100,200,500,1000]} # performing a grid search to find the best parameters
grid = GridSearchCV(xgb, paramgrid, cv=5, verbose=3) # grid is the model, could be KNN or random forests
grid.fit(X_train, y_train.values.ravel()) # fitting the model
xgb_predictions = grid.predict(X_test) # getting the preds


filename_xgb = 'D:/MSc_project/func_testing/new_sets/models/xgb_B62_win21.sav' # saving the model
joblib.dump(grid, filename_xgb)

grid_mcc_xgb_B62.append(metrics.matthews_corrcoef(y_test, xgb_predictions).round(3))
grid_params_xgb_B62.append(grid.best_params_)
print(metrics.plot_confusion_matrix(grid,X_test,y_test, display_labels=['cis', 'trans'], cmap=plt.cm.Blues, normalize='true'))
print(metrics.plot_roc_curve(grid, X_test, y_test))


