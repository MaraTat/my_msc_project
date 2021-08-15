import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from glob import iglob


testpath = 'D:/MSc_project/func_testing/new_sets/torsions_pisces_small/*.txt'

trainpath = 'D:/MSc_project/func_testing/new_sets/torsions_pisces_big/*.txt'


training = pd.concat((pd.read_csv(f, skiprows=(0,1), header=None, delimiter='\s+') 
                    for f in iglob(trainpath, recursive=False)), ignore_index=True)

testing = pd.concat((pd.read_csv(f, skiprows=(0,1), header=None, delimiter='\s+') 
                    for f in iglob(testpath, recursive=False)), ignore_index=True)


testme = cleanupmydata(testing)
trainme = cleanupmydata(training)

trainpro = prowindows(trainme, 11)
testpro = prowindows(testme,11)


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib


##### training the model below - same set train-test-split #######
blosum = pd.read_csv('BLOSUM45', skiprows=6, sep='\s+', index_col=0)
blosum = blosum.reset_index(drop=True)
# blosum
def blosum_encode(seq):
    #encode a peptide into blosum features
    # s=list(seq)
    x = pd.DataFrame([blosum[i] for i in seq]).reset_index(drop=True)
    # show_matrix(x)
    m = x.values.flatten()
    return m

df = trainpro

X = df.drop(['posmiddlePRO', 'omegaPRO', 'cistrans'], axis=1)
Xcols = [c for c in X.columns]
y = pd.get_dummies(df['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'}) # creating dummies for cis trans conformation using the rename() method

X_enc = X[Xcols].apply(lambda k: pd.Series(blosum_encode(k)),axis=1) # encoding the X using the blosum matrix and function below

X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size= 0.4, random_state=101)

########### developing Random Forests model #############

rfc = RandomForestClassifier(n_estimators=100, oob_score=True, verbose=3)
rfc.fit(X_train, y_train.values.ravel())
rfc_predictions = rfc.predict(X_test)

print(metrics.plot_confusion_matrix(rfc,X_test,y_test, display_labels=['cis', 'trans'], cmap=plt.cm.Blues, normalize='true'))
print(metrics.plot_roc_curve(rfc, X_test, y_test))
print(metrics.matthews_corrcoef(y_test, rfc_predictions).round(3))
print(rfc.oob_score_.round(3))

    
filename = 'D:/MSc_project/func_testing/new_sets/models/randfor_B62_pisces_big_win11.sav'
joblib.dump(rfc, filename)


######### encoding the test set ###################
## pisces small ###
df = testpro

X_test = df.drop(['posmiddlePRO', 'omegaPRO', 'cistrans'], axis=1)
X_testcols = [c for c in X_test.columns]
X_test_enc = X_test[X_testcols].apply(lambda k: pd.Series(blosum_encode(k)),axis=1) # encoding the X using the blosum matrix and function below
y_test_enc = pd.get_dummies(df['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'}) # creating dummies for cis trans conformation using the rename() method

####### applying the model on the test set ########

model = joblib.load(filename)
result_test = model.score(X_test_enc, y_test_enc)
print(result_test.round(3))

