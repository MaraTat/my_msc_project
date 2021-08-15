import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from glob import iglob


trainpath = 'D:/MSc_project/func_testing/new_sets/andrew_abs_torsions/*.txt' # more than 700 proteins, takes long time to extract windows and train

testpath = 'D:/MSc_project/func_testing/new_sets/torsions_pisces_small/*.txt' # 200 proteins


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
df = trainpro

X = df.drop(['posmiddlePRO', 'omegaPRO', 'cistrans'], axis=1)
Xcols = [c for c in X.columns]
ohe = OneHotEncoder() # creating the onehotencoder for the data
X_enc = pd.DataFrame(ohe.fit_transform(X[Xcols]).toarray())
y = pd.get_dummies(df['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'}) # creating dummies for cis trans conformation using the rename() method
X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size= 0.4, random_state=101)

########### developing SVM model #############

paramgrid = {'C': [0.1,1,10,100], 'gamma':[10,1,0.1,0.01]} # performing a grid search to find the best parameters
grid = GridSearchCV(SVC(), paramgrid, cv=5, verbose=3) # grid is the model, could be KNN or random forests

grid.fit(X_train, y_train.values.ravel()) # fitting the model
svm_predictions = grid.predict(X_test) # getting the preds

filename = 'D:/MSc_project/func_testing/new_sets/models/ohe_svm_andrew_abs_win11.sav'
joblib.dump(grid, filename)

# these are just for the training of the model
print(metrics.plot_confusion_matrix(grid,X_test,y_test, display_labels=['cis', 'trans'], cmap=plt.cm.Blues, normalize='true'))
print(metrics.plot_roc_curve(grid, X_test, y_test))
print(metrics.matthews_corrcoef(y_test, svm_predictions).round(3))
print(grid.best_params_)


######### encoding the test set ###################
## pisces small ###
df = testpro

X_test = df.drop(['posmiddlePRO', 'omegaPRO', 'cistrans'], axis=1)
X_testcols = [c for c in X_test.columns]
X_test_enc = pd.DataFrame(ohe.fit_transform(X_test[X_testcols]).toarray())
y_test_enc = pd.get_dummies(df['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'}) # creating dummies for cis trans conformation using the rename() method


####### applying the model on the test set ########

model = joblib.load(filename)
result_test = model.score(X_test_enc, y_test_enc)
print(result_test.round(3))


