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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

######

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

######

error_rate = []
neighbors = []
for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_enc,y_train.values.ravel())
    pred_i = knn.predict(X_test_enc)
    error_rate.append(np.mean(pred_i != y_test.values.ravel()))
    neighbors.append(i)

knns = dict(zip(neighbors, error_rate))
num_neighbors = min(knns, key=knns.get)

knn = KNeighborsClassifier(n_neighbors=num_neighbors)
knn.fit(X_train_enc,y_train.values.ravel())
knn_predictions = knn.predict(X_test_enc)

print(metrics.matthews_corrcoef(y_test, knn_predictions).round(3))
print(metrics.plot_confusion_matrix(knn,X_test_enc,y_test, display_labels=['cis', 'trans'], cmap=plt.cm.Blues, normalize='true'))
print(metrics.plot_roc_curve(knn, X_test_enc, y_test))
print(num_neighbors)

