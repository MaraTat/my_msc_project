# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 22:43:46 2021

@author: MaraT
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 22:39:14 2021

@author: MaraT
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from glob import iglob

# mypath = 'D:/MSc_project/testprots/*.txt'
mypath = 'D:/MSc_project/testprots/IgGtestprots/*.txt'
# mypath = 'D:/MSc_project/immunoglobulins/torsions/*.txt'

dftest = pd.concat((pd.read_csv(f, skiprows=(0,1), header=None, delimiter='\s+') 
                    for f in iglob(mypath, recursive=False)), ignore_index=True)


dftest.columns = ['Resnum', 'Resnam', 'PHI', 'PSI', 'OMEGA']


prot_df = dftest[['Resnam', 'OMEGA']]
# getting rid of weird values like 9999.000
prot_df = prot_df[prot_df['OMEGA'] != 9999.000]

# With replace it is possible to replace values in a Series or DataFrame without 
# knowing where they occur. it works both with Series and DataFrames
prot_df.replace(to_replace=['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY',
                            'HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR',
                            'TRP','TYR','VAL'],
           value= ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S',
                   'T','W','Y','V'], 
           inplace=True)

# getting rid of non standard aas
# later I'll consider replacing them with their parent aas but not now
prot_df = prot_df[~prot_df['Resnam'].isin(['CSD','HYP','BMT','5HP','ACE','ABA','BAL',
                                           'AIB','NH2','CBX','CSW','OCS','1ZG','DHI',
                                           'DAR','DSG','DSP','DCY','CRO','DGL','DGN',
                                           'DIL','DIV','DLE','DLY','DPN','DPR','DSN',
                                           'DTR','DTY','DVA','FOR','CGU','IVA','KCX',
                                           'CXM','FME','MLE','MVA','NLE','PTR','ORN',
                                           'SEP','TPO','PCA','PVL','SAR','CEA','CSO',
                                           'CSS','CSX','CME','TYS','BOC','TPQ','STY'])]

p = prot_df.loc[(prot_df['Resnam']=='P')]
i = list(range(len(p)))
i2 = set(range(len(p)))

# for the moment this takes the longest time to run in the code, I'll try to improve it
indices = []
for n in i:
    idx = prot_df.index.get_loc(prot_df.loc[(prot_df['Resnam']=='P')].index[n])
    indices.append(idx)

print(len(indices))

# setting a couple of variables to be able to change the window size outside the fn
k = 5
l = k+1

window = k+l
middlepos = int((window+1)/2)
middlepos
prot_slices = []
omegas = []
for n in indices:
    protslice = prot_df.iloc[n-k: n+l]
    prot_slices.append(protslice['Resnam'].values)
    omegas.append(protslice['OMEGA'].values)


columns = ['pos1', 'pos2', 'pos3', 'pos4', 'pos5', 'middlePRO', 
           'pos7', 'pos8', 'pos9', 'pos10', 'pos11']
omegacols = ['omega1', 'omega2', 'omega3','omega4', 'omega5','middleOmega',
             'omega7', 'omega8', 'omega9', 'omega10', 'omega11']
protsdf = pd.DataFrame.from_records(prot_slices, columns = columns)
omegadf = pd.DataFrame.from_records(data=omegas, columns= omegacols)

protsdf['omegaPRO'] = omegadf['middleOmega']
protsdf.dropna(axis=0, inplace=True)


# note to self: iterating over dfs is not recommended because it takes too long 
# vectors are the recommended way to itterate over dfs
# masks are a vectorised method to iterate over dfs
maskcis = (protsdf['omegaPRO']<30) & (protsdf['omegaPRO']>-30)
protsdf.loc[maskcis, 'cistrans'] = 'cis'
masktrans = protsdf['cistrans'] != 'cis'
protsdf.loc[masktrans, 'cistrans'] = 'trans'

# sci-kit learn for prediction

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = protsdf.copy()

# below the ohe with window = 11

X = df.drop(['middlePRO', 'omegaPRO', 'cistrans'], axis=1)
y = pd.get_dummies(df['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'})
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.4, random_state=101)

col_trans = make_column_transformer((OneHotEncoder(), ['pos1','pos2','pos3','pos4','pos5','pos7','pos8','pos9','pos10','pos11']), remainder='passthrough')
knn = KNeighborsClassifier(n_neighbors=1)

pipe = make_pipeline(col_trans, knn)

pipe.fit(X_train, y_train.values.ravel())
predictions = pipe.predict(x_test)
# predictions

# I don't know why the error rate is everywhere the same
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    pipe.fit(X_train,y_train.values.ravel())
    pred_i = pipe.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test.values.ravel()))

plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize='10')
plt.title('Error rate vs k value')
plt.xlabel('K')
plt.ylabel('Error rate')
plt.show()


print(metrics.confusion_matrix(y_test, predictions))
class_names = ['cis', 'trans']
metrics.plot_confusion_matrix(pipe,x_test,y_test, display_labels=class_names, cmap=plt.cm.Blues, normalize='true')
print(metrics.classification_report(y_test, predictions))
metrics.plot_roc_curve(pipe, x_test, y_test)




