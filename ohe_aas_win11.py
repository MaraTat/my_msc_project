# -*- coding: utf-8 -*-
"""
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
# eliminating unecessary information --> Resnum, PHI, PSI from the df

aas =['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET',
      'PHE','PRO','SER','THR','TRP','TYR','VAL']

aaletters = ['A','R','N','D','C','Q','E','G','H','I',
             'L','K','M','F','P','S','T','W','Y','V']

aadict = dict(zip(aas, aaletters))

# eliminating unecessary information --> Resnum, PHI, PSI from the df
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

# this indexing below works --> try to get it into a loop and set up boundaries based on window size
# maybe even try to use a smaller df
p = prot_df.loc[(prot_df['Resnam']=='P')]
i = list(range(len(p)))

indices = []
for n in i:
    idx = prot_df.index.get_loc(prot_df.loc[(prot_df['Resnam']=='P')].index[n])
    indices.append(idx)

print(indices)
idx

# setting a couple of variables to be able to change the window size outside the fn
# now working with window size=5
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

# there's problem when Prolines are very close to the beginning of the protein 
# when it comes to bigger windows e.g. 11 etc 
# when P[index]<k --> I am dropping them for now
protsdf.dropna(axis=0, inplace=True)


# note to self: iterating over dfs is not recommended because it takes too long 
# vectors are the recommended way to itterate over dfs
# masks are a vectorised method to iterate over dfs
# below adding a column with the conformation of the Proline residue using masks
maskcis = (protsdf['omegaPRO']<30) & (protsdf['omegaPRO']>-30)
protsdf.loc[maskcis, 'cistrans'] = 'cis'
masktrans = protsdf['cistrans'] != 'cis'
protsdf.loc[masktrans, 'cistrans'] = 'trans'

# below using sci-kit learn and SVM to run predictions

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn import metrics

# I am just making a copy of the df to use with the ML so that I don't run the code again
df = protsdf.copy()

X = df.drop(['middlePRO', 'omegaPRO', 'cistrans'], axis=1)
# creating dummies for cis trans conformation using the rename() method
# note to self: because we drop_first the inplace=True in the rename doesn't work
# also rename() requires a dictionary
y = pd.get_dummies(df['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'})

# creating the onehotencoder for the data
ohe = OneHotEncoder()
X_enc = pd.DataFrame(ohe.fit_transform(X[['pos1','pos2','pos4','pos5']]).toarray())

X_train, x_test, y_train, y_test = train_test_split(X_enc, y, test_size= 0.4, random_state=101)

col_trans = make_column_transformer((OneHotEncoder(), ['pos1', 'pos2', 'pos3', 'pos4', 'pos5', 
           'pos7', 'pos8', 'pos9', 'pos10', 'pos11']), remainder='passthrough')

# performing a grid search to find the best parameters (see if it's necessary to change the numbers)
paramgrid = {'C': [0.1,1,10,100], 'gamma':[10,1,0.1,0.01]}
grid = GridSearchCV(SVC(), paramgrid, verbose=3)
pipe = make_pipeline(col_trans, grid)

pipe.fit(X_train, y_train)
predictions = pipe.predict(x_test)
predictions

print(metrics.confusion_matrix(y_test, predictions))
class_names = ['cis', 'trans']
metrics.plot_confusion_matrix(pipe,x_test,y_test, display_labels=class_names, cmap=plt.cm.Blues, normalize='true')
print(metrics.classification_report(y_test, predictions))
print(grid.best_params_)
print(grid.best_estimator_)
print(grid.best_score_)
metrics.plot_roc_curve(pipe, x_test, y_test)





