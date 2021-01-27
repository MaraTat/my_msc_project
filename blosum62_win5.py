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

# for the moment this takes the longest time to run in the code, I'll try to improve it
indices = []
for n in i:
    idx = prot_df.index.get_loc(prot_df.loc[(prot_df['Resnam']=='P')].index[n])
    indices.append(idx)

# setting a couple of variables to be able to change the window size outside the fn
k = 2
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

# setting up the encoding based on BLOSUM62 matrix, we can change that later
blosum = np.genfromtxt('BLOSUM62', comments="#", skip_header=7, delimiter=None, 
              names='Index,A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V,B,Z,X,m', dtype=float, 
              usecols=('A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V,B,Z,X,m')) 

def blosum_encode(seq):
    #encode a peptide into blosum features
    s=list(seq)
    x = pd.DataFrame([blosum[i] for i in seq]).reset_index(drop=True)
    # show_matrix(x)
    e = x.values.flatten()
    return e

columns = ['pos1', 'pos2', 'middlePRO', 'pos4', 'pos5']
omegacols = ['omega1', 'omega2', 'middleOmega', 'omega4', 'omega5']
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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics

df = protsdf.copy()

X = df.drop(['middlePRO', 'omegaPRO', 'cistrans'], axis=1)
# encoding the X using the blosum matrix and function below
# this takes a while, I want to work on making a column transformer that works better
X = df[['pos1','pos2','pos4','pos5']].apply(lambda k: pd.Series(blosum_encode(k)),1)

# creating dummies for cis trans conformation using the rename() method
# note to self: because we drop_first the inplace=True in the rename doesn't work
# also rename() requires a dictionary
y = pd.get_dummies(df['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'})

# setting the training and testing data
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.4, random_state=101)

# setting the parameters to be tested
paramgrid = {'C': [0.1,1,10,100], 'gamma':[10,1,0.1,0.01]}
grid = GridSearchCV(SVC(), paramgrid, verbose=3)

grid.fit(X_train, y_train)
predictions = grid.predict(x_test)

print(metrics.confusion_matrix(y_test, predictions))
class_names = ['cis', 'trans']
metrics.plot_confusion_matrix(grid,x_test,y_test, display_labels=class_names, cmap=plt.cm.Blues, normalize='true')
print(metrics.classification_report(y_test, predictions))
print(grid.best_params_)
print(grid.best_estimator_)
print(grid.best_score_)
metrics.plot_roc_curve(grid, x_test, y_test)













