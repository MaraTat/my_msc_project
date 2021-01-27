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

prot_df = dftest[['Resnam', 'OMEGA']]
# getting rid of weird values like 9999.000
prot_df = prot_df[prot_df['OMEGA'] != 9999.000]

# With replace it is possible to replace values in a Series or DataFrame without 
# knowing where they occur. it works both with Series and DataFrames
prot_df.replace(to_replace=['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY',
                            'HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR',
                            'TRP','TYR','VAL'],
           value= ['Aliphatic','Basic','Amidic','Acidic','Sulphur','Amidic','Acidic',
                   'Aliphatic','Basic','Aliphatic','Aliphatic','Basic','Sulphur',
                   'Aromatic','Aliphatic_PRO','Hydroxylic','Hydroxylic','Aromatic',
                   'Aromatic','Aliphatic'], 
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

p = prot_df.loc[(prot_df['Resnam']=='Aliphatic_PRO')]
p

i = list(range(len(p)))
i

indices = []
for n in i:
    idx = prot_df.index.get_loc(prot_df.loc[(prot_df['Resnam']=='Aliphatic_PRO')].index[n])
    indices.append(idx)

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

columns = ['pos1', 'pos2', 'middlePRO', 'pos4', 'pos5']
omegacols = ['omega1', 'omega2', 'middleOmega', 'omega4', 'omega5']
protsdf = pd.DataFrame.from_records(prot_slices, columns = columns)
omegadf = pd.DataFrame.from_records(data=omegas, columns= omegacols)

protsdf['omegaPRO'] = omegadf['middleOmega']

maskcis = (protsdf['omegaPRO']<30) & (protsdf['omegaPRO']>-30)
protsdf.loc[maskcis, 'cistrans'] = 'cis'
masktrans = protsdf['cistrans'] != 'cis'
protsdf.loc[masktrans, 'cistrans'] = 'trans'

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
# from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import make_pipeline
# from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, auc, roc_curve, recall_score
from sklearn import metrics

df = protsdf.copy()
# df.head()

X = df.drop(['middlePRO', 'omegaPRO', 'cistrans'], axis=1)
y = pd.get_dummies(df['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'})

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.4, random_state=101)

col_trans = make_column_transformer((OneHotEncoder(), ['pos1','pos2','pos4','pos5']), remainder='passthrough')

paramgrid = {'C': [0.1,1,10,100], 'gamma':[10,1,0.1,0.01]}
grid = GridSearchCV(SVC(), paramgrid, verbose=3)
pipe = make_pipeline(col_trans, grid)

pipe.fit(X_train, y_train)
predictions = pipe.predict(x_test)
# predictions

print(metrics.confusion_matrix(y_test, predictions))
class_names = ['cis', 'trans']
metrics.plot_confusion_matrix(pipe,x_test,y_test, display_labels=class_names, cmap=plt.cm.Blues, normalize='true')
metrics.plot_confusion_matrix(pipe,x_test,y_test, display_labels=class_names, cmap=plt.cm.Blues)
print(metrics.classification_report(y_test, predictions))
print(grid.best_params_)
print(grid.best_estimator_)
print(grid.best_score_)
metrics.plot_roc_curve(pipe, x_test, y_test)






