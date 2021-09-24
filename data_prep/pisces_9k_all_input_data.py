import pandas as pd
import numpy as np

fullset = pd.read_pickle('D:/MSc_project/datapickles/fullset.pkl')

ciswindf = fullset.loc[(fullset['cistrans']=='cis')] # 10420 cis windows
transwindf = fullset.loc[(fullset['cistrans']=='trans')] # 216879 trans windows

# copies of the master input dfs
ciswindf21 = ciswindf.copy()
transwindf21 = transwindf.copy()

# creating a validation dataset 
ciswindf21_val = ciswindf21.sample(frac=0.1) # 1042
# ciswindf21_val_pkl = pd.to_pickle(ciswindf21_val, 'D:/MSc_project/datapickles/ciswindf21_val.pkl')
for i in ciswindf21_val.index:
    if i in ciswindf21.index:
        ciswindf21.drop(index=i, inplace=True)

transwindf21_val = transwindf21.sample(n=len(ciswindf21_val)) # 1042
# transwindf21_val_pkl = pd.to_pickle(transwindf21_val, 'D:/MSc_project/datapickles/transwindf21_val.pkl')
for i in transwindf21_val.index:
    if i in transwindf21.index:
        transwindf21.drop(index=i, inplace=True)

# the main validation set to trim 
validation_21 = transwindf21_val.append(ciswindf21_val)
fullset_21 = ciswindf21.append(transwindf21)

# slicing the dfs to create the different window datasets
allmixdfs19 = [] # change that name 
for f in allmixdfs21: # change the source df
    last2 = f[['omegaPRO', 'cistrans']]
    maindf = f.iloc[:, -(len(f.columns)-1):-3]
    addme = pd.concat([maindf,last2], axis= 1)
    allmixdfs19.append(addme) # change the destination df name

# slicing the dfs to create the different validation datasets
dfv = validation_21
last2 = dfv[['omegaPRO', 'cistrans']] # change the source df
maindf = dfv.iloc[:, -(len(dfv.columns)-1):-3]
validation_19 = pd.concat([maindf,last2], axis= 1) # change the destination df name

# same process for full set
f = fullset21 # change the source df
last2 = f[['omegaPRO', 'cistrans']]
maindf = f.iloc[:, -(len(f.columns)-1):-3]
addme = pd.concat([maindf,last2], axis= 1)
fullset19 = addme # change the destination df name







