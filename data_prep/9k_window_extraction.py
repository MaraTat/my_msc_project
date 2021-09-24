import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from glob import iglob
import random

trainpath = 'D:/MSc_project/func_testing/new_sets/pisces_9k/*.txt' # personal folder on my laptop
 
training = pd.concat((pd.read_csv(f, skiprows=(0,1), header=None, delimiter='\s+') 
                    for f in iglob(trainpath, recursive=False)), ignore_index=True)

def cleanupmydata(protdf):
    """
    Parameters
    ----------
    protdf : TYPE .txt file which has been generated through another function pdbtorsions outside of this function
        DESCRIPTION.

    Returns
    -------
    None.

    """
    try:
        if not isinstance(protdf, pd.DataFrame):
            with open (protdf, 'rt') as file:
                dftest = pd.read_csv(file, skiprows=(0,1), header=None, delimiter='\s+')
        else:
            dftest = protdf
            
        dftest.columns = ['Resnum', 'Resnam', 'PHI', 'PSI', 'OMEGA'] # eliminating unecessary information --> Resnum, PHI, PSI from the df
        prot_df = dftest[['Resnam', 'OMEGA']] # eliminating unecessary information --> Resnum, PHI, PSI from the df
        prot_df = prot_df.loc[(prot_df['OMEGA'] >= -360) & (prot_df['OMEGA'] <= 360)] # getting rid of weird values like 9999.000
        
        aas =['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET',
              'PHE','PRO','SER','THR','TRP','TYR','VAL','5HP','ABA','AIB','BAL','BMT','CEA','CGU','CME','CSD','CSO',
                      'CSS','CSW','CSX','CXM','DAL','DAR','DCY','DGL','DGN','DHI','DIL',
                      'DIV','DLE','DLY','DPN','DPR','DSG','DSN','DSP','DTH','DTR','DTY',
                      'DVA','FME','HYP','KCX','LLP','MLE','MVA','NLE','OCS','ORN',
                      'PCA','PTR','SAR','SEP','STY','TPO','TPQ','TYS','MSE']
    
        aaletters = ['A','R','N','D','C','Q','E','G','H','I',
                  'L','K','M','F','P','S','T','W','Y','V',
                  'E','A','A','A','T','C','E','C','C','C','C','C','C','M','A','R','C',
                            'H','I','V','L','K','F','P','N','S','D','W','Y','T','V','M','P','K',
                            'K','L','C','A','E','Y','G','S','Y','T','F','Y','T','A','A','Y','M']
    
        
        prot_df = prot_df[prot_df['Resnam'].isin(aas)] # getting rid of residues that are not aas but just active groups
    
        prot_df.replace(to_replace=aas, value= aaletters, inplace=True) # replacing aas with their single letters
    
        prot_df = prot_df.reset_index(drop=True)
        return prot_df
    
    except:
        return print('Please use a dataframe with your data or run the function pdbtorsions on the raw .ent pdb file and generate a .txt file')

      def prowindows(prot_df, window):
    """
    

    Parameters
    ----------
    protdf : TYPE
        DESCRIPTION.
    window : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if not isinstance(prot_df, pd.DataFrame):
        return print ('Please use a dataframe as your input file')
    elif window%2 == 0:
        return print ('Please select an odd-number window higher than 5 e.g. 5,7,9,11,13,15,17,19,21')
    else:
        k = int((window-1)/2)
        middlepos = int((window+1)/2)
        # getting the index of the prolines so that I can center the window around them evenly by each side
        p = prot_df.loc[(prot_df['Resnam']=='P')]
        i = list(range(len(p)))
        
        indices = []
        for n in i:
            idx = prot_df.index.get_loc(prot_df.loc[(prot_df['Resnam']=='P')].index[n])
            indices.append(idx)
        
        prot_slices = []
        omegas = []
        for n in indices:
            protslice = prot_df.iloc[n-k: n+middlepos]
            prot_slices.append(protslice['Resnam'].values)
            omegas.append(protslice['OMEGA'].values)
            
        append_cols = 'pos' # adding a prefix to aa column names
        append_omegas = 'omega' # adding a prefix to torsion column names
        
        columns = [str(e) for e in range(1,(window+1))]
        columns = list(map(lambda x: x if x != str(middlepos) else 'middlePRO', columns))
        columns = [append_cols + y for y in columns]
        
        omegacols = [str(e) for e in range(1,(window+1))]
        omegacols = list(map(lambda x: x if x != str(middlepos) else 'middlePRO', omegacols))
        omegacols = [append_omegas + j for j in omegacols]

        protsdf = pd.DataFrame.from_records(prot_slices, columns = columns)
        omegadf = pd.DataFrame.from_records(data=omegas, columns= omegacols)
        protsdf['omegaPRO'] = omegadf['omegamiddlePRO']
        
        # there's problem when Prolines are very close to the beginning of the protein 
        # when it comes to bigger windows e.g. 11 etc 
        # when P[index]<k --> I am dropping them for now
        protsdf.dropna(axis=0, inplace=True)
        
        maskcis = (protsdf['omegaPRO']<30) & (protsdf['omegaPRO']>-30)
        protsdf.loc[maskcis, 'cistrans'] = 'cis'
        masktrans = protsdf['cistrans'] != 'cis'
        protsdf.loc[masktrans, 'cistrans'] = 'trans'
        protsdf = protsdf.reset_index(drop=True)
        
    return protsdf

trainme = cleanupmydata(training)
trainpro = prowindows(trainme,21) # from the biggest window all the smaller ones will be created through trimming

trainpro_pkl = pd.to_pickle(trainpro, 'D:/MSc_project/datapickles/win21pro_all.pkl')
