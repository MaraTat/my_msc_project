import pandas as pd

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
